"""Run self-evolving data synthesis across connected Android devices."""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import adbutils
from tqdm import tqdm

from ..config import get_data_synthesis_config, resolve_path
from .gui_init import run_gui_task


config = get_data_synthesis_config()


def get_device_serial_list() -> list[str]:
    return [device.serial for device in adbutils.adb.device_list()]


def _load_packages(path: str) -> list[str]:
    package_path = resolve_path(path)
    if not package_path.is_file():
        raise FileNotFoundError(f"Package list not found: {package_path}")
    with package_path.open(encoding="utf-8") as handle:
        packages = json.load(handle)
    if not isinstance(packages, list) or not all(isinstance(item, str) for item in packages):
        raise ValueError(f"Package list must be a JSON array of strings: {package_path}")
    return packages


def get_package_names() -> list[str]:
    package_type = config.data_collect.pkg_type
    android_world = _load_packages(config.data_collect.package_path_aw)
    extra = _load_packages(config.data_collect.package_path_extra)
    if package_type == "aw":
        return android_world
    if package_type == "extra":
        return extra
    if package_type == "all":
        return list(dict.fromkeys(android_world + extra))
    raise ValueError(f"Unsupported pkg_type {package_type!r}; use 'aw', 'extra', or 'all'.")


async def run_package(
    package_name: str, device_serial: str, max_retries: int, executor: ProcessPoolExecutor
) -> bool:
    log_dir = resolve_path(config.data_collect.log_dir)
    data_root = resolve_path(config.data_collect.data_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{package_name}.log"

    for attempt in range(1, max_retries + 1):
        try:
            return_code = await asyncio.get_running_loop().run_in_executor(
                executor,
                partial(
                    run_gui_task,
                    device=device_serial,
                    package=package_name,
                    data_root_dir=str(data_root),
                    log_file=str(log_file),
                ),
            )
            if return_code == 0:
                return True
        except Exception as exc:
            print(f"[Device {device_serial}] {package_name} failed: {exc}")
            traceback.print_exc()
        if attempt < max_retries:
            await asyncio.sleep(1)
    return False


async def device_worker(
    device_serial: str,
    task_queue: asyncio.Queue[str],
    max_retries: int,
    semaphore: asyncio.Semaphore,
    progress: tqdm,
    executor: ProcessPoolExecutor,
) -> list[str]:
    failed = []
    while True:
        try:
            package_name = task_queue.get_nowait()
        except asyncio.QueueEmpty:
            return failed
        async with semaphore:
            if not await run_package(package_name, device_serial, max_retries, executor):
                failed.append(package_name)
        progress.update(1)


async def main() -> None:
    devices = get_device_serial_list()
    if not devices:
        raise RuntimeError("No Android devices found. Check `adb devices` before starting synthesis.")

    packages = get_package_names()
    if not packages:
        raise RuntimeError("The configured package list is empty.")

    runtime = config.get("runtime", {})
    max_retries = int(runtime.get("max_retries", 2))
    max_concurrent = int(runtime.get("max_concurrent_devices", len(devices)))
    if max_retries < 1 or max_concurrent < 1:
        raise ValueError("max_retries and max_concurrent_devices must be at least 1")
    semaphore = asyncio.Semaphore(max_concurrent)
    queue: asyncio.Queue[str] = asyncio.Queue()
    for package in packages:
        queue.put_nowait(package)

    # run_gui_task redirects stdout/stderr, so each task needs its own process.
    with ProcessPoolExecutor(
        max_workers=min(len(devices), max_concurrent),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor, tqdm(total=len(packages), desc="processing") as progress:
        workers = [
            asyncio.create_task(device_worker(device, queue, max_retries, semaphore, progress, executor))
            for device in devices
        ]
        results = await asyncio.gather(*workers)
    failed = [package for failures in results for package in failures]
    if failed:
        raise RuntimeError(f"Collection failed for {len(failed)} packages: {', '.join(failed)}")


def cli() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    asyncio.run(main())


if __name__ == "__main__":
    cli()
