"""Create isolated copies of a local Android Virtual Device and start them."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
from pathlib import Path

from gsar.config import get_data_synthesis_config, resolve_path


async def wait_until_ready(serial: str, timeout: int = 240) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        process = await asyncio.create_subprocess_exec(
            "adb", "-s", serial, "shell", "getprop", "sys.boot_completed",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await process.communicate()
        if stdout.strip() == b"1":
            print(f"{serial} is ready")
            return
        await asyncio.sleep(5)
    raise TimeoutError(f"{serial} did not boot within {timeout} seconds")


def copy_avd(source_name: str, avd_dir: Path, work_dir: Path, destination_name: str) -> None:
    source = avd_dir / f"{source_name}.avd"
    source_ini = avd_dir / f"{source_name}.ini"
    if not source.is_dir() or not source_ini.is_file():
        raise FileNotFoundError(f"Base AVD {source_name!r} not found under {avd_dir}")

    destination = work_dir / f"{destination_name}.avd"
    destination_ini = work_dir / f"{destination_name}.ini"
    if destination.exists():
        shutil.rmtree(destination)
    destination_ini.unlink(missing_ok=True)
    shutil.copytree(source, destination)
    shutil.copy2(source_ini, destination_ini)

    ini = destination_ini.read_text(encoding="utf-8")
    ini = ini.replace(source_name, destination_name)
    ini = re.sub(r"^path=.*$", f"path={destination}", ini, flags=re.MULTILINE)
    destination_ini.write_text(ini, encoding="utf-8")

    config_path = destination / "config.ini"
    config_path.write_text(config_path.read_text(encoding="utf-8").replace(source_name, destination_name), encoding="utf-8")


async def start_one(index: int, args: argparse.Namespace) -> None:
    name = f"gsar-{index + 1}"
    port = args.base_port + index * 2
    grpc_port = args.base_grpc_port + index * 2
    copy_avd(args.source_name, args.avd_dir, args.work_dir, name)

    env = os.environ.copy()
    env["ANDROID_AVD_HOME"] = str(args.work_dir)
    log_path = args.work_dir / f"emulator-{port}.log"
    with log_path.open("wb") as log:
        await asyncio.create_subprocess_exec(
            "emulator", "-avd", name, "-no-window", "-no-audio", "-no-boot-anim",
            "-gpu", "auto", "-no-snapshot", "-grpc", str(grpc_port), "-port", str(port),
            stdout=log, stderr=log, start_new_session=True, env=env,
        )
    await wait_until_ready(f"emulator-{port}", args.timeout)


async def run(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*(start_one(index, args) for index in range(args.count)))


def main() -> None:
    config = get_data_synthesis_config().emulator
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-name", default=config.source_name)
    parser.add_argument("--avd-dir", type=Path, default=Path(config.avd_dir).expanduser())
    parser.add_argument("--work-dir", type=Path, default=resolve_path(config.working_dir))
    parser.add_argument("--count", type=int, default=int(config.count))
    parser.add_argument("--base-port", type=int, default=int(config.base_port))
    parser.add_argument("--base-grpc-port", type=int, default=int(config.base_grpc_port))
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
