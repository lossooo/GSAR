"""Analyze task and screen complexity across synthesis iterations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..config import resolve_path


def load_trajectories(paths: Iterable[str | Path]) -> list[dict]:
    """Load and merge trajectory arrays from JSON files or directories."""
    files: list[Path] = []
    for value in paths:
        path = resolve_path(value)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Trajectory input not found: {path}")

    trajectories = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        trajectories.extend(item for item in data if isinstance(item, dict))
    return trajectories


def _mean(values: list[float | int]) -> float:
    return round(float(np.mean(values)), 6) if values else 0.0


def _balanced_sample(
    first: list,
    second: list,
    sample_size: int | None,
    rng: random.Random,
) -> tuple[list, list]:
    count = min(len(first), len(second))
    if sample_size is not None:
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        count = min(count, sample_size)
    if count == 0:
        return [], []
    return rng.sample(first, count), rng.sample(second, count)


def task_complexity_effect(
    trajectories: list[dict],
    *,
    sample_size: int | None = None,
    seed: int = 42,
) -> dict:
    """Compare original and complexified task length and trajectory length."""
    normal = [item for item in trajectories if not item.get("is_complexified", False)]
    complexified = [item for item in trajectories if item.get("is_complexified", False)]
    normal_sample, complex_sample = _balanced_sample(
        normal,
        complexified,
        sample_size,
        random.Random(seed),
    )
    return {
        "normal_count": len(normal),
        "complexified_count": len(complexified),
        "compared_count_per_group": len(normal_sample),
        "normal_query_length": _mean([len(item.get("query", "")) for item in normal_sample]),
        "complexified_query_length": _mean(
            [len(item.get("query", "")) for item in complex_sample]
        ),
        "normal_steps": _mean(
            [item.get("step_nums", len(item.get("step_datas", []))) for item in normal_sample]
        ),
        "complexified_steps": _mean(
            [item.get("step_nums", len(item.get("step_datas", []))) for item in complex_sample]
        ),
    }


def edge_density(image: np.ndarray) -> float:
    """Return the fraction of pixels detected as edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    edges = cv2.Canny(adaptive, 50, 150)
    return float(np.mean(edges > 0))


def image_entropy(image: np.ndarray) -> float:
    """Return grayscale Shannon entropy for one screenshot."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histogram = histogram / histogram.sum()
    nonzero = histogram[histogram > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _valid_steps(trajectory: dict) -> list[tuple[Path, Path]]:
    candidates = []
    for step in (trajectory.get("step_datas") or [])[1:]:
        image_value = step.get("before_screenshot")
        ui_value = step.get("before_ui_elements_path")
        if not image_value or not ui_value:
            continue
        image_path = resolve_path(image_value)
        ui_path = resolve_path(ui_value)
        if image_path.is_file() and ui_path.is_file():
            candidates.append((image_path, ui_path))
    return candidates


def _screen_metrics(
    samples: list[tuple[Path, Path]],
) -> dict[str, float | int]:
    edges = []
    entropies = []
    element_counts = []
    for image_path, ui_path in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        with ui_path.open(encoding="utf-8") as handle:
            elements = json.load(handle)
        edges.append(edge_density(image))
        entropies.append(image_entropy(image))
        element_counts.append(len(elements))
    return {
        "sample_count": len(edges),
        "edge_density": _mean(edges),
        "image_entropy": _mean(entropies),
        "ui_elements": _mean(element_counts),
    }


def screen_complexity_effect(
    trajectories: list[dict],
    *,
    early_iterations: tuple[int, ...] = (1, 2),
    late_iterations: tuple[int, ...] = (5, 6),
    sample_size: int | None = None,
    seed: int = 42,
) -> dict:
    """Compare GUI complexity between early and late synthesis iterations."""
    rng = random.Random(seed)

    def collect(iterations: tuple[int, ...]) -> list[tuple[Path, Path]]:
        collected = []
        for trajectory in trajectories:
            if int(trajectory.get("iteration", 0)) not in iterations:
                continue
            steps = _valid_steps(trajectory)
            if steps:
                collected.append(rng.choice(steps))
        return collected

    early = collect(early_iterations)
    late = collect(late_iterations)
    early_sample, late_sample = _balanced_sample(early, late, sample_size, rng)
    return {
        "early_iterations": list(early_iterations),
        "late_iterations": list(late_iterations),
        "available_early": len(early),
        "available_late": len(late),
        "early": _screen_metrics(early_sample),
        "late": _screen_metrics(late_sample),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze task and screen complexity in synthesized trajectories."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Trajectory JSON files or directories")
    parser.add_argument("--mode", choices=["all", "task", "screen"], default="all")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-iterations", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--late-iterations", nargs="+", type=int, default=[5, 6])
    args = parser.parse_args()

    trajectories = load_trajectories(args.input)
    results = {}
    if args.mode in {"all", "task"}:
        results["task_complexity"] = task_complexity_effect(
            trajectories,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    if args.mode in {"all", "screen"}:
        results["screen_complexity"] = screen_complexity_effect(
            trajectories,
            early_iterations=tuple(args.early_iterations),
            late_iterations=tuple(args.late_iterations),
            sample_size=args.sample_size,
            seed=args.seed,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
