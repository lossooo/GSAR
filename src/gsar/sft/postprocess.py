"""Filter synthesized trajectories into a deterministic SFT source file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ..config import resolve_path


def _json_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    if source.is_dir():
        yield from sorted(source.glob("*.json"))
        return
    raise FileNotFoundError(f"Trajectory input not found: {source}")


def load_trajectories(source: str | Path) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for path in _json_files(resolve_path(source)):
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array in {path}")
        trajectories.extend(item for item in value if isinstance(item, dict))
    return trajectories


def postprocess(
    trajectories: Iterable[dict[str, Any]],
    *,
    min_reward: float = 3,
    min_steps: int = 3,
    strip_prompts: bool = True,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    removable = {"action_prompt", "action_output", "summary_prompt", "summary_raw_response"}
    for trajectory in trajectories:
        steps = trajectory.get("step_datas") or []
        step_count = int(trajectory.get("step_nums", len(steps)))
        if float(trajectory.get("reward", 0)) < min_reward or step_count < min_steps:
            continue
        if strip_prompts:
            for step in steps:
                if isinstance(step, dict):
                    for key in removable:
                        step.pop(key, None)
        selected.append(trajectory)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter synthesized trajectories for SFT conversion.")
    parser.add_argument("--input", required=True, help="Trajectory JSON file or directory")
    parser.add_argument("--output", required=True, help="Filtered trajectory JSON")
    parser.add_argument("--min-reward", type=float, default=3)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--keep-prompts", action="store_true")
    args = parser.parse_args()

    trajectories = load_trajectories(args.input)
    selected = postprocess(
        trajectories,
        min_reward=args.min_reward,
        min_steps=args.min_steps,
        strip_prompts=not args.keep_prompts,
    )
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
    print(f"Selected {len(selected)}/{len(trajectories)} trajectories -> {output}")


if __name__ == "__main__":
    main()

