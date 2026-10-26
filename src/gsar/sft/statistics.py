"""Print compact statistics for trajectory or SFT JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from ..config import resolve_path


def summarize(items: list[dict]) -> dict:
    rewards = Counter(item.get("reward", "missing") for item in items)
    step_counts = [int(item.get("step_nums", len(item.get("step_datas", [])))) for item in items]
    packages = Counter(item.get("package_name", "missing") for item in items)
    return {
        "items": len(items),
        "packages": len(packages),
        "total_steps": sum(step_counts),
        "average_steps": round(sum(step_counts) / len(step_counts), 2) if step_counts else 0,
        "rewards": dict(sorted(rewards.items(), key=lambda pair: str(pair[0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a GSAR trajectory JSON file.")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    path = resolve_path(args.input)
    with path.open(encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON array in {path}")
    print(json.dumps(summarize(items), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

