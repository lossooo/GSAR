"""Score synthesized trajectories before SFT post-processing."""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..common.model_client import MLLM_Model
from ..config import get_data_synthesis_config, resolve_path

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_FILTER = """
You are an expert in evaluating Android GUI agent task trajectories. Assess the
quality and effectiveness of the supplied trajectory.

A trajectory contains a high-level instruction, an action history with a summary
for each step, and screenshots from its last three steps.

Evaluate:
1. Trajectory coherence: whether the actions form a logical sequence, are clearly
   described, and avoid unnecessary repetition.
2. Task completion: whether the instruction is completed and errors are handled.

Score the trajectory from 1 to 5:
- 5: Fully completed with a clear and efficient sequence.
- 4: Mostly completed, with minor ambiguity or inefficiency.
- 3: Partially completed or ended because of an environmental constraint.
- 2: Made little useful progress or deviated from the goal.
- 1: Failed without meaningful progress.

For a difficult task, credit useful progress even when it is incomplete.

Respond with exactly:
Reason: <brief reasoning>
Score: <integer from 1 to 5>

"""


def _parse_evaluation(response: str) -> tuple[str, int] | None:
    reason_match = re.search(r"Reason:\s*(.+?)\s*Score:", response, re.DOTALL | re.IGNORECASE)
    score_match = re.search(r"Score:\s*([1-5])\b", response, re.IGNORECASE)
    if not reason_match or not score_match:
        return None
    return reason_match.group(1).strip(), int(score_match.group(1))


def get_reward(item: dict, model: MLLM_Model, force: bool = False) -> dict:
    """Add a 1–5 quality score and reason to one trajectory."""
    if "reward" in item and not force:
        return item

    steps = item.get("step_datas") or []
    if not steps:
        logger.warning("Skipping trajectory without step_datas: %r", item.get("query"))
        return item

    history = []
    for index, step in enumerate(steps, start=1):
        summary = step.get("summary") or step.get("action_reason") or "No summary available."
        history.append(f"Step {index}: {summary}")
    history_text = "\n".join(history)

    screenshots = [
        step["before_screenshot"]
        for step in steps[-3:]
        if step.get("before_screenshot")
    ]
    if not screenshots:
        logger.warning("Skipping trajectory without screenshots: %r", item.get("query"))
        return item

    trajectory_prompt = (
        f"Instruction: {item.get('query', '')}\n"
        f"Action History:\n{history_text}\n"
        "The last three screenshots are attached."
    )

    for attempt in range(1, 4):
        try:
            response, _ = model.get_model_response(
                prompt=PROMPT_TEMPLATE_FILTER + trajectory_prompt,
                image_paths=screenshots,
            )
            parsed = _parse_evaluation(response or "")
            if parsed:
                item["reward_reason"], item["reward"] = parsed
                return item
            logger.warning("Could not parse trajectory score (attempt %s/3)", attempt)
        except Exception as exc:
            logger.warning("Trajectory scoring failed (attempt %s/3): %s", attempt, exc)

    return item


def _json_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(source.glob("*.json"))
    raise FileNotFoundError(f"Trajectory input not found: {source}")


def filter_traj(max_workers: int = 8, traj_dir: str | Path | None = None, force: bool = False) -> None:
    """Score every trajectory in a JSON file or directory, updating files in place."""
    config = get_data_synthesis_config()
    default_dir = resolve_path(config.data_collect.data_root) / "traj_json"
    source = resolve_path(traj_dir) if traj_dir else default_dir
    files = _json_files(source)
    if not files:
        raise FileNotFoundError(f"No trajectory JSON files found in {source}")

    model = MLLM_Model()
    for path in files:
        with path.open(encoding="utf-8") as handle:
            trajectories = json.load(handle)
        if not isinstance(trajectories, list):
            raise ValueError(f"Trajectory JSON must contain an array: {path}")

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = [
                executor.submit(get_reward, trajectory, model, force)
                for trajectory in trajectories
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=path.name):
                future.result()

        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(trajectories, handle, ensure_ascii=False, indent=2)
        temporary.replace(path)
        logger.info("Saved trajectory scores to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score synthesized trajectories from 1 to 5.")
    parser.add_argument("--input", help="Trajectory JSON file or directory; defaults to synthesis output")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent model requests")
    parser.add_argument("--force", action="store_true", help="Re-score trajectories that already have a reward")
    args = parser.parse_args()
    filter_traj(max_workers=args.workers, traj_dir=args.input, force=args.force)


if __name__ == "__main__":
    main()
