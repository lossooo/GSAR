"""Run trajectory reward evaluators and compute binary classification metrics."""

from __future__ import annotations

import argparse
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from PIL import Image
from tqdm import tqdm

from ..common.model_client import OPENAI_CONFIG, configure_model
from ..config import get_verification_config, resolve_path
from .digirl import EndResultEvaluator as DigiRLEvaluator
from .distrl import EndResultEvaluator as DistRLEvaluator
from .goalstate import EndResultEvaluator as GoalStateEvaluator
from .metrics import compute_metrics
from .stepcritic import EndResultEvaluator as StepCriticEvaluator
from .stepgoal import EndResultEvaluator as StepGoalEvaluator
from .utils import draw_img_name, draw_red_border

CONFIG = get_verification_config()
RewardFunction = Callable[[dict], dict]


def _last_step(sample: dict) -> list[dict]:
    steps = sample.get("step_datas")
    if not steps:
        raise ValueError("Trajectory sample has no step_datas")
    return steps


def _action_history(steps: list[dict]) -> list[str]:
    history = []
    for index, step in enumerate(steps):
        response = step.get("response", "")
        thinking = ""
        operation = ""
        if "<thinking>" in response and "</thinking>" in response:
            thinking = response.split("<thinking>", 1)[1].split("</thinking>", 1)[0].strip()
        if "<tool_call>" in response and "</tool_call>" in response:
            operation = response.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        history.append(f"## Round {index}\n* Analysis: {thinking}\n* Operation:\n{operation}")
    return history


def _score_groups(traj: dict, score: Callable[[dict], int]) -> dict:
    rewards: dict[str, list[int]] = {}
    for group in ("neg", "pos"):
        values = []
        for sample in traj.get(group, []):
            try:
                values.append(int(score(sample)))
            except Exception as exc:
                print(f"[ERROR] query={traj.get('query', 'unknown')} (reward_{group}): {exc}")
                values.append(-2)
        rewards[f"reward_{group}"] = values
    traj["rewards"] = rewards
    return traj


def reward_digirl(traj: dict) -> dict:
    query = traj["query"]

    def score(sample: dict) -> int:
        steps = _last_step(sample)
        if len(steps) < 2:
            raise ValueError("DigiRL evaluation requires at least two steps")
        images = [steps[-2]["before_screenshot"], steps[-1]["before_screenshot"]]
        return DigiRLEvaluator()(images, query)

    return _score_groups(traj, score)


def reward_distrl(traj: dict) -> dict:
    query = traj["query"]

    def score(sample: dict) -> int:
        image_path = _last_step(sample)[-1]["before_screenshot"]
        return DistRLEvaluator()(image_path, query)

    return _score_groups(traj, score)


def reward_stepcritic(traj: dict) -> dict:
    query = traj["query"]
    evaluator = StepCriticEvaluator()

    def score(sample: dict) -> int:
        steps = _last_step(sample)
        image_path = steps[-1]["before_screenshot"]
        with open(steps[-1]["before_ui_elements_path"], encoding="utf-8") as handle:
            ui_elements = json.load(handle)
        return evaluator(query, _action_history(steps), ui_elements, image_path)

    return _score_groups(traj, score)


def _open_named_image(path: str, name: str) -> Image.Image:
    with Image.open(path) as image:
        return draw_img_name(image.copy(), name)


def _goal_anchor(traj: dict, draw_anchor: bool) -> tuple[Image.Image, str]:
    label_info = traj["label_info"]
    with Image.open(label_info["image"]) as image:
        label_image = image.copy()
    if draw_anchor:
        label_image = draw_red_border(label_image, label_info.get("coord", []))
    return draw_img_name(label_image, "label image"), label_info.get("state", "")


def reward_goalstate(traj: dict) -> dict:
    query = traj["query"]
    evaluator = GoalStateEvaluator()
    label_image, label_state = _goal_anchor(traj, draw_anchor=True)

    def score(sample: dict) -> int:
        current = _open_named_image(_last_step(sample)[-1]["before_screenshot"], "current image")
        return evaluator(query, current, label_image, label_state)

    return _score_groups(traj, score)


def reward_stepgoal(traj: dict) -> dict:
    query = traj["query"]
    evaluator = StepGoalEvaluator()
    label_image, label_state = _goal_anchor(traj, draw_anchor=False)

    def score(sample: dict) -> int:
        steps = _last_step(sample)
        current = _open_named_image(steps[-1]["before_screenshot"], "current image")
        return evaluator(query, _action_history(steps), current, label_image, label_state)

    return _score_groups(traj, score)


METHODS: dict[str, RewardFunction] = {
    "digirl": reward_digirl,
    "distrl": reward_distrl,
    "stepcritic": reward_stepcritic,
    "goalstate": reward_goalstate,
    "stepgoal": reward_stepgoal,
}


def process_label_info(traj_data: list[dict]) -> list[dict]:
    labeled_path = resolve_path(CONFIG.labeled_data_path)
    with labeled_path.open(encoding="utf-8") as handle:
        labeled_data = json.load(handle)

    labeled_map = {item["query"]: item for item in labeled_data}
    for item in traj_data:
        label_item = labeled_map.get(item.get("query"))
        if label_item and label_item.get("step_datas"):
            last_step = label_item["step_datas"][-1]
            item["label_info"] = {
                "coord": last_step.get("coord", []),
                "state": last_step.get("state", ""),
                "image": last_step.get("before_screenshot", ""),
            }

    missing = [item.get("query", "<missing query>") for item in traj_data if "label_info" not in item]
    if missing:
        preview = ", ".join(repr(query) for query in missing[:3])
        raise ValueError(f"Missing goal-state anchors for {len(missing)} trajectories: {preview}")
    return traj_data


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._") or "model"


def reward_path(method: str, model: str) -> Path:
    output_dir = resolve_path(CONFIG.reward_output_dir)
    return output_dir / f"reward_{method}_{_safe_model_name(model)}.json"


def get_reward(method: str, model: str) -> Path:
    trajectory_path = resolve_path(CONFIG.traj_data_path)
    output_path = reward_path(method, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with trajectory_path.open(encoding="utf-8") as handle:
        trajectories = json.load(handle)
    if method in {"goalstate", "stepgoal"}:
        trajectories = process_label_info(trajectories)

    results: list[dict | None] = [None] * len(trajectories)
    with ThreadPoolExecutor(max_workers=int(CONFIG.max_workers)) as executor:
        futures = {
            executor.submit(METHODS[method], trajectory): index
            for index, trajectory in enumerate(trajectories)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                print(f"[ERROR] processing trajectory {index}: {exc}")
                traceback.print_exc()

    completed = [item for item in results if item is not None]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(completed, handle, ensure_ascii=False, indent=2)
    print(f"Processed {len(completed)}/{len(trajectories)} trajectories; saved to {output_path}")
    return output_path


def get_metric(method: str, model: str) -> dict[str, int | float]:
    with reward_path(method, model).open(encoding="utf-8") as handle:
        trajectories = json.load(handle)

    tp = tn = fp = fn = 0
    for trajectory in trajectories:
        rewards = trajectory.get("rewards", {})
        for value in rewards.get("reward_pos", []):
            if value == 1:
                tp += 1
            else:
                fn += 1
        for key, values in rewards.items():
            if not key.startswith("reward_neg"):
                continue
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if value in {0, -1}:
                    tn += 1
                else:
                    fp += 1

    metrics = compute_metrics(tp, tn, fp, fn)
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSAR reward inference and metric calculation.")
    parser.add_argument("--method", choices=sorted(METHODS), default=CONFIG.method)
    parser.add_argument("--model", default=OPENAI_CONFIG["model_name"], help="OpenAI-compatible model name")
    parser.add_argument("--stage", choices=["reward", "metric", "both"], default="both")
    args = parser.parse_args()

    configure_model(model=args.model)
    if args.stage in {"reward", "both"}:
        get_reward(args.method, args.model)
    if args.stage in {"metric", "both"}:
        get_metric(args.method, args.model)


if __name__ == "__main__":
    main()
