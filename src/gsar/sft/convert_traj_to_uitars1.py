"""Convert GSAR trajectories to the UI-TARS high-level SFT format."""

from __future__ import annotations

import argparse
import json
import uuid

from tqdm import tqdm

from ..config import resolve_path

DEFAULT_SCREEN_SIZE = (1080, 2400)


def convert_coordinate(
    x: float,
    y: float,
    screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
) -> tuple[int, int]:
    """Convert absolute screen coordinates to UI-TARS's 1000x1000 space."""
    width, height = screen_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid screen size: {screen_size}")
    relative_x = max(0, min(999, int(x / width * 1000)))
    relative_y = max(0, min(999, int(y / height * 1000)))
    return relative_x, relative_y


def convert_action_to_tool_call(
    action_value: str | dict | None,
    screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
) -> dict | None:
    """Convert one collected action to the common mobile-use representation."""
    if not action_value:
        return None
    try:
        action = json.loads(action_value) if isinstance(action_value, str) else action_value
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(action, dict):
        return None

    action_type = action.get("action_type", "")
    arguments: dict
    if action_type in {"click", "double_tap", "long_press"}:
        coordinate = convert_coordinate(action.get("x", 0), action.get("y", 0), screen_size)
        arguments = {
            "action": "long_press" if action_type == "long_press" else "click",
            "coordinate": list(coordinate),
        }
        if action_type == "long_press":
            arguments["time"] = action.get("time", 2)
    elif action_type in {"type", "input_text"}:
        arguments = {"action": "type", "text": action.get("text", "")}
    elif action_type in {"scroll", "swipe"}:
        width, height = screen_size
        start = (
            action.get("start_x", width // 2),
            action.get("start_y", height // 2),
        )
        end = (
            action.get("end_x", width // 2),
            action.get("end_y", 0),
        )
        arguments = {
            "action": "swipe",
            "coordinate": list(convert_coordinate(*start, screen_size)),
            "coordinate2": list(convert_coordinate(*end, screen_size)),
        }
    elif action_type == "keyboard_enter":
        arguments = {"action": "system_button", "button": "Enter"}
    elif action_type == "navigate_back":
        arguments = {"action": "system_button", "button": "Back"}
    elif action_type == "navigate_home":
        arguments = {"action": "system_button", "button": "Home"}
    elif action_type == "open_app":
        arguments = {"action": "open", "text": action.get("app_name", "")}
    elif action_type == "wait":
        arguments = {"action": "wait", "time": action.get("time", 2)}
    elif action_type == "status":
        status = action.get("goal_status", "")
        arguments = {
            "action": "terminate",
            "status": "success" if status in {"complete", "successful", "success"} else "failure",
        }
    elif action_type == "answer":
        arguments = {"action": "terminate", "status": "success"}
    else:
        return None
    return {"name": "mobile_use", "arguments": arguments}


def convert_action_to_uitars_str(tool_call: dict) -> str:
    """Render one mobile-use call in UI-TARS action syntax."""
    arguments = tool_call["arguments"]
    action_type = arguments["action"]
    if action_type == "click":
        x, y = arguments["coordinate"]
        return f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"
    if action_type == "long_press":
        x, y = arguments["coordinate"]
        return (
            f"long_press(start_box='<|box_start|>({x},{y})<|box_end|>', "
            f"time={arguments.get('time', 2)!r})"
        )
    if action_type == "type":
        return f"type(content={arguments.get('text', '')!r})"
    if action_type == "swipe":
        start_x, start_y = arguments["coordinate"]
        end_x, end_y = arguments["coordinate2"]
        delta_x, delta_y = end_x - start_x, end_y - start_y
        if abs(delta_x) > abs(delta_y):
            direction = "right" if delta_x > 0 else "left"
        else:
            direction = "down" if delta_y > 0 else "up"
        return f"scroll(direction={direction!r})"
    if action_type == "system_button":
        return {
            "Back": "press_back()",
            "Home": "press_home()",
            "Enter": "press_enter()",
        }.get(arguments.get("button"), "press_back()")
    if action_type == "open":
        return f"open_app(app_name={arguments.get('text', '')!r})"
    if action_type == "wait":
        return "wait()"
    if action_type == "terminate":
        return "finished()"
    raise ValueError(f"Unsupported UI-TARS action: {action_type}")


def _instruction_prompt(query: str) -> str:
    return (
        "You are a GUI agent. You are given a task and your action history, with screenshots. "
        "You need to perform the next action to complete the task.\n\n"
        "## Output Format\n\nThought: ...\nAction: ...\n\n"
        "## Action Space\n"
        "click(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
        "long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')\n"
        "type(content='')\n"
        "scroll(direction='down or up or right or left')\n"
        "press_back()\npress_home()\npress_enter()\nwait()\n"
        "finished() # Submit the task regardless of whether it succeeds or fails.\n\n"
        "## Note\n- Use English in Thought part.\n"
        "- Summarize your next action (with its target element) in one sentence in Thought part.\n\n"
        f"## User Instruction\n{query}"
    )


def _screen_size(step: dict) -> tuple[int, int]:
    value = step.get("logical_screen_size") or DEFAULT_SCREEN_SIZE
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return DEFAULT_SCREEN_SIZE
    return int(value[0]), int(value[1])


def build_sft_samples_from_traj(trajectory: dict) -> list[dict]:
    """Build one training sample per valid trajectory step."""
    query = trajectory.get("query", "")
    trajectory_id = uuid.uuid4().hex
    converted_steps = []
    for step in trajectory.get("step_datas", []):
        tool_call = convert_action_to_tool_call(
            step.get("action_wo_index"),
            _screen_size(step),
        )
        thought = (
            step.get("new_action_reason")
            or step.get("action_reason")
            or step.get("summary")
            or "Perform the next action."
        )
        converted_steps.append((tool_call, thought, step.get("before_screenshot", "")))

    samples = []
    for step_index, (tool_call, thought, image_path) in enumerate(converted_steps):
        if not tool_call or not image_path:
            continue

        history = [
            previous
            for previous in converted_steps[:step_index]
            if previous[0] and previous[2]
        ][-4:]
        images = [previous[2] for previous in history] + [image_path]
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        prompt = _instruction_prompt(query)

        if history:
            messages.append({"role": "user", "content": prompt + "\n<image>"})
            for history_index, (previous_call, previous_thought, _) in enumerate(history):
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Thought: {previous_thought}\n"
                            f"Action: {convert_action_to_uitars_str(previous_call)}"
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "<image>",
                    }
                )
        else:
            messages.append({"role": "user", "content": prompt + "\n<image>"})

        messages.append(
            {
                "role": "assistant",
                "content": f"Thought: {thought}\nAction: {convert_action_to_uitars_str(tool_call)}",
            }
        )
        samples.append(
            {
                "id": f"{trajectory_id}-{step_index + 1}",
                "images": images,
                "messages": messages,
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GSAR trajectories to UI-TARS high-level SFT format."
    )
    parser.add_argument("--input", required=True, help="Filtered trajectory JSON")
    parser.add_argument("--output", required=True, help="Output SFT JSON")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    with input_path.open(encoding="utf-8") as handle:
        trajectories = json.load(handle)
    if not isinstance(trajectories, list):
        raise ValueError(f"Expected a JSON array in {input_path}")

    converted = []
    for trajectory in tqdm(trajectories, desc="Converting"):
        converted.extend(build_sft_samples_from_traj(trajectory))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False, indent=2)
    print(f"Converted {len(trajectories)} trajectories into {len(converted)} samples -> {output_path}")


if __name__ == "__main__":
    main()
