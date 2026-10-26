"""Small parsing and action-conversion helpers used by data synthesis."""

import json

from .json_action import JSONAction


def extract_last_json(text: str):
    """Return the last valid JSON object or array embedded in text."""
    last_object = None
    index = 0
    while index < len(text):
        try:
            value, end_index = json.JSONDecoder().raw_decode(text[index:])
            last_object = value
            index += end_index
        except json.JSONDecodeError:
            index += 1
    return last_object


def convert_action_index_to_coords(action: JSONAction, coordinates) -> str | None:
    """Convert an index-based action into a coordinate-based action."""
    if not coordinates:
        return None

    converted = {"action_type": action.action_type}
    if action.action_type in {"click", "double_tap", "long_press"}:
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
            raise ValueError(f"Click-based action must have 2 coordinates, got {coordinates}")
        converted["x"], converted["y"] = coordinates
    elif action.action_type == "input_text":
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
            raise ValueError(f"Input_text action must have 2 coordinates, got {coordinates}")
        converted.update({"text": action.text, "x": coordinates[0], "y": coordinates[1]})
    elif action.action_type in {"scroll", "swipe"}:
        if len(coordinates) != 4:
            raise ValueError(f"Swipe action must have 4 coordinates, got {coordinates}")
        converted.update(
            {
                "start_x": coordinates[0],
                "start_y": coordinates[1],
                "end_x": coordinates[2],
                "end_y": coordinates[3],
                "direction": action.direction,
            }
        )
    else:
        raise ValueError(f"Unsupported action type: {action.action_type}")

    return json.dumps(converted, ensure_ascii=False)
