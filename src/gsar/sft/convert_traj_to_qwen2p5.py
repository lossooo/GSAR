import copy
import re
import uuid
import json
import random
import traceback
from PIL import Image
from tqdm import tqdm
random.seed(42)

from ..config import resolve_path

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
# from qwen_vl_utils import smart_resize
from .utils import smart_resize, extract_json
from .mobile_use import MobileUse

MIN_PIXELS=3136
MAX_PIXELS=10035200

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

ACTION_TYPE_LOCK = threading.Lock()
ACTION_TYPE_COUNTER = Counter()


def build_system_messages(resized_width, resized_height):
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )

    query_messages = [
        Message(
            role="system", content=[ContentItem(text="You are a helpful assistant.")]
        )
    ]

    messages = NousFnCallPrompt().preprocess_fncall_messages(
        messages=query_messages,
        functions=[mobile_use.function],
        lang=None,
    )
    messages = [m.model_dump() for m in messages]

    system_prompt_part = {
        'role': 'system', 'content': messages[0]['content'][0]['text'] + messages[0]['content'][1]['text']
    }

    return system_prompt_part


def build_user_messages(
    instruction, enable_think=False, history=None, 
    think_tag_begin='<thinking>', think_tag_end='</thinking>', 
    level="high", low_instruction=""
):
    if history is None:
        history = []
    history = ''.join([f'Step {si+1}: {_}; 'for si, _ in enumerate(history)])

    user_prompt = f'''The user query: {instruction}'''

    if enable_think:
        user_prompt += f'\nBefore answering, explain your reasoning step-by-step in {think_tag_begin}{think_tag_end} tags, and insert them before the <tool_call></tool_call> XML tags.'
        user_prompt += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'

    if level == "low":
        user_prompt += f"\nCurrent step query: {low_instruction}"
    user_prompt += f'\nTask progress (You have done the following operation on the current device): {history}'

    # user_messages = {"role": "user", "content": [{"text": user_prompt + '\n'}]}
    user_messages = {"role": "user", "content": user_prompt + '<image>'}

    return user_messages


def action_m3a_2_qwen_2p5(action: dict, resized_width, resized_height, original_width, original_height) -> str:
    """
    Convert processed M3A action (with coordinates) into Qwen-2.5 tool_call format.
    Returns a string containing the <tool_call>...</tool_call> XML block.
    """

    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    action_type = action.get("action_type", "").lower()
    tool_name = "mobile_use"
    arguments = {}

    if action_type in ["click", "long_press"]:
        arguments["action"] = action_type
        arguments["coordinate"] = [int(action["x"]*scale_x), int(action["y"]*scale_y)]

        if action_type == "long_press":
            arguments["time"] = action.get("time", 2)  # seconds

    elif action_type == "scroll":
        arguments["action"] = "swipe"
        cx, cy = resized_width // 2, resized_height // 2
        # 滑动幅度：屏幕的 1/4（可调）
        dx = resized_width // 4
        dy = resized_height // 4
        direction = action["direction"]
        if direction == "down":
            start = (cx, cy + dy)
            end   = (cx, cy - dy)
        elif direction == "up":
            start = (cx, cy - dy)
            end   = (cx, cy + dy)
        elif direction == "left":
            start = (cx - dx, cy)
            end   = (cx + dx, cy)
        elif direction == "right":
            start = (cx + dx, cy)
            end   = (cx - dx, cy)
        else:
            raise ValueError(f"Invalid scroll direction: {direction}")

        arguments["coordinate"]  = [int(start[0]), int(start[1])]
        arguments["coordinate2"] = [int(end[0]), int(end[1])]     

    elif action_type in ["type", "input_text", "answer"]:
        arguments["action"] = "type" if action_type in ["type", "input_text"] else "answer"
        text = action["text"]
        # escape backslashes and quotes
        text = text.replace("\\", "\\\\").replace('"', '\\"')
        arguments["text"] = text

    elif action_type == "keyboard_enter":
        arguments["action"] = "system_button"
        arguments["button"] = "Enter"

    elif action_type == "navigate_home":
        arguments["action"] = "system_button"
        arguments["button"] = "Home"

    elif action_type == "navigate_back":
        arguments["action"] = "system_button"
        arguments["button"] = "Back"

    elif action_type == "wait":
        arguments["action"] = "wait"
        arguments["time"] = action.get("time", 2.0)  # default 2.0s

    elif action_type == "open_app":
        arguments["action"] = "open"
        arguments["text"] = action.get("app_name", "")

    elif action_type == "status":
        arguments["action"] = "terminate"
        if action.get("goal_status", "") in ["complete", "successful"]:
            arguments["status"] = "success"
        elif action.get("goal_status", "") in ["infeasible", "failure"]:
            arguments["status"] = "failure"
        else:
            raise NotImplementedError(f"Unknown goal_status: {action.get('goal_status', '')}")

    else:
        raise NotImplementedError(f"Unknown action_type: {action_type}")

    # return XML-wrapped JSON
    tool_call = {
        "name": tool_name,
        "arguments": arguments
    }

    # return f"<tool_call>\n{json.dumps(tool_call)}\n</tool_call>"
    return json.dumps(tool_call)


def process_traj_to_sft_data(item, enable_think=False, level="high"):
    sft_step_datas = []
    query = item['query']
    query_id = uuid.uuid4().hex
    history = []

    step_datas = copy.deepcopy(item["step_datas"])
    for i, step in enumerate(step_datas):
        image_path = step["before_screenshot"]
        dummy_image = Image.open(image_path)
        original_height, original_width = dummy_image.height, dummy_image.width
        resized_height, resized_width  = smart_resize(
            dummy_image.height,
            dummy_image.width,
            factor=28,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,)
        dummy_image = dummy_image.resize((resized_width, resized_height))

        system_messages = build_system_messages(dummy_image.width, dummy_image.height)
        if level == "low":
            low_instruction = (
                step.get("new_action_reason")
                or step.get("action_reason")
                or step.get("summary")
                or "Perform the next action."
            )
            pattern = r"\([^)]*index\s*\d+[^)]*\)"
            low_instruction = re.sub(pattern, "", low_instruction)

            user_messages = build_user_messages(
                instruction=query, enable_think=enable_think, 
                history=history, level="low", low_instruction=low_instruction
            )
        else:
            user_messages = build_user_messages(instruction=query, enable_think=enable_think, history=history)

        action_str = step["action_wo_index"]
        if "Invalid action" in action_str:
            return []
        action_m3a = extract_json(action_str)
        if action_m3a["action_type"] == "answer":
            continue

        # 统计action信息
        with ACTION_TYPE_LOCK:
            ACTION_TYPE_COUNTER[action_m3a["action_type"]] += 1

        # # 对input_tex操作分为两步，先click后type
        # if action_m3a["action_type"] == "input_text":
        #     scale_x = resized_width / original_width
        #     scale_y = resized_height / original_height
        #     x, y = int(action_m3a["x"] * scale_x), int(action_m3a["y"] * scale_y)

        #     action_click = {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [x, y]}}
        #     action_type = {"name": "mobile_use", "arguments": {"action": "type", "text": action_m3a["text"]}}
        #     action_click = json.dumps(action_click)
        #     action_type = json.dumps(action_type)
            
        #     if level == "low":
        #         user_messages_1 = build_user_messages(
        #             instruction=query, enable_think=enable_think, 
        #             history=history, level="low", low_instruction=low_instruction
        #         )
        #     else:
        #         user_messages_1 = build_user_messages(instruction=query, enable_think=enable_think, history=history)

        #     assistant_content_1 = f"<tool_call>\n{action_click}\n</tool_call>"
        #     assistant_messages_1 = {"role": "assistant", "content": assistant_content_1}
        #     messages_1 = [system_messages, user_messages_1, assistant_messages_1]
        #     step_data_1 = {"id": f"{query_id}-{i+1}", "images": [image_path], "messages": messages_1}
        #     sft_step_datas.append(step_data_1)
        #     history.append(action_click)

        #     if level == "low":
        #         user_messages_2 = build_user_messages(
        #             instruction=query, enable_think=enable_think, 
        #             history=history, level="low", low_instruction=low_instruction
        #         )
        #     else:
        #         user_messages_2 = build_user_messages(instruction=query, enable_think=enable_think, history=history)
        #     assistant_content_2 = f"<tool_call>\n{action_type}\n</tool_call>"
        #     assistant_messages_2 = {"role": "assistant", "content": assistant_content_2}
        #     messages_2 = [system_messages, user_messages_2, assistant_messages_2]
        #     step_data_2 = {"id": f"{query_id}-{i+1}", "images": [image_path], "messages": messages_2}
        #     sft_step_datas.append(step_data_2)
        #     history.append(action_type)
        # else:
        #     action = action_m3a_2_qwen_2p5(action_m3a, resized_width, resized_height, original_width, original_height)
        #     action_m3a = json.dumps(action_m3a)

        #     if enable_think:
        #         thinking = step["new_action_reason"]
        #         pattern = r"\([^)]*index\s*\d+[^)]*\)"
        #         thinking = re.sub(pattern, "", thinking)
        #         conclusion = step["conclusion"]

        #         thinking_part = f"<thinking>\n{thinking}\n</thinking>"
        #         action_part = f"<tool_call>\n{action}\n</tool_call>"
        #         conclusion_part = f"<conclusion>\n{conclusion}\n</conclusion>"
        #         assistant_content = thinking_part + "\n" +  action_part + "\n" + conclusion_part
        #     else:
        #         assistant_content = f"<tool_call>\n{action_m3a}\n</tool_call>"

        #     assistant_messages = {"role": "assistant", "content": assistant_content}

        #     messages = [system_messages, user_messages, assistant_messages]
        #     step_data = {"id": f"{query_id}-{i+1}", "images": [image_path], "messages": messages}
        #     sft_step_datas.append(step_data)

        #     history.append(action)


        action = action_m3a_2_qwen_2p5(action_m3a, resized_width, resized_height, original_width, original_height)
        action_m3a = json.dumps(action_m3a)

        if enable_think:
            thinking = (
                step.get("new_action_reason")
                or step.get("action_reason")
                or step.get("summary")
                or "Perform the next action."
            )
            pattern = r"\([^)]*index\s*\d+[^)]*\)"
            thinking = re.sub(pattern, "", thinking)
            conclusion = step.get("conclusion", "Action completed.")

            thinking_part = f"<thinking>\n{thinking}\n</thinking>"
            action_part = f"<tool_call>\n{action}\n</tool_call>"
            conclusion_part = f"<conclusion>\n{conclusion}\n</conclusion>"
            assistant_content = thinking_part + "\n" +  action_part + "\n" + conclusion_part
        else:
            assistant_content = f"<tool_call>\n{action}\n</tool_call>"

        assistant_messages = {"role": "assistant", "content": assistant_content}

        messages = [system_messages, user_messages, assistant_messages]
        step_data = {"id": f"{query_id}-{i+1}", "images": [image_path], "messages": messages}
        sft_step_datas.append(step_data)

        history.append(action)

    return sft_step_datas


def process_one_traj(item):
    step_nums = item.get("step_nums", len(item.get("step_datas", [])))

    try:
        sft_data_high = process_traj_to_sft_data(item, enable_think=False, level="high")
        sft_data_low = process_traj_to_sft_data(item, enable_think=False, level="low")
    except Exception:
        traceback.print_exc()
        return None
    
    sft_data = sft_data_high + sft_data_low
    if not sft_data:
        return None
    # item = process_traj_to_sft_data(item, enable_think=False, level="high")
    # if not item:
    #     return None
    return {
        "sft_data": sft_data,
        "traj_count": 1,
        "step_count": step_nums,
    }

    
def get_sft_data(traj_path, out_path, num_workers=16, max_trajectories=None, seed=42):

    sft_data_total = []
    traj_count = 0
    step_count = 0

    traj_path = resolve_path(traj_path)
    out_path = resolve_path(out_path)

    with traj_path.open(encoding="utf-8") as f:
        json_data = json.load(f)
    if not isinstance(json_data, list):
        raise ValueError(f"Expected a JSON array in {traj_path}")

    random.Random(seed).shuffle(json_data)

    ACTION_TYPE_COUNTER.clear()
    ordered_results = [None] * len(json_data)
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = {
            executor.submit(process_one_traj, item): index
            for index, item in enumerate(json_data)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            ordered_results[futures[future]] = future.result()

    for result in ordered_results:
        if result is None:
            continue

        sft_data_total.extend(result["sft_data"])
        traj_count += result["traj_count"]
        step_count += result["step_count"]
        if max_trajectories and traj_count >= max_trajectories:
            break

    print(f"traj count: {traj_count}")
    print(f"step count: {step_count}")
    print(f"average step per traj: {round(step_count / traj_count, 2) if traj_count else 0}")
    print(f"sft data total: {len(sft_data_total)}")

    # delete_additional_messages()

    print("\n===== Action Type Statistics =====")
    for k, v in ACTION_TYPE_COUNTER.most_common():
        print(f"{k:15s}: {v}")
    print("=================================\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(sft_data_total, f, ensure_ascii=False, indent=4)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert GSAR trajectories to Qwen2.5-VL SFT format.")
    parser.add_argument("--input", required=True, help="Filtered trajectory JSON")
    parser.add_argument("--output", required=True, help="Output SFT JSON")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    get_sft_data(args.input, args.output, args.workers, args.max_trajectories, args.seed)


if __name__ == '__main__':
    main()
