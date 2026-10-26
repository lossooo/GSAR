import re
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
import traceback

from openai import OpenAI

from ..common.model_client import OPENAI_CONFIG, configure_model, create_openai_client
from ..config import get_verification_config, resolve_path

from .utils import (
    dict_to_ui_element,
    _validate_ui_element,
    _format_element,
    add_ui_element_mark,
    encode_image_to_data_url,
)


"""对轨迹中的最后一步截图进行标注，标注出与任务最相关的UI组件"""
"""同一个指令，只标注一个轨迹。先进行anchor标注, 如果没有合适的框，则再进行状态描述。中间关键步骤也要考虑"""

PROMPT_TEMPLATE_LABEL_ANCHOR = """
You are an expert in GUI understanding. Your task is to analyze the input image and UI Elements, and select the indices of the UI elements that are relevant to the instruction. These selected UI elements should uniquely identify the state after the instruction task has been completed.

Input Description:
- Instruction: A natural language description of the GUI task to be performed.
- UI Elements: The complete list of UI elements extracted from the final-state screen (including all element metadata).
- Image: The screenshot of the final state after completing the task, with UI element bounding boxes and indices overlaid to indicate their positions.

Requirements:
- Compare the image and the UI Elements list, and find UI elements whose content or visual role reflects the completion of the instruction.
- Select key UI elements that can indicate the final page state, such as navigation bars, menu titles, tab titles, selected buttons, status indicators, etc.
- Avoid selecting UI elements whose state may change dynamically due to time or network content.
- If you believe that no UI element in the screen is relevant to the instruction, stop the reasoning and output "No related elements".

Now output in the following format:
Thinking Process: Your thinking process
Output Index: [x, x, ...]

Input:
Instruction: {query}
UI Elements: {ui_elements_info}
"""


def extract_indices(model_output: str):
    matches = re.findall(r"Output Index:\s*\[([^\]]*)\]", model_output)
    if not matches:
        return []
    content = matches[-1]
    indices = re.findall(r"-?\d+", content)
    return [int(x) for x in indices]


def traj_label_index(item: dict, client: OpenAI):
    query = item["query"]
    if not item.get("step_datas"):
        raise ValueError(f"Trajectory has no steps: {query!r}")
    last_step = item["step_datas"][-1]

    before_screenshot_path = last_step["before_screenshot"]
    before_ui_elements_path = last_step["before_ui_elements_path"]

    if not before_screenshot_path or not before_ui_elements_path:
        print(f"Skipping trajectory with incomplete final-step data: {query!r}")
        return None
    
    with Image.open(before_screenshot_path) as image:
        before_screenshot_pil = image.convert("RGB")
    screen_width_height_px = before_screenshot_pil.size  # (width, height)


    with open(before_ui_elements_path, "r", encoding="utf-8") as f:
        ui_elements_json = json.load(f)
    ui_elements = [dict_to_ui_element(el) for el in ui_elements_json]

    # format element and add ui element mark on before screenshot.
    tree_info = ''
    width, height = screen_width_height_px
    physical_frame_boundary = (0, 0, width, height)
    orientation = 0 if width < height else 1
    before_screenshot_np = np.array(before_screenshot_pil)
    for index, ui_element in enumerate(ui_elements):
        if _validate_ui_element(ui_element, screen_width_height_px):
            
            formated_element = _format_element(ui_element, index)
            if formated_element:
                tree_info += formated_element + '\n'

            add_ui_element_mark(
                before_screenshot_np,
                ui_element,
                index,
                screen_width_height_px,
                physical_frame_boundary,
                orientation,
            )
    
    marked_image_pil = Image.fromarray(before_screenshot_np)
    image_url = encode_image_to_data_url(marked_image_pil)

    prompt = PROMPT_TEMPLATE_LABEL_ANCHOR.format(
        query=query,
        ui_elements_info=tree_info.strip()
    )
    # print(Fore.GREEN + query + tree_info.strip())

    user_content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "\nImage:"},
        {"type": "image_url","image_url": {"url": image_url}},
    ]
    messages = [
        {"role": "user", "content": user_content}
    ]

    max_retry = 3
    labeled = False
    for attempt in range(1, max_retry + 1):
        res = client.chat.completions.create(
            model=OPENAI_CONFIG["model_name"],
            messages = messages,
            temperature=0.5,
            top_p=0.5
        )
        response = res.choices[0].message.content

        # extract index and save coordinates
        coordinates_list = []
        state_desc = ""
        indices = extract_indices(response)

        if indices:
            # print(Fore.YELLOW + str(indices))
            for index in indices:
                if index < 0 or index >= len(ui_elements):
                    print(f"Invalid element index: {index}, must be 0-{len(ui_elements)-1}.")
                    continue
                bbox = ui_elements[index].bbox_pixels
                coordinates = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
                coordinates_list.append(coordinates)
        elif "No related elements" in response:
            state_desc = ""
            coordinates = (0, 0, width, height)
            coordinates_list.append(coordinates)
        else:
            continue

        if coordinates_list or state_desc:
            last_step["coord"] = coordinates_list
            last_step["state"] = state_desc
            item["step_datas"][-1] = last_step
            labeled = True
            break

    if not labeled:
        print(f"{item['query']} not labeled")
        return None
    return item



def label_image(input_path=None, output_path=None, max_workers=None):
    config = get_verification_config()
    traj_data_path = resolve_path(input_path or config.anchor_source_path)
    label_data_path = resolve_path(output_path or config.labeled_data_path)
    label_data_path.parent.mkdir(parents=True, exist_ok=True)

    with traj_data_path.open(encoding="utf-8") as f:
        json_data = json.load(f)
    # json_data = json_data[:5]

    client = create_openai_client()

    results = [None] * len(json_data)
    with ThreadPoolExecutor(max_workers=max_workers or int(config.max_workers)) as executor:
        futures = {
            executor.submit(traj_label_index, item, client): index
            for index, item in enumerate(json_data)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"[ERROR] processing trajectory {index}: {e}")
                traceback.print_exc()

    results = [item for item in results if item is not None]
    with label_data_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Finished processing {len(results)} items")



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate goal-state anchors from successful trajectories.")
    parser.add_argument("--input", help="Trajectory JSON; defaults to verification config")
    parser.add_argument("--output", help="Anchor JSON; defaults to verification config")
    parser.add_argument("--model", help="OpenAI-compatible model name")
    parser.add_argument("--workers", type=int, help="Concurrent model requests")
    args = parser.parse_args()
    configure_model(model=args.model)
    label_image(args.input, args.output, args.workers)


if __name__ == "__main__":
    main()
