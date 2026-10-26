import os
import re
import time
import json
import uuid
import random
import logging
from PIL import Image
from collections import deque
from dataclasses import asdict
import numpy as np
from typing import List, Tuple, Optional, Any
from colorama import init, Fore, Style
init(autoreset=True)

from .utils import json_action
from .utils import adb_utils

from ..common.prompts import (
    PROMPT_TEMPLATE_TASK_GENERATOR,
    PROMPT_TEMPLATE_WALK_SUMMARY,
    DEFAULT_EXAMPLES,
    PROMPT_TEMPLATE_TASK_OPTIMIZE,
)
from ..common.prompts import (
    PROMPT_TEMPLATE_TASK_INHERIT,
    PROMPT_TEMPLATE_TASK_MERGE,
    PROMPT_TEMPLATE_TASK_CHANGE
)
from ..common.model_client import MLLM_Model
from ..common.device import (
    Device,
    UIElement,
    validate_ui_element,
    add_screenshot_label
)
from ..config import get_data_synthesis_config, resolve_path

cfg = get_data_synthesis_config()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
"""
This module is used to collect tasks for the GSAR synthesis pipeline.
"""


def parse_task(text: str) -> list[str]:
    # pattern = r"\d+\.+(.*)"
    # matches = re.findall(pattern, text)

    pattern = r"\d+\.+(.*?)(?=\n\d+\.|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    matches = [m.strip() for m in matches]

    for i, match in enumerate(matches):
        matches[i] = match.strip()
    return matches


def task_generate(
    max_tasks: int,
    package_name: str,
    app_name: str,
    image_path: str,
    accessibility_tree: list=None,
    action_summary: str=None
):
    """Generate tasks based on the current screen state."""
    # 根据package_name动态加载提示词
    tasks_path = os.path.join(resolve_path(cfg.data_collect.data_root), "data_task", f"task_{package_name}.json")
    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        task_data = random.sample(task_data, min(10, len(task_data)))
        examples = [task["query"] for task in task_data]
        examples = "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(examples)])
    except FileNotFoundError:
        examples = DEFAULT_EXAMPLES
    # print(Fore.YELLOW + f"Examples for task generation:\n{examples}")

    prompt = PROMPT_TEMPLATE_TASK_GENERATOR.format(
        max_tasks=max_tasks if max_tasks > 0 else 5,
        app_name=app_name if app_name else "Not Available",
        accessibility_tree=accessibility_tree if accessibility_tree else "Not Available",
        action_summary=action_summary if action_summary else "Not Available",
        examples=examples if examples else "None"
    )
    model = MLLM_Model()
    response, _ = model.get_model_response(prompt, [image_path])
    new_tasks = parse_task(response)
    if len(new_tasks) > max_tasks:
        new_tasks = random.sample(new_tasks, max_tasks)
    
    # 对每个任务进行任务指令优化
    optimized_tasks = []
    for task in new_tasks:
        optimize_prompt = PROMPT_TEMPLATE_TASK_OPTIMIZE.format(
            task=task,
            app_name=app_name if app_name else "Not Available"
        )
        response, _ = model.get_model_response(optimize_prompt, [image_path])
        if "<task>" in response and "</task>" in response:
            optimized_task = response.split("<task>")[-1].split("</task>")[0].strip()
            if "No task" in optimized_task or not optimized_task:
                continue    # Task description is too simple, skip.
        else:
            continue

        optimized_tasks.append(optimized_task[0])
        # print(Fore.GREEN + f"Original Task: {task}")
        # print(Fore.CYAN + f"Optimized Task: {optimized_task}")

    return optimized_tasks


def _generate_ui_elements_description_list_full(
    ui_elements: list[UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate description for a list of UIElement using full information.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: Logical screen size.

  Returns:
    Information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if validate_ui_element(ui_element, screen_width_height_px):
      tree_info += f'UI element {index}: {str(ui_element)}\n'
  return tree_info


def get_state(logical_screen_size, ui_elements):
    element_list_text = _generate_ui_elements_description_list_full(
        ui_elements,
        logical_screen_size,
    )
    return element_list_text


def element_to_identifier(element):
    """Converts an element to a JSON-serializable identifier."""
    bbox = getattr(element, 'bbox_pixels', None)
    bbox_dict = {'x_min': bbox.x_min, 'x_max': bbox.x_max, 'y_min': bbox.y_min, 'y_max': bbox.y_max} if bbox else None
    identifier = {
        'resource_id': getattr(element, 'resource_id', None),
        'text': getattr(element, 'text', None),
        'content_description': getattr(element, 'content_description', None),
        'class_name': getattr(element, 'class_name', None),
        'bbox_pixels': bbox_dict,
        'hint_text': getattr(element, 'hint_text', None),
        'is_checkable': getattr(element, 'is_checkable', None),
        'is_enabled': getattr(element, 'is_enabled', None),
        'is_visible': getattr(element, 'is_visible', None),
        'is_clickable': getattr(element, 'is_clickable', None),
        'is_editable': getattr(element, 'is_editable', None),
        'is_focused': getattr(element, 'is_focused', None),
        'is_focusable': getattr(element, 'is_focusable', None),
        'is_long_clickable': getattr(element, 'is_long_clickable', None),
        'is_scrollable': getattr(element, 'is_scrollable', None),
        'is_selected': getattr(element, 'is_selected', None),
        'package_name': getattr(element, 'package_name', None),
        'resource_name': getattr(element, 'resource_name', None),
    }
    return identifier


def filter_interactive_elements(
    elements: list[UIElement], 
    screen_width_height_px, 
    unc_elem_pool=None
):
    interactive_elements = []
    screen_width, screen_height = screen_width_height_px

    # List of excluded package names, adding other keyboard app package names can improve filtering effect
    # Keyboard and system UI (e.g., battery indicator is actually not clickable)
    excluded_packages = {'com.google.android.inputmethod.latin', 'com.android.systemui'}

    for index, element in enumerate(elements):
        if element.package_name in excluded_packages:
            continue
        if element.is_enabled and element.is_visible and element.is_clickable:
            if element.bbox_pixels:
                x_min = element.bbox_pixels.x_min
                x_max = element.bbox_pixels.x_max
                y_min = element.bbox_pixels.y_min
                y_max = element.bbox_pixels.y_max

                # Check if bounding box is within screen bounds and coordinates are valid
                if not (x_min >= x_max or x_min >= screen_width or x_max <= 0 or
                        y_min >= y_max or y_min >= screen_height or y_max <= 0):

                    # Compute element identifier
                    # element_identifier = element_to_identifier(element)
                    # element_identifier_str = json.dumps(element_identifier, sort_keys=True)
                    # if element_identifier_str not in unc_elem_pool:
                    #     interactive_elements.append([index, element])
                    interactive_elements.append([index, element])

    return interactive_elements


def sample_action_element(element, element_list_text):
    index, interactive_element = element
    x, y = interactive_element.bbox_pixels.center
    actions = []
    # Assume all elements are clickable, add click action to action list
    actions.append({"action_type": "click", "x": x, "y": y})
    # actions.append({"action_type": "click", "index": index})
    # If element can be long pressed, add long press action to action list
    if interactive_element.is_long_clickable:
        actions.append({"action_type": "long_press", "x": x, "y": y})
        # actions.append({"action_type": "long_press", "index": index})
    # If multiple actions are available, choose based on given probabilities
    if actions:
        if len(actions) == 1:
            return actions[0]
        else:
            # Choose click or long press with 90% and 10% probability (assuming both actions are available)
            return random.choices(actions, weights=[9, 1], k=1)[0]


def has_screen_changed(before_elements, after_elements):
    # Convert UI elements to identifiers, excluding system UI elements
    before_set = set(
        json.dumps(element_to_identifier(elem), sort_keys=True)
        for elem in before_elements
        if elem.package_name != 'com.android.systemui'
    )
    after_set = set(
        json.dumps(element_to_identifier(elem), sort_keys=True)
        for elem in after_elements
        if elem.package_name != 'com.android.systemui'
    )
    return before_set != after_set


def get_avail_actions(
    package_name,
    device_controller: Device,
    max_breadth: int=10,
    with_addition_actions: bool=False
):
    """
    Get available actions for the current screen state.

    :param package_name: package name of the app
    :param device_controller: device_controller
    :param max_breadth: maximum number of actions to return
    :param with_addition_actions: whether to include additional actions (scroll, navigate back)
    :return: list of available actions
    """
    ui_elements = device_controller._get_ui_elements()
    logical_screen_size = device_controller.get_screen_size()
    # screen = device_controller.get_screenshot()
    element_list_text = get_state(logical_screen_size, ui_elements)
    # Get executable actions and exclude keyboard elements.
    interactive_elements = filter_interactive_elements(ui_elements, logical_screen_size)
    addition_actions = [{"action_type": "scroll", "direction": "down"}]
    all_actions = []

    # 遍历所有可交互元素
    for element in interactive_elements:
        sampled_action = sample_action_element(element, element_list_text)
        if sampled_action:
            all_actions.append(sampled_action)
    # 加上额外动作，默认不加
    if with_addition_actions:
        all_actions.extend(addition_actions)

    if len(all_actions) <= max_breadth:
        selected_actions = all_actions
    else:
        selected_actions = random.sample(all_actions, max_breadth)
    return selected_actions


def get_allysibility_tree(ui_elements):
    allysibility_tree = "{"
    for item in ui_elements:
        bbox_pixels = item["bbox_pixels"]
        x_min = bbox_pixels["x_min"]
        x_max = bbox_pixels["x_max"]
        y_min = bbox_pixels["y_min"]
        y_max = bbox_pixels["y_max"]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        center_str = f"({x_center}, {y_center})"
        for key in ("text", "content_description"):
            val = item.get(key, "")
            if val:
                break
        if not val:
            class_name = item.get("class_name", "")
            package_name = item.get("package_name", "")
            resource_id = item.get("resource_id", "")
            val = f"{class_name} {package_name} {resource_id}".strip()
        allysibility_tree += f"'{val}': {center_str}, "
    allysibility_tree += "}"

    return allysibility_tree


def random_walk(
    device_serial: str,
    package_name: str,
    app_name: str,
    env_id: str,
    max_steps: int = 15,
    max_tasks: int = 5,
    max_breadth: int = 10,
    cur_dir: str = None,
    iteration: int = 0
):
    """
    在移动环境中进行随机游走。

    :param device_serial: 要连接的设备序列号
    :param device_controller: 设备控制器
    :param package_name: 应用的包名
    :param app_name: app名称
    :param max_steps: 随机游走的最大步数
    :param max_tasks: 每一步生成的最大任务数
    :param max_breadth: 每个页面可选择操作的最大数量
    :param cur_dir: 根目录
    :param iteration: 迭代次数

    :return: task_list: 存储所有探索到的任务
    """
    device_controller = Device(device_serial)
    task_list = []
    action_history = []
    summary_history = []
    model = MLLM_Model()

    device_controller.stop_all_apps()
    time.sleep(5.0)
    before_screenshot = device_controller.get_screenshot()
    before_screenshot = np.array(before_screenshot)
    ui_elements_before = device_controller._get_ui_elements()
    screen_size = device_controller.get_screen_size()

    for step in range(max_steps):
        logger.info(f"--- Step {step + 1}/{max_steps} ---")
        if step == 0:   # 第一步先打开app
            device_controller.launch_app(package_name, front=True)
            time.sleep(5.0)
            device_controller.wait_to_stabilize()
        else:   # 获取当前页面的可用操作
            cur_package_name = device_controller.get_top_package_name()
            cur_avail_actions = get_avail_actions(package_name, device_controller, max_breadth, True)
            logger.debug(f"Found {len(cur_avail_actions)} available actions on the current screen.")
            logger.debug(f"Available actions: {cur_avail_actions}")

            if cur_package_name != package_name or not cur_avail_actions:
                logger.warning(
                    f"[{package_name}] Package name changed or No available actions found. "
                    "Restarting the app to continue walk."
                )
                # 如果没有可用操作，则重启应用并清空历史记录，从头开始新的随机路径
                device_controller.stop_all_apps()
                time.sleep(4.0)
                device_controller.launch_app(package_name, front=True)
                time.sleep(4.0)
                device_controller.wait_to_stabilize()
                action_history = []
                continue

            # 随机选择一个操作来执行
            chosen_action = random.choice(cur_avail_actions)
            logger.debug(f"Executing action: {chosen_action}")
            
            # 执行操作
            adb_utils.execute_adb_action(
                action=json_action.JSONAction(**chosen_action),
                device_controller=device_controller, 
                screen_size=screen_size
            )
            time.sleep(2.0)
            action_history.append(chosen_action)        

        # 检查是否到达了应用的新页面, ui element判断 or 图片相似性判断
        ui_elements_after = device_controller._get_ui_elements()
        is_changed = has_screen_changed(ui_elements_before, ui_elements_after)
        if not is_changed:
            logger.info(f"[{package_name}] The screen not change, continue to next step.")
            continue    # 如果页面未变化，跳过任务生成，继续下一步

        # 获取当前状态信息
        after_screenshot = device_controller.get_screenshot()
        img_cur_dir = os.path.join(cur_dir, "screenshots")
        os.makedirs(img_cur_dir, exist_ok=True)
        image_path = os.path.join(img_cur_dir, f"{uuid.uuid4().hex[:20]}.png")
        after_screenshot.save(image_path)

        after_screenshot = np.array(after_screenshot)
        ui_elements = [asdict(ui_element) for ui_element in ui_elements_after]
        accessibility_tree = get_allysibility_tree(ui_elements)

        # 任务生成时添加历史操作总结
        add_screenshot_label(before_screenshot, "before")
        add_screenshot_label(after_screenshot, "after")
        # 添加action信息
        summary, _ = model.get_model_response(
            prompt=PROMPT_TEMPLATE_WALK_SUMMARY,
            images=[
                before_screenshot,
                after_screenshot,
            ],
        )
        summary_history.append(f"step {len(summary_history) + 1}: {summary}")
        action_summary = "\n".join(summary_history)
        # print(Fore.GREEN + action_summary)

        ui_elements_before = ui_elements_after
        before_screenshot = after_screenshot

        # 根据当前状态生成任务
        # 后续合成的时候，例子中加上上一轮合成的任务
        new_tasks = task_generate(
            max_tasks=max_tasks,
            package_name=package_name,
            app_name=app_name,
            image_path=image_path,
            accessibility_tree=accessibility_tree,
            action_summary=action_summary,
        )
        # print(Fore.MAGENTA + f"{(new_tasks)}")
        for task in new_tasks:
            item = {
                "query": task,
                "package_name": package_name,
                "app_name": app_name,
                "env_id": env_id,
                "source_img": image_path,
                "iteration": iteration
            }
            task_list.append(item)
        
        task_file_path = os.path.join(cur_dir, "tasks.json")
        with open(task_file_path, "w", encoding="utf-8") as f:
            json.dump(task_list, f, indent=4, ensure_ascii=False)
        logger.debug(f"Generated {len(new_tasks)} new tasks at step {step + 1}.")

    logger.info(f"[{package_name}] Random walk finished. Total tasks generated: {len(task_list)}")
    device_controller.stop_all_apps()
    time.sleep(5)
    return task_list

def bfs_random_walk(
    device_serial: str,
    package_name: str,
    app_name : str,
    env_id: str,
    max_tasks: int = 15,
    max_breadth: int = 10,
    max_depth: int = 5,
    cur_dir: str = None,
    iteration: int = 0
):
    """
    BFS traversal in mobile environment.

    :param device_serial: device serial to connect
    :param device_controller: device controller
    :param package_name: package name of the app
    :param apk_object: apk object
    :param max_tasks: maximum tasks to generate per level
    :param max_breadth: maximum nodes to expand per level
    :param max_depth: maximum traversal depth
    """
    device_controller = Device(device_serial)
    task_list = []  # 存储探索过的任务
    queue = deque()

    image_dir = os.path.join(cur_dir, "task_src_imgs")
    os.makedirs(image_dir, exist_ok=True)

    # 启动应用
    logger.debug(f"{device_serial} Killing all apps.")
    device_controller.stop_all_apps()
    time.sleep(2.0)
    logger.debug(f"Launching the app: {package_name}")
    device_controller.launch_app(package_name, front=True)
    time.sleep(5.0)
    device_controller.wait_to_stabilize()

    # 初始状态：获取可用操作
    cur_avail_actions = get_avail_actions(package_name, device_controller, max_breadth)
    logger.debug(f"Initial available actions: {cur_avail_actions}")
    # 初始恢复操作 = 打开应用
    # recover_actions = open_app(package_name, device_serial)
    recover_actions = [{"action_type": "open_app", "app_name": app_name}]
    # recover_actions = json_action.JSONAction(**recover_actions)
    # 将 (action, recover_actions, depth) 入队
    for action in cur_avail_actions:
        queue.append((action, recover_actions, 1))
    # pdb.set_trace()
    while queue:
        cur_action, recover_action, depth = queue.popleft()
        logger.info(f"{'>'*depth} Exploring depth={depth}, action={cur_action}")

        # 执行恢复操作（保证环境一致性）
        device_controller.stop_all_apps()   # 重新打开应用
        time.sleep(5.0)
        logger.debug(f"开始执行恢复操作.")
        for act in recover_action:
            logger.debug(f"执行恢复操作: {act}")
            if act['action_type'] == "open_app":
                device_controller.launch_app(package_name, front=True)
                time.sleep(5.0)
                device_controller.wait_to_stabilize()
                continue
            adb_utils.execute_adb_action(
                action=json_action.JSONAction(**act), 
                device_controller=device_controller
            )
            time.sleep(2.0)

        # 执行当前操作
        logger.debug(f"执行当前操作: {cur_action}")
        adb_utils.execute_adb_action(
            action=json_action.JSONAction(**cur_action), 
            device_controller=device_controller
        )
        time.sleep(2.0)

        # 获取当前状态信息
        screenshot_pil = device_controller.get_screenshot()
        image_path = os.path.join(image_dir, f"{uuid.uuid4().hex[:20]}.png")
        screenshot_pil.save(image_path)

        # 根据当前状态生成任务
        new_tasks = task_generate(
            max_tasks=max_tasks,
            package_name=package_name,
            app_name=app_name,
            image_path=image_path,  # 图片保存路径
        )
        for task in new_tasks:
            item = {
                "package_name": package_name,
                "query": task,
                "env_id": env_id,
                "source_img": image_path,
                "iteration": iteration
            }
            task_list.append(item)
        task_file_path = os.path.join(cur_dir, "tasks.json")
        with open(task_file_path, "w", encoding="utf-8") as f:
            json.dump(task_list, f, indent=4, ensure_ascii=False)
        logger.debug(f"{'>'*depth} Generated {len(new_tasks)} new tasks.")

        # 达到最大深度，跳过后续操作
        if depth + 1 > max_depth:
            continue

        # 获取下一层可用 action
        cur_avail_actions = get_avail_actions(package_name, device_controller, max_breadth)
        logger.debug(f"当前页面获得: {len(cur_avail_actions)} 个可用操作")
        # 更新恢复action序列
        next_recover_actions = recover_action + [cur_action]
        logger.debug(f"下一层recover action : {next_recover_actions}")

        # 将下一层 action 入队
        for action in cur_avail_actions:
            queue.append((action, next_recover_actions, depth + 1))
    
    logger.info(f"BFS walk finished. Total tasks generated: {len(task_list)}")
    device_controller.stop_all_apps()
    time.sleep(5)
    return task_list


def collect_task(
    package_name: str,
    device_serial: str,
    env_id: str,
    max_tasks: int = 5,
    max_breadth: int = 10,
    max_depth: int = 5,
    iteration: int = 1,
    output_dir: str = "./collect_task_test",
    walk_type: str = "random"
) -> list:
    """
    :param package_name: package name of the app
    :param device_serial: device serial
    :param env_id: environment id
    :param max_tasks: maximum tasks to generate per level
    :param max_breadth: maximum nodes to expand per level
    :param max_depth: maximum traversal depth
    :param iteration: iteration number
    :param output_dir: output directory
    :param walk_type: random or bfs
    """
    task_list = []
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cur_output_dir = os.path.join(output_dir, package_name, str(iteration))
    os.makedirs(cur_output_dir, exist_ok=True)
    
    app_info_path = os.path.join(output_dir, package_name, "app_info.json")
    app_info = {"app_name": package_name}
    if os.path.exists(app_info_path):
        with open(app_info_path, "r", encoding="utf-8") as file:
            app_info = json.load(file)

    if walk_type == "bfs":
        task_list = bfs_random_walk(
            device_serial=device_serial,
            package_name=package_name,
            app_name=app_info["app_name"],
            env_id=env_id,
            max_tasks=max_tasks,
            max_breadth=max_breadth,
            max_depth=max_depth,
            cur_dir=cur_output_dir,
            iteration=iteration
        )
    elif walk_type == "random":
        task_list = random_walk(
            device_serial=device_serial,
            package_name=package_name,
            app_name=app_info["app_name"],
            env_id=env_id,
            max_tasks=max_tasks,
            max_breadth=max_breadth,
            max_steps=max_depth,
            cur_dir=cur_output_dir,
            iteration=iteration
        )
    else:
        raise ValueError(f"Unsupported walk_type {walk_type!r}; use 'random' or 'bfs'.")
    logger.info(f"{package_name} {device_serial} Auto exploration finished.")

    return task_list


def complexify_task(traj, cur_env_id) -> list:
    """基于上一轮完整轨迹生成长难任务"""
    task_list = []
    model = MLLM_Model()
    prompt_template_map = {
        "task_inherit": PROMPT_TEMPLATE_TASK_INHERIT,
        "task_merge": PROMPT_TEMPLATE_TASK_MERGE,
        "task_change": PROMPT_TEMPLATE_TASK_CHANGE
    }

    for complexify_type in prompt_template_map:
        prompt_template = prompt_template_map[complexify_type]

        prompt = prompt_template.format(
            app_name=traj["app_name"],
            initial_goal=traj["query"],
            action_summary="\n".join(
                step["summary"].split("}.")[-1].strip() for step in traj["step_datas"]
            )
        )
        try:
            response, _ = model.get_model_response(
                prompt=prompt,
                image_paths=[
                    step["before_screenshot"] for step in traj["step_datas"][-3:]
                ],
            )
            if "<task>" in response and "</task>" in response:
                new_task = response.split("<task>")[-1].split("</task>")[0].strip()
                if len(new_task) < 50 or "No task" in new_task:
                    continue    # Task description is too simple, skip.
                # print(Fore.GREEN + f"Original Task: {traj["query"]}")
                # print(Fore.CYAN + f"Complex Task: {new_task}")
            else:
                continue

            iteration = int(traj["iteration"])
            env_id = traj["env_id"]
            if complexify_type == "task_inherit":
                iteration += 1
                env_id = cur_env_id

            item = {
                "query": new_task,
                "package_name": traj["package_name"],
                "app_name": traj["app_name"],
                "device_serial": traj.get("device_serial", ""),
                "env_id": env_id,
                "source_img": traj["step_datas"][-1]["before_screenshot"],
                "source_task": traj["query"],
                "iteration": iteration,
                "is_complexified": True,
                "complexify_type": complexify_type
            }
            task_list.append(item)
        except Exception as exc:
            logger.warning("Failed to generate %s task from %r: %s", complexify_type, traj.get("query"), exc)
            continue

    return task_list
