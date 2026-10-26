import os
import re
import ast
import time
import json
import logging
from PIL import Image
from dataclasses import asdict
from typing import Optional, Any
import numpy as np 
from colorama import init, Fore
init(autoreset=True)

from .utils import json_action
from .utils import adb_utils
from ..common.device import (
    _generate_ui_elements_description_list,
    validate_ui_element,
    add_ui_element_mark,
    _ui_element_logical_corner,
    _logical_to_physical,
    add_screenshot_label,
    Device,
)

from ..common.prompts import ACTION_SELECTION_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE
from ..common.model_client import MLLM_Model
from .utils.utils import convert_action_index_to_coords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42

"""
This module is used to collect trajectories for the GSAR synthesis pipeline.
"""

def extract_json(s: str) -> Optional[dict[str, Any]]:
    """Extracts the first JSON object found in a string.

    Handles multi-line JSON and JSON embedded within other text.

    Args:
      s: A string potentially containing a JSON object.
         E.g., "{'hello': 'world'}" (Python-like) or '"key": "value", "boolean": true, "nothing": null' (Standard JSON) or CoT: "let's think step-by-step, ..., { ... json ... } ... more text"

    Returns:
      The parsed JSON object as a Python dictionary, or None if no valid
      JSON object is found or parsing fails.
    """
    pattern = r'\{.*?\}'
    match = re.search(pattern, s, re.DOTALL)
    if match:
        potential_json_string = match.group()
        try:
            return json.loads(potential_json_string)
        except json.JSONDecodeError as json_error:
            # logger.info(f"JSON parsing failed ({json_error}), attempting Python literal eval.")
            try:
                return ast.literal_eval(potential_json_string)
            except (SyntaxError, ValueError) as eval_error:
                # logger.info(f"Python literal eval also failed ({eval_error}), cannot extract dictionary.")
                return None
    else:
        return None
    

def parse_reason_action_output(
    raw_reason_action_output: str,
) -> tuple[Optional[str], Optional[str]]:
  r"""Parses llm action reason output.

  Args:
    raw_reason_action_output: Raw string output that supposes to have the format
      'Reason: xxx\nAction:xxx'.

  Returns:
    If parsing successfully, returns reason and action.
  """
  reason_result = re.search(
      r'Reason:(.*)Action:', raw_reason_action_output, flags=re.DOTALL
  )
  reason = reason_result.group(1).strip() if reason_result else None
  action_result = re.search(
      r'Action:(.*)', raw_reason_action_output, flags=re.DOTALL
  )
  action = action_result.group(1).strip() if action_result else None
  if action:
    extracted = extract_json(action)
    if extracted is not None:
      action = json.dumps(extracted)

  return reason, action


def _action_selection_prompt(
    prompt_template: str,
    goal: str,
    history: list[str],
    ui_elements: str=None,
    additional_guidelines: list[str] | None = None,
) -> str:
    """Generate the prompt for the action selection.

    Args:
      goal: The current goal.
      history: Summaries for previous steps.
      ui_elements: A list of descriptions for the UI elements.
    additional_guidelines: Task specific guidelines.

    Returns:
      The text prompt for action selection that will be sent to MLLM.
    """
    if history:
        history = "\n".join(history)
    else:
        history = "You just started, no action has been performed yet."

    extra_guidelines = ''
    if additional_guidelines:
        extra_guidelines = 'For The Current Task:\n'
        for guideline in additional_guidelines:
            extra_guidelines += f'- {guideline}\n'

    return prompt_template.format(
        goal=goal,
        history=history,
        ui_elements=ui_elements if ui_elements else "Not available",
        additional_guidelines=extra_guidelines,
    )


def _summarize_prompt(
    action: str,
    reason: str,
    goal: str,
    before_elements: str,
    after_elements: str,
) -> str:
    """Generate the prompt for the summarization step.

    Args:
      action: Action picked.
      reasoning: The reasoning to pick the action.
      goal: The overall goal.
      before_elements: Information for UI elements on the before screenshot.
      after_elements: Information for UI elements on the after screenshot.

    Returns:
      The text prompt for summarization that will be sent to gpt4v.
    """
    return SUMMARY_PROMPT_TEMPLATE.format(
        goal=goal,
        before_elements=before_elements,
        after_elements=after_elements,
        action=action,
        reason=reason,
    )

class M3A_Agent():
    """
    Agent class for run task.

    :param llm: The LLM instance responsible for reasoning and generating actions.
    :param device_serial: The Android device serial identifier used for ADB communication.
    :param action_interval: Seconds to wait for the screen to stablize after executing an action
    """
    def __init__(self, llm: MLLM_Model, device_serial: str, action_interval: float = 2.0,):
        self.llm = llm
        self.device = Device(device_serial=device_serial)
        self.additional_guidelines = None
        self.action_interval = action_interval
        self.history = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = True) -> None:
        if go_home_on_reset:
            self.device.home()
            time.sleep(2.0)
        self.device.stop_all_apps()
        time.sleep(4.0)
        self.history = []

    def run(self, task_goal: str, package_name: str, app_name: str,
        max_rounds: int=20, traj_output_dir: str="./tmp",
    ) -> list:
        # task planning
        # input: source_image, task_goal

        # task execute
        self.reset()
        os.makedirs(traj_output_dir, exist_ok=True)
        step_datas = []

        for i in range(max_rounds):
            step_dir = os.path.join(traj_output_dir, f"step{i + 1:03d}")
            os.makedirs(step_dir, exist_ok=True)
            stop, step_data = self.step(
                goal=task_goal,
                package_name=package_name,
                app_name=app_name,
                step_save_dir=step_dir
            )
            step_data["step"] = i + 1
            step_datas.append(step_data)
            if stop:
                break

        return step_datas

    def step(
        self, 
        goal: str,
        step_save_dir: str,
        package_name: str=None,
        app_name: str=None
    ) -> tuple[bool, dict[str, Any]]:
        step_data = {
            'step': None,
            'top_app_package_name': None,
            'before_screenshot': None,
            'before_screenshot_with_som': None,
            'before_ui_elements_path': None,
            'after_screenshot': None,
            'action_prompt': None,
            'action_output': None,
            'action_output_json': None,
            'action_wo_index': None,
            'action_reason': None,
            'action_raw_response': None,
            'summary_prompt': None, 
            'summary': None,
            'summary_raw_response': None,
        }
        step_index = len(self.history) + 1
        # self.logger.info('----------step ' + str(step_index))

        before_ui_elements = self.device.wait_to_stabilize()
        logical_screen_size = self.device.get_screen_size()
        step_data["logical_screen_size"] = list(logical_screen_size)
        orientation = self.device.get_orientation()
        physical_frame_boundary = self.device.get_physical_frame_boundary()
        before_ui_elements_list = _generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size
        )

        # save current app package name
        top_app_package_name = self.device.get_top_package_name()
        step_data["top_app_package_name"] = top_app_package_name
        # save the before screenshot.
        before_screenshot_pil = self.device.get_screenshot()
        before_screenshot_path = os.path.join(step_save_dir, f"before_screenshot.png")
        before_screenshot_pil.save(before_screenshot_path)
        step_data["before_screenshot"] = before_screenshot_path
        # save the before ui_elements data.
        before_ui_elements_json = [asdict(ui_element) for ui_element in before_ui_elements]
        before_ui_elements_path = os.path.join(step_save_dir, "before_ui_elements.json")
        with open(before_ui_elements_path, "w", encoding="utf-8") as f:
            json.dump(before_ui_elements_json, f, ensure_ascii=False, indent=2)
        step_data["before_ui_elements_path"] = before_ui_elements_path

        # add ui element mark on before screenshot.
        before_screenshot_np = np.array(before_screenshot_pil)
        for index, ui_element in enumerate(before_ui_elements):
            if validate_ui_element(ui_element, logical_screen_size):
                add_ui_element_mark(
                    before_screenshot_np,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )
        
        # save the before screenshot with set-of-mrak.
        before_screenshot_with_som_path = os.path.join(step_save_dir, f"before_screenshot_with_som.png")
        Image.fromarray(before_screenshot_np).save(before_screenshot_with_som_path)
        step_data['before_screenshot_with_som'] = before_screenshot_with_som_path

        if step_index == 1:
            reason = (f"To achieve this goal, I need to first open the {app_name} app.")
            action = json.dumps({"action_type": "open_app", "app_name": app_name})
        else:
            action_prompt = _action_selection_prompt(
                ACTION_SELECTION_PROMPT_TEMPLATE,
                goal,
                [
                    "Step " + str(i + 1) + "- " + step_info["summary"]
                    for i, step_info in enumerate(self.history)
                ],
                before_ui_elements_list,
            )
            step_data["action_prompt"] = action_prompt
            action_output, raw_response = self.llm.get_model_response(
                prompt=action_prompt,
                image_paths=[
                    before_screenshot_path,
                    before_screenshot_with_som_path,
                ]
            )

            if not raw_response:
                raise RuntimeError("Error calling LLM in action selection phase.")
            step_data['action_output'] = action_output
            step_data['action_raw_response'] = raw_response
            reason, action = parse_reason_action_output(action_output)

        # If the output is not in the right format, add it to step summary which
        # will be passed to next step and return.
        if (not reason) or (not action):
            self.logger.info("Action prompt output is not in the correct format.")
            step_data["summary"] = (
                "Output for action selection is not in the correct format, so no"
                " action is performed."
            )
            self.history.append(step_data)
            return (False, step_data)

        # self.logger.info('Action: ' + action)
        # self.logger.info('Reason: ' + reason)
        step_data["action_reason"] = reason

        try:
            converted_action = json_action.JSONAction(
                **extract_json(action),
            )
            step_data["action_output_json"] = str(action)
            step_data["action_wo_index"] = str(action)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Failed to convert the output to a valid action.")
            print(str(e))
            step_data["summary"] = (
                "Can not parse the output to a valid action. Please make sure to pick"
                " the action from the list with required parameters (if any) in the"
                " correct JSON format!"
            )
            self.history.append(step_data)
            return (False, step_data)

        action_index = converted_action.index
        num_ui_elements = len(before_ui_elements)
        if (
            converted_action.action_type
            in ['click', 'long_press', 'input_text', 'scroll']
            and action_index is not None
        ):
            if action_index >= num_ui_elements:
                print(
                    f'Index out of range, prediction index is {action_index}, but the'
                    f' UI element list only has {num_ui_elements} elements.'
                )
                step_data['summary'] = (
                    'The parameter index is out of range. Remember the index must be in'
                    ' the UI element list!'
                )
                self.history.append(step_data)
                return (False, step_data)

            # # Add mark to the target element.
            # add_ui_element_mark(
            #     before_screenshot_np,
            #     before_ui_elements[action_index],
            #     action_index,
            #     logical_screen_size,
            #     physical_frame_boundary,
            #     orientation,
            # )


        if converted_action.action_type == "status":
            if converted_action.goal_status == "infeasible":
                self.logger.info("Agent stopped since it thinks mission impossible.")
                step_data["summary"] = "Agent thinks the mission is infeasible and stopped."
            else:
                step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return (True, step_data)

        if converted_action.action_type == "answer":
            self.logger.info("Agent answered with: %s", converted_action.text or "")
        stop_after_action = converted_action.action_type == "answer"

        try:
            if converted_action.action_type == "open_app":
                self.device.launch_app(package_name)
                actual_action_coordinates = None
            else:
                actual_action_coordinates = adb_utils.execute_adb_action(
                    converted_action,
                    self.device,
                    before_ui_elements,
                    logical_screen_size,
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(Fore.YELLOW + "Failed to execute action.")
            print(str(e))
            step_data["summary"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            self.history.append(step_data)
            return (False, step_data)
        
        try:
            if actual_action_coordinates:
                action_wo_index = convert_action_index_to_coords(converted_action, actual_action_coordinates)
                step_data["action_wo_index"] = str(action_wo_index)
        except Exception as e:
            print(Fore.YELLOW + "Failed to convert action.")
            print(str(e))

        time.sleep(self.action_interval)
        after_ui_elements = self.device.wait_to_stabilize()
        logical_screen_size = self.device.get_screen_size()
        orientation = self.device.get_orientation()
        physical_frame_boundary = self.device.get_physical_frame_boundary()
        after_ui_elements_list = _generate_ui_elements_description_list(
            after_ui_elements, logical_screen_size
        )
        # save the after screenshot
        after_screenshot_pil = self.device.get_screenshot()
        after_screenshot_path = os.path.join(step_save_dir, f"after_screenshot.png")
        after_screenshot_pil.save(after_screenshot_path)
        step_data["after_screenshot"] = after_screenshot_path

        # add ui element mark on after screenshot.
        after_screenshot_np = np.array(after_screenshot_pil)
        for index, ui_element in enumerate(after_ui_elements):
            if validate_ui_element(ui_element, logical_screen_size):
                add_ui_element_mark(
                    after_screenshot_np,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )

        add_screenshot_label(before_screenshot_np, "before")
        add_screenshot_label(after_screenshot_np, "after")
        summary_prompt = _summarize_prompt(
            action,
            reason,
            goal,
            before_ui_elements_list,
            after_ui_elements_list,
        )
        summary, raw_response = self.llm.get_model_response(
            prompt=summary_prompt,
            images=[
                before_screenshot_np,
                after_screenshot_np,
            ],
        )

        if not raw_response:
            self.logger.info(
                'Error calling LLM in summarization phase. This should not happen: '
                f'{summary}'
            )
            step_data['summary'] = (
                'Some error occurred calling LLM during summarization phase: %s'
                % summary
            )
            self.history.append(step_data)
            return (False, step_data)
        
        step_data['summary_prompt'] = summary_prompt
        step_data['summary'] = f'Action selected: {action}. {summary}'
        # self.logger.info('Summary: ' + summary)
        step_data['summary_raw_response'] = raw_response

        self.history.append(step_data)
        return (stop_after_action, step_data)


def run_single_task(
    query: str,
    package_name: str,
    app_name: str,
    device_serial: str,
    max_step: int,
    traj_output_dir: str
):
    logger.debug(f"[{device_serial}] 运行任务: {query}")
    model = MLLM_Model()
    agent = M3A_Agent(llm=model, device_serial=device_serial)

    step_datas = agent.run(
        task_goal=query,
        package_name=package_name,
        app_name=app_name,
        max_rounds=max_step,
        traj_output_dir=traj_output_dir
    )
    return step_datas


def collect_traj(
    task_list: list,
    device_serial: str,
    max_step: int=10,
    add_step: int=5,
    output_dir: str="./tmp",
):
    """根据task收集轨迹数据"""
    traj_list = []
    task_list_copy = [item.copy() for item in task_list]
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Collecting trajectories for {len(task_list_copy)} tasks.")
    for idx, item in enumerate(task_list_copy):
        query = item["query"]
        package_name = item["package_name"]
        app_name = item["app_name"]
        iteration = item["iteration"]
        traj_output_dir = os.path.join(output_dir, package_name, f"iteration_{iteration}", f"task_{idx}")

        if item.get("is_complexified", False):
            traj_max_step = max_step + add_step
        else:
            traj_max_step = max_step
        traj_info = run_single_task(
            query=query,
            package_name=package_name,
            app_name=app_name,
            device_serial=device_serial,
            max_step=traj_max_step,
            traj_output_dir=traj_output_dir
        )
        item["step_nums"] = len(traj_info)
        item["step_datas"] = traj_info
        traj_list.append(item)
        time.sleep(2.0)

    return traj_list
