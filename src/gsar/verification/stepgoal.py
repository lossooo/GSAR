import re
import io
import base64
from openai import OpenAI
from typing import Union
from tenacity import retry, stop_after_attempt, wait_fixed, wait_chain
from PIL import Image
from colorama import Fore, init
init(autoreset=True)

from ..common.model_client import OPENAI_CONFIG, create_openai_client
from .utils import encode_image_to_data_url

verify_template_wo_anchor = """You are an expert in evaluating the performance of a GUI agent and in GUI task understanding.

Your task is to determine whether the GUI interface has completed the specified task by jointly reasoning over:
- User Instruction: A natural language description of the GUI task to be completed.
- Action History: The chronological list of actions taken by the agent.
- Current Image: The GUI image representing the current state (displayed at the top as "current image").
- Reference Image: A reference image showing the completed task state (displayed at the top as "reference image").

Task Objective:
Determine whether the current image satisfies all task requirements defined by the user instruction, using the reference image as the completion standard.

Evaluation Principles:
1. Treat each key UI element in the reference image as a key task condition (sub-task) required for completion.
   These conditions may correspond to:
   - Reaching a specific page or screen
   - Presence or absence of specific UI elements
   - Correct states of switches, buttons, or checkboxes
   - Correct text content, selections, or configurations

2. For each key condition:
   - Check whether the corresponding element exists in the current image
   - Verify whether its semantic state matches the one shown in the reference image
   - Ignore differences in position, layout, scale, or style

3. Use the agent's action history to assess action sufficiency:
   - If a condition logically requires navigation, sorting, filtering, or confirmation steps,
     and these steps are missing or incomplete, the condition should be considered not satisfied
   - Visual similarity alone is not sufficient if the required interaction steps were skipped

4. If all key conditions indicated by the reference image are satisfied in the current image,
   the task should be judged as completed, even if the overall layout differs.

5. If any required condition is missing, incorrect, or ambiguous, the task should be judged as not completed.

Output Requirements:
- First explain your reasoning by referencing which key conditions are satisfied or not
- Then output a final binary judgment. Output only one of the following judgment results: "yes" or "no".

Output Format:
Reasoning Process: Explain how the current image compares to the reference image with respect to each key condition.
Judgment Result: [yes or no]

Input:
User Instruction: {query}
Action History: {action_history}

"""


verify_template = """You are an expert in evaluating the performance of a GUI agent and in GUI task understanding.

Your task is to determine whether the GUI interface has completed the specified task by jointly reasoning over:
- User Instruction: A natural language description of the GUI task to be completed.
- Action History: The chronological list of actions taken by the agent.
- Current Image: The GUI image representing the current state (displayed at the top as "current image").
- Labeled Image: A reference image showing the completed task state, where key elements related to the instruction are highlighted with red bounding boxes (displayed at the top as "label image").

Task Objective:
Determine whether the current image satisfies all task requirements defined by the user instruction, using the labeled image as the completion standard.

Evaluation Principles:
1. Treat each red bounding box in the labeled image as a key task condition (sub-task) required for completion.
   These conditions may correspond to:
   - Reaching a specific page or screen
   - Presence or absence of specific UI elements
   - Correct states of switches, buttons, or checkboxes
   - Correct text content, selections, or configurations

2. For each key condition:
   - Check whether the corresponding element exists in the current image
   - Verify whether its semantic state matches the one shown in the labeled image
   - Ignore differences in position, layout, scale, or style

3. Use the agent's action history to assess action sufficiency:
   - If a condition logically requires navigation, sorting, filtering, or confirmation steps,
     and these steps are missing or incomplete, the condition should be considered not satisfied
   - Visual similarity alone is not sufficient if the required interaction steps were skipped

4. If all key conditions indicated by the labeled image are satisfied in the current image,
   the task should be judged as completed, even if the overall layout differs.

5. If any required condition is missing, incorrect, or ambiguous, the task should be judged as not completed.

Output Requirements:
- First explain your reasoning by referencing which key conditions (red-box elements) are satisfied or not
- Then output a final binary judgment. Output only one of the following judgment results: "yes" or "no".

Output Format:
Reasoning Process: Explain how the current image compares to the labeled image with respect to each key condition.
Judgment Result: [yes or no]

Input:
User Instruction: {query}
Action History: {action_history}

"""


@retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] + [wait_fixed(3) for i in range(2)] + [wait_fixed(5)]),
         stop=stop_after_attempt(5))
def call_openai_compatible(client: OpenAI, model_name, prompt, current_img, label_img):

    current_img_url = encode_image_to_data_url(current_img)
    label_img_url = encode_image_to_data_url(label_img)
    user_content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "\nCurrent Image:"},
        {"type": "image_url","image_url": {"url": current_img_url}},
        {"type": "text", "text": "\nLabel Image:"},
        {"type": "image_url","image_url": {"url": label_img_url}},
    ]
    messages = [
        {"role": "user", "content": user_content}
    ]
    # print(Fore.GREEN + f"{json.dumps(messages)}")

    response = client.chat.completions.create(
        model=model_name,
        messages = messages,
        temperature=0.5,
        top_p=0.5
    )

    return response.choices[0].message.content


class EndResultEvaluator:
    def __init__(self):
        self.model_name = OPENAI_CONFIG["model_name"]
        self.client = create_openai_client()

    def __call__(self, task, action_list, current_img, label_img, label_state):
        action_history = '\n'.join(action_list)
        prompt = verify_template_wo_anchor.format(query=task, action_history=action_history)            

        max_retry = 3
        for attempt in range(1, max_retry + 1):
            response_text = call_openai_compatible(self.client, self.model_name, prompt, current_img, label_img)
            # print("response:" + Fore.YELLOW + response_text)

            # matches = re.findall(r'Judgment Result:[\s\S]*?(yes|no)', response_text, re.DOTALL)
            matches = re.findall(r'(yes|no)', response_text.lower(), re.DOTALL)
            if matches:
                return 1 if matches[-1] == "yes" else 0

        print(f'Failed to verify the task: {task}, skip it.')
        return -2
