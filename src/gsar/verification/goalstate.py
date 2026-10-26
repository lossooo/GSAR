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


verify_template = """You are an expert in GUI task understanding. Your task is to determine whether a GUI interface has completed the specified task based on the instruction, the labeled image, and the current image, and to explain your reasoning.

Input Description:
User Instruction: A natural language description of the GUI task to be performed.
Current Image: The GUI image representing the current state (displayed at the top as "current image").
Labeled Image: A reference image showing the completed task state, where key elements related to the instruction are highlighted with red bounding boxes (displayed at the top as "label image").

Task Objective:
- Determine whether the current image has reached the completion state required by the instruction.

Requirements:
1. Base your judgment on whether the key elements related to the instruction in the labeled image have the same states as the corresponding key elements in the current image. Pay attention to factors such as the presence of elements, button on/off states, whether the specified page has been reached, and whether all instruction-required details are satisfied.
2. The positions of key elements may differ between the two images. Ignore positional differences and focus only on whether the elements satisfy the instruction requirements.
3. If the current image differs significantly from the labeled image but still completes the instruction task, output "yes".
4. Output only one of the following judgment results: "yes" or "no".

Output Format:
Reasoning Process: [your reasoning]
Judgment Result: [yes or no]

Input:
User Instruction: {query}
"""

verify_template_wo_anchor = """You are an expert in GUI task understanding. Your task is to determine whether a GUI interface has completed the specified task based on the instruction, the reference image, and the current image, and to explain your reasoning.

Input Description:
User Instruction: A natural language description of the GUI task to be performed.
Current Image: The GUI image representing the current state (displayed at the top as "current image").
Reference Image: A reference image showing the completed task state (displayed at the top as "reference image").

Task Objective:
- Determine whether the current image has reached the completion state required by the instruction.

Requirements:
1. Base your judgment on whether the key elements related to the instruction in the reference image have the same states as the corresponding key elements in the current image. Pay attention to factors such as the presence of elements, button on/off states, whether the specified page has been reached, and whether all instruction-required details are satisfied.
2. The positions of key elements may differ between the two images. Ignore positional differences and focus only on whether the elements satisfy the instruction requirements.
3. If the layout of the reference image and the current image differs significantly but both still fulfill the instruction, output "yes".
4. Output only one of the following judgment results: "yes" or "no".

Output Format:
Reasoning Process: [your reasoning]
Judgment Result: [yes or no]

Input:
User Instruction: {query}
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

    def __call__(self, task, current_img, label_img, label_state):
        prompt = verify_template.format(query=task) # with anchor
        # prompt = verify_template_wo_anchor.format(query=task)   # without anchor

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
