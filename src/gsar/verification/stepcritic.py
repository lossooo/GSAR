"""
Code based on https://github.com/THUDM/AndroidGen/blob/main/model/judge/judge.py
"""
import re
import os
import json
import sys
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed, wait_chain
import base64
from PIL import Image
from colorama import Fore, init
init(autoreset=True)

from .utils import (
    dict_to_ui_element,
    _generate_ui_elements_description_list_full
)
from ..common.model_client import OPENAI_CONFIG, create_openai_client
from .utils import encode_image_to_data_url


system_prompt = '''You are an expert in evaluating the performance of an android agent. The agent is designed to help a human user navigate on their device to complete a task. Given the user's intent, the agent's action history, the final state of the device, and the agent's response to the user, your goal is to list the conditions the task requires and their respective completion step (or -1 for not completed). There are two types of tasks:
1. Information seeking: The user wants to obtain certain information from the screen, such as information about a product, reviews, map info, comparison of map routes, etc. The bot's response must contain the information the user wants or explicitly state that the information is not available. Otherwise, e.g., if the bot encounters an exception and responds with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked to list the top-searched items in a shop, the agent should order the items by the number of searches and then return the top items. If the ordering action is missing, the task is likely to fail.
2. Application Operation: The user wants to do operations in a specific application, such as purchasing, modifying content or configuration, starting a project, etc. Carefully examine the bot's action history and the final state of the page to determine whether the bot completes the task. No need to consider the bot's response.

*IMPORTANT* Your output should STRICTLY follow the format below and DONOT output other content:
```
"condition1": "completion step1 (or -1 for not completed)"
"condition2": "completion step2 (or -1 for not completed)"
...
```'''


verify_template = '''# User Intent: %s

# Action History:
%s

# The detailed final state of the screen:
```
%s
```'''


example_input = '''# User Intent: Tell me the submission page limit of ICLR 2024.

# Action History:
## Round 0
* Analysis: The user want to find the submission page limit of ICLR 2024. This is an information seeking task, and I should first try search on Google.
* Operation:

```
open_app(app_name="Chrome")
```
## Round 1
* Confirm: The screen shows that the browser has opened Google.
* Analysis: I should type "ICLR 2024 submission page limit" to search.
* Operation:

```
do(action="Type", argument="ICLR 2024 submission page limit", element_id="16") # the Google search bar in the center
```
## Round 2
* Confirm: The search bar has been filled with the query.
* Analysis: I should press Enter to search.
* Operation:

```
do(action="Press Enter") # Press the Enter key
```
## Round 3
* Confirm: The screen shows that Google returns many related results, indicating that the search is successful.
* Analysis: I think the result "2024 - Call For Papers" is likely to contain the submission page limit of ICLR 2024. So I should click it.
* Operation:

```
do(action="Click", argument=None, element_id="19") # the link '2024 - Call For Papers' from ICLR at top left
```
## Round 4
* Confirm: The screen shows that the browser has opened the Call for Papers of ICLR 2024.
* Analysis: The screen does not yet contain the submission page limit. I should try scrolling down until I find detailed requirement about page limit, or reach the end of page.
* Operation:

```
do(action="Scroll Down")    
```
## Round 5
* Confirm: The screen shows that the scrolling has been performed.
* Analysis: The screen shows that the strict page limit is 9 page in ICLR 2024's submission. This should satisfy user's initial instruction. The task is ended.
* Operation:

```
exit(message="According to the Call for Papers, the strict page limit is 9 in ICLR 2024.")
```

# The detailed final state of the screen:
```
<element id="0" class="TextView" resource-name="com.android.chrome:id/status_text"> No internet connection </element>
<element id="1" class="ImageButton" resource-name="com.android.chrome:id/home_button" clickable> Home </element>
<element id="2" class="ImageButton" resource-name="com.android.chrome:id/tab_switcher_button" clickable> Switch or close tabs </element>
<element id="3" class="WebView"> Call for Papers </element>
<element id="4" class="EditText" resource-name="com.android.chrome:id/url_bar" clickable editable> iclr.cc/Conferences/2024/CallForPapers </element>
<element id="5" class="TextView"> This year we are asking authors to submit paper abstracts by the abstract submission deadline of Sept 23, 2023. Please note that no changes on the authors and their orders can be made after the abstract submission deadline. Also, please make sure that all authors have an OpenReview profile with the latest information.  Abstracts submitted by the abstract submission deadline must be genuine; placeholder or duplicate abstracts will be removed. </element>
<element id="6" class="TextView"> There will be a strict upper limit of 9 pages for the main text of the submission, with unlimited additional pages for citations. This page limit applies to both the initial and final camera ready version. </element>
<element id="7" class="TextView"> Style files and Templates </element>
<element id="8" class="TextView"> To prepare your submission to ICLR 2024, please use the LaTeX style files provided at: </element>
<element id="9" class="TextView"> Reviewing Process </element>
<element id="10" class="TextView"> The full paper submission deadline is Sept 28, 2023 11:59pm AOE. Abstracts and papers must be submitted using the conference submission system at: </element>
<element id="11" class="View" clickable>  https://openreview.net/group?id=ICLR.cc/2024/Conference </element>
<element id="12" class="TextView"> . The submission site will be open on September 15, 2023.  Supplementary material is due at the same time as the main paper. </element>
```'''


example_output = '''# Conditions: 
Search for "ICLR 2024 submission": 2
Find the requirement of submission page limit: 5
Answer the user's question: 5'''


def verify(judger, env, task, action_list, state_list):
    
    action_history = '\n'.join(action_list)
    
    last_state = state_list[-1]
    
    query = verify_template % (task, action_history, last_state)
    
    history = [
        {
            "role": "user",
            "content": {
                "text": example_input
            }
        },
        {
            "role": "assistant",
            "content": {
                "text": example_output
            }
        }
    ]
    
    for _ in range(3):
        try:
            output = judger.generate(query, system_prompt=system_prompt, history=history).strip()
        except Exception as e:
            print(e)
            continue
        
        conditions = []
        for condition in output.split('\n'):
            match_contidion = re.match(r'(.+): (-?\d+)', condition)
            if match_contidion:
                conditions.append((match_contidion.group(1).strip(' \n\"'), int(match_contidion.group(2))))
        
        if conditions:
            return -1 not in [step for _, step in conditions], conditions
    
    print(f'Failed to verify the task: {task}, skip it.')
    return None, None


def get_id(record):
    env = record[0]['env']
    if env == 'android':
        return record[0]['app']
    else:
        raise ValueError(f'Unsupported environment: {env}')


def load_data(data_path):
    with open(data_path, 'r') as f:
        record = [json.loads(line) for line in f.readlines()]
    
    env = record[0]['env']
    task = record[0]['instruction']
    init_state = record[0]['app']
    action_list = [step['response'] for step in record]
    state_list = [step['state'] for step in record]
    
    return env, task, init_state, action_list, state_list


def format_example(task, action_list):
    example = f'# A reference example:\n<|user|>\nUser Instruction: {task}'
    for i, action in enumerate(action_list[:20]):
        example += f'\n\n<|user|>\n** Environment State (Omitted) **\n\n<|assistant|>\n{action}'
    
    return example


@retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] + [wait_fixed(3) for i in range(2)] + [wait_fixed(5)]),
         stop=stop_after_attempt(5))
def call_openai_compatible(client: OpenAI, model_name, system_prompt, prompt, history=None):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(
            {"role": rec["role"], "content": rec["content"]["text"]}
            for rec in history
        )

    messages.append({
        "role": "user",
        "content": prompt if isinstance(prompt, str) else prompt["text"]
    })
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

    def __call__(self, task, action_list, state_list, image_path):
        action_history = '\n'.join(action_list)
        # last_state = state_list[-1]

        # Convert a list of UI element dicts into a formatted string of UI elements.
        with Image.open(image_path) as img:
            screen_width_height_px = img.size  # 返回 (width, height)
        ui_elements_list = [dict_to_ui_element(el) for el in state_list]
        last_state = _generate_ui_elements_description_list_full(
            ui_elements_list, 
            screen_width_height_px
        )

        prompt = verify_template % (task, action_history, last_state)

        # 构造 history
        history = [
            {"role": "user", "content": {"text": example_input}},
            {"role": "assistant", "content": {"text": example_output}}
        ]

        max_retry = 3
        for attempt in range(1, max_retry + 1):
            response_text = call_openai_compatible(self.client, self.model_name, system_prompt, prompt, history)
            # print(Fore.YELLOW + response_text)

            conditions = []
            for condition in response_text.split('\n'):
                # match_condition = re.match(r'(.+): (-?\d+)', condition)
                # match_condition = re.match(r'"?(.+?)"?\s*:\s*"?(-?\d+)"?', condition)
                match_condition = re.match(r'"?(.+?)"?\s*:\s*.*?(-?\d+)', condition)

                if match_condition:
                    conditions.append((match_condition.group(1).strip(' \n"'), int(match_condition.group(2))))

            if conditions:
                task_complete = -1 not in [step for _, step in conditions]
                # print(task_complete)
                return task_complete

        print(f'Failed to verify the task: {task}, skip it.')
        return -2
