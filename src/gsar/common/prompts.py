# INPUTS: schema_info
PROMPT_TEMPLATE_FIND_FILE_FIELDS = """
You are given information about one or more SQLite databases, including:
- Database name
- Table names
- Table schemas
- Example rows of data from each table
Your task:
1. Identify fields that contain paths to external files required by the app state.
2. For each suitable table:
   - Return the database name
   - Return the table name
   - Return the field/column that stores the file path
   - Return the file suffix (extension) that matches the existing records in that table, only if examples exist.
3. If no suitable tables are found, return an empty list.

Rules:
- Use the sample data to infer the correct file type and suffix (e.g., `.jpg`, `.mp3`, `.txt`, `.mp4`).
- Only output tables that actually store file records or paths.
- If a table has no relevant file data, do not include it in the output.

Input format:
{db_info}

Output format (JSON list of objects):
[
  {{
    "database": "<database_name>",
    "table": "<table_name>",
    "field": "<field_name>",
    "suffix": "<file_suffix>"
  }},
  ...
]
"""


PROMPT_TEMPLATE_WALK_SUMMARY = """You are given two screenshots of a mobile app: one before an action, and one after. 
Determine what action the user performed. 

Requirements:
1. Base your answer only on visible UI changes.
2. Summarize the action as a specific user operation (e.g., click, type, toggle, create, delete, rename, scroll, open page).
3. Give a short, factual description.
4. If multiple changes occurred, summarize them as a single meaningful operation.
5. Do not guess hidden behavior or invent items.

Output:
- One concise sentence describing the action (under 50 words).

"""


DEFAULT_EXAMPLES = """
1. Create a “Team Meeting” event for the upcoming Friday at 3:00 PM in Simple Calendar Pro, set it to repeat weekly, and add a reminder 15 minutes before.
2. Add a new contact named Sophie Zelmani with phone number 123-456-7890 and email emily.johnson@example.com in the Contacts app, and label it as 'Work'.
3. Launch Markor and rename the file 'test.md' to 'hello.md', and then write 'Hello, World!' in the top of the note.
4. Change the recording format to 'Wav' in Audio Recorder and then record a short audio clip, name it 'goodbye' and save it.
5. Open the Settings app, turn off vibration & haptics, and adjust the screen brightness to somewhere between 40% and 60%.
6. In Broccli, set the preferred category to Favorites, and then create a new recipe named 'Pasta Salad' with ingredients 'pasta, tomatoes, cucumbers, olive oil' and directions 'Boil pasta, chop vegetables, mix with olive oil'.
"""


# INPUTS: app_name, screenshot, accessibility_tree, action_summary, examples
PROMPT_TEMPLATE_TASK_GENERATOR = """You are an expert in analyzing mobile app user interfaces and generating clear, purposeful, and executable task instructions.
You will receive:
- The app name
- A current mobile screenshot
- UI component information extracted from the screenshot
- The action history from app launch to the current screen

Your goal is to analyze the current user interface and generate purposeful instructions based on the action history, representing the tasks a human user is likely to perform next.

Task types:
- Task-Oriented: A series of operations to achieve a specific goal.
- Question-Oriented: A series of operations that lead to answering a specific question.

Requirements:
- Each task must mention the app name and provide a high-level instruction without referencing any specific UI elements.
- You can combine multiple simple tasks (from the example tasks) into richer, more complex, and more diverse tasks.
- If the task involves entering information, specify the exact content to be input.
- DO NOT use words such as 'or', 'any', 'e.g.', or any other expression that makes the task description vague or ambiguous.
- DO NOT mention any non-existent files.
- DO NOT generate tasks related to user login, account registration, account creation, or authentication flows.
- Do not generate any tasks if the screen is a permission dialog, an app initialization popup, or unrelated to the current app.
- Generate between 0 and {max_tasks} tasks depending on the complexity of the screen and action history.

Context:
- App name: {app_name}
- UI components: {accessibility_tree}
- Action history: {action_summary}

Example tasks: {examples}

Output format (strictly follow this format):
your analysis:
- A brief explanation of the navigation history, user intent, and key UI elements that affect the tasks.
your tasks:
1. Task description
2. Task description
3. Task description
...
Do not add any extra sections or commentary outside this format.
"""


# # INPUTS: app_name, task, screenshot
PROMPT_TEMPLATE_TASK_OPTIMIZE = """You are a task instruction optimization expert, skilled at transforming task descriptions into clearer, specific, and executable versions.

You will receive:
- The app name
- An initial task instruction
- A current mobile screenshot

Your goal is to analyze the given task instruction and the current user interface, then generate an optimized version of the task instruction.

Input:
- App name: {app_name}
- Current screenshot: <image>
- Initial task instruction: {task}

Requirements:
- Use natural language and high-level actions without referencing specific UI elements (such as buttons, icons, or menu paths).
- Clearly identify the task's primary subject, which can be a contact, file, project, settings or navigation page.
- Specify the content, attributes, or metadata that need to be created or modified.
- Fill in missing contextual information when appropriate (such as required app names, field purposes, or data types).
- Combine all required operations into a single, directly executable natural-language task instruction.

Your analysis:
- A brief explanation of how the optimized task improves clarity, completeness, and executability compared to the original instruction.
Output task:
<task>
(contain only a single task; if no task needs to be generated, write "No task")
</task>

Do not add any extra sections or commentary outside this format.

"""


# INPUTS: initial_goal, action_summary, last_few_screenshots
PROMPT_TEMPLATE_TASK_COMPLEXIFY = """You are an expert at generating complex mobile app tasks based on existing task.  
You will receive:
- A user task trajectory, including the initial user goal, the last few screenshots, and a sequence of action summaries taken to achieve it.
Your goal is to analyze the initial task and generate a new task that is more complex than the original task by adding additional step or subtask to extend the original task.


Requirements:
- The task must include the original task and add additional step or subtask to form a more complex task.
- The task must mention the app name and provide a high-level instruction without referencing any specific UI elements.
- If the task involves entering information, specify the exact content to be input.
- DO NOT use words such as 'or', 'any', 'e.g.', or any other expression that makes the task description vague or ambiguous.
- DO NOT mention any non-existent files.

Input:
- App name: {app_name}
- Initial user goal: {initial_goal}
- Action summary: {action_summary}
- Last few screenshots: <image>

Output format (strictly follow this format):
Your analysis:
- A brief explanation of which steps or subtasks could be added to make the task more complete or complex.
Output task:
- 1. Task description (Must start with "1. " and contain only a single task)

Do not add any extra sections or commentary outside this format.
"""

PROMPT_TEMPLATE_TASK_INHERIT = """
You are an expert at generating follow-up mobile app tasks based on an existing completed task.

You will receive:
- A user task trajectory, including the initial user goal, the last few screenshots, and a sequence of action summaries that indicate the task has already been completed.

Your goal is to generate a new task that STARTS FROM the environment state after the previous task is finished.
The new task should build upon the existing result and introduce additional steps or subtasks to continue the workflow.

Requirements:
- The new task must assume the previous task has already been successfully completed.
- The task must start from the current environment state shown in the screenshots.
- The task should extend the workflow by adding new steps or subtasks, without repeating the original task.
- The task must mention the app name and provide a high-level instruction without referencing any specific UI elements.
- If the task involves entering information, specify the exact content to be input.
- DO NOT use words such as 'or', 'any', 'e.g.', or any other expression that makes the task description vague or ambiguous.
- DO NOT mention any non-existent files.

Input:
- App name: {app_name}
- Previous completed user goal: {initial_goal}
- Action summary: {action_summary}
- Current environment screenshots: <image>

Output format (strictly follow this format):
Your analysis:
- A brief explanation of how the new task continues from the completed state and what new steps are added.
Output task:
<task>
(contain only a single task; if no task needs to be generated, write "No task")
</task>

Do not add any extra sections or commentary outside this format.
"""


PROMPT_TEMPLATE_TASK_MERGE = """
You are an expert at generating more complex mobile app tasks by merging an existing task with additional subtasks.

You will receive:
- A user task trajectory, including the initial user goal, the last few screenshots, and a sequence of action summaries.

Your goal is to generate a NEW combined task that integrates the original task and newly added steps into a single, more complex task.
The initial environment of the new task corresponds to the environment state shown in the screenshots.

Requirements:
- The new task must explicitly include the original task goal and extend it with additional steps or subtasks.
- The task should describe a complete, merged workflow as a single high-level user goal.
- The task must mention the app name and provide a high-level instruction without referencing any specific UI elements.
- If the task involves entering information, specify the exact content to be input.
- DO NOT use words such as 'or', 'any', 'e.g.', or any other expression that makes the task description vague or ambiguous.
- DO NOT mention any non-existent files.

Input:
- App name: {app_name}
- Original user goal: {initial_goal}
- Action summary: {action_summary}
- Initial environment screenshots: <image>

Output format (strictly follow this format):
Your analysis:
- A brief explanation of how the original task and new subtasks are merged into a more complex task.
Output task:
<task>
(contain only a single task; if no task needs to be generated, write "No task")
</task>

Do not add any extra sections or commentary outside this format.
"""

PROMPT_TEMPLATE_TASK_CHANGE = """
You are an expert at generating variant mobile app tasks by modifying parameters in an existing task instruction.

You will receive:
- A user task trajectory, including the initial user goal, the last few screenshots, and a sequence of action summaries.

Your goal is to generate a new task that has the SAME task structure as the original task,
but differs by changing specific parameters such as names, values, text content, or identifiers.

Requirements:
- The new task must keep the same overall workflow and steps as the original task.
- Only modify concrete parameters in the task instruction, such as file names, text content, titles, numbers, or labels.
- Do NOT add new steps or remove existing steps.
- The task must mention the app name and provide a high-level instruction without referencing any specific UI elements.
- If the task involves entering information, specify the exact content to be input.
- DO NOT use words such as 'or', 'any', 'e.g.', or any other expression that makes the task description vague or ambiguous.
- DO NOT mention any non-existent files.

Input:
- App name: {app_name}
- Original user goal: {initial_goal}
- Action summary: {action_summary}
- Environment screenshots: <image>

Output format (strictly follow this format):
Your analysis:
- A brief explanation of which parameters are modified and how they differ from the original task.
Output task:
<task>
(contain only a single task; if no task needs to be generated, write "No task")
</task>

Do not add any extra sections or commentary outside this format.
"""


# INPUTS: goal, history, ui_elements, knowledge
PROMPT_TEMPLATE_EXPLORE_ACTION = """## Role Definition
You are an Android operation AI that fulfills user requests through precise screen interactions.
The current screenshot and the same screenshot with bounding boxes and labels added are also given to you.

## Action Catalog
Available actions (STRICT JSON FORMAT REQUIRED):
1. Status Operations:
   - Task Complete: {{"action_type": "status", "goal_status": "complete"}}
   - Task Infeasible: {{"action_type": "status", "goal_status": "infeasible"}}
2. Information Actions:
   - Answer Question: {{"action_type": "answer", "text": "<answer_text>"}}
3. Screen Interactions:
   - Tap Element: {{"action_type": "click", "index": <visible_index>}}
   - Long Press: {{"action_type": "long_press", "index": <visible_index>}}
   - Scroll: Scroll the screen or a specific scrollable UI element. Use the `index` of the target element if scrolling a specific element, or omit `index` to scroll the whole screen. {{"action_type": "scroll", "direction": <"up"|"down"|"left"|"right">, "index": <optional_target_index>}}
4. Input Operations:
   - Text Entry: {{"action_type": "input_text", "text": "<content>", "index": <text_field_index>}}
   - Keyboard Enter: {{"action_type": "keyboard_enter"}}
5. Navigation:
   - Home Screen: {{"action_type": "navigate_home"}}
   - Back Navigation: {{"action_type": "navigate_back"}}
6. System Actions:
   - Launch App: {{"action_type": "open_app", "app_name": "<exact_name>"}}
   - Wait Refresh: {{"action_type": "wait"}}

## Current Objective
User Goal: {goal}

## Execution Context
Action History:
{history}

Visible UI Elements (Only interact with *visible=true elements):
{ui_elements}

## Core Strategy
1. Path Optimization:
   - Prefer direct methods (e.g., open_app > app drawer navigation)
   - Always use the `input_text` action for entering text into designated text fields.
   - Verify element visibility (`visible=true`) before attempting any interaction (click, long_press, input_text). Do not interact with elements marked `visible=false`.
   - Use `scroll` when necessary to bring off-screen elements into view. Prioritize scrolling specific containers (`index` provided) over full-screen scrolls if possible.

2. Error Handling Protocol:
   - Switch approach after ≥ 2 failed attempts
   - Prioritize scrolling (`scroll` action) over force-acting on invisible elements
   - If an element is not visible, use `scroll` in the likely direction (e.g., 'down' to find elements below the current view).
   - Try opposite scroll direction if initial fails (up/down, left/right)
   - If the `open_app` action fails to correctly open the app, find the corresponding app in the app drawer and open it.

3. Information Tasks:
   - MANDATORY: Use answer action for questions
   - Verify data freshness (e.g., check calendar date)

## Response Format
STRICTLY follow:
Reasoning: [Step-by-step analysis covering:
           - Visibility verification
           - History effectiveness evaluation
           - Alternative approach comparison
           - Consideration of scrolling if needed]
Action: [SINGLE JSON action from catalog]

Generate response:
"""


# INPUTS: goal, before_ui_elements, after_ui_elements, action, reasoning
PROMPT_TEMPLATE_EXPLORE_SUMMARY="""
Goal: {goal}

Before screenshot elements:
{before_ui_elements}

After screenshot elements:
{after_ui_elements}

Action: {action}
Reasoning: {reasoning}

Provide a concise single-line summary (under 50 words) of this step by comparing screenshots and action outcome. Include:
- What was intended
- Whether it succeeded
- Key information for future actions
- Critical analysis if action/reasoning was flawed
- Important data to remember across apps

For actions like 'answer' or 'wait' with no screen change, assume they worked as intended.

Summary:
"""


PROMPT_TEMPLATE_FILTER = """
You are an expert in evaluating Android GUI agent task trajectories. Your task is to assess the quality and effectiveness of task trajectories for GUI manipulation tasks. 

A trajectory consists of the following components:
1. High-level Instruction: Describes the user's intended task (e.g., "Create a new travel itinerary document in a folder").
2. Action History: Includes two key parts:
   - Low-level Actions & Summaries: A sequence of actions, where each step includes:
     - The executed action.
     - A summary of the action, indicating the effect after the action is executed.
   - GUI Screenshots: Screenshots captured when the last three actions are executed: the third-to-last, second-to-last, and final actions (if there are at least three actions; otherwise, include all actions).

When evaluating a trajectory, consider these key aspects:

### Evaluation Criteria:
1. Trajectory Coherence:
   - Do the low-level steps and corresponding actions follow a logical sequence toward the goal?
   - Are the actions clearly described and specific?
   - Are there redundant or unnecessary actions?

2. Task Completion:
   - Does the trajectory successfully achieve the instructed task?
   - Are all necessary interactions completed?
   - Are error cases handled appropriately?

### Scoring Guidelines:
Rate the trajectory on a scale of 1 to 5 based on the evaluation criteria:

- 5: The task is perfectly completed, successfully executing multiple actions to achieve the goal. The sequence is logically clear with no noticeable redundancies.
- 4: The task is mostly completed, successfully executing multiple actions. However, due to challenges or ambiguities in the instructions, the completion is not perfect, or there are inefficiencies in the process.
- 3: The task is partially completed, with some successful actions executed. However, due to task or environmental constraints, the goal is not fully achieved, or the sequence ends in a loop or error.
- 2: Only a few actions are executed. Although there is an attempt to complete the task, the trajectory deviates from the goal early on or demonstrates significant inefficiencies in execution and logic.
- 1: The task fails completely, with no meaningful actions executed at the start. The sequence either falls into an immediate deadlock, a repetitive loop, or demonstrates no value in completing the task.

Note: If the task is relatively complex, but the trajectory demonstrates valuable attempts, even if the task is not fully completed, consider adjusting the score upward. However, if the task is complex but the trajectory fails to perform actions that contribute meaningfully to task completion, no extra points should be awarded.

### Response Format:
Format your response into two lines as shown below: 
Reason: <your thoughts and reasoning process for the score>
Score: <your score from 1-5>
"""


PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message),'
    ' like user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by'
    ' performing actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step.'
    ' At each step, you will be given the current screenshot (including the'
    ' original screenshot and the same screenshot with bounding'
    ' boxes and numeric indexes added to some UI elements) and a history of'
    ' what you have done (in text). Based on these pieces of information and'
    ' the goal, you must choose to perform one of the'
    ' action in the following list (action description followed by the JSON'
    ' format) by outputing the action in the correct JSON format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    "- If you think the task is not feasible (including cases like you don't"
    ' have enough information or can not perform some necessary actions),'
    ' finish by using the `status` action with infeasible as goal_status:'
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
    "- Answer user's question:"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on an element on the screen. We have added marks (bounding'
    ' boxes with numeric indexes on their TOP LEFT corner) to most of the UI'
    ' elements in the screenshot, use the numeric index to indicate which'
    ' element you want to click:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on an element on the screen, similar with the click action'
    ' above, use the numeric label on the bounding box to indicate which'
    ' element you want to long press:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into a text field (this action contains clicking the text'
    ' field, typing in the text and pressing the enter, so no need to click on'
    ' the target field to start), use the numeric label'
    ' on the bounding box to indicate the target text field:'
    ' `{{"action_type": "input_text", "text": <text_input>,'
    ' "index": <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    '- Open an app (nothing will happen if the app is not'
    ' installed): `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)


GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General:\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history),"
    ' SWITCH to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations:\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)


ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}\n\n'
    'Here is a history of what you have done so far:\n{history}\n\n'
    'The current screenshot and the same screenshot with bounding boxes'
    ' and labels added are also given to you.\n'
    'Here is a list of detailed'
    ' information for some of the UI elements (notice that some elements in'
    ' this list may not be visible in the current screen and so you can not'
    ' interact with it, can try to scroll the screen to reveal it first),'
    ' the numeric indexes are'
    ' consistent with the ones in the labeled screenshot:\n{ui_elements}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your Answer:\n'
)


SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is: {goal}\n'
    'Now I want you to summerize the latest step.\n'
    'You will be given the screenshot before you performed the action (which'
    ' has a text label "before" on the bottom right), the action you chose'
    ' (together with the reason) and the screenshot after the action was'
    ' performed (which has a text label "after" on the bottom right).\n'
    'Also here is the list of detailed information for some UI elements'
    ' in the before screenshot:\n{before_elements}\n'
    'Here is the list for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    'By comparing the two screenshots (plus the UI element lists) and the'
    ' action performed, give a brief summary of this step. This summary'
    ' will be added to action history and used in future action selection,'
    ' so try to include essential information you think that will be most'
    ' useful for future action selections like what you'
    ' intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might be'
    ' wrong), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short (better less than 50 words) and in a single line\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)


THOUGHT_GENERATION = """
You are an expert at reconstructing thought processes from UI tasks.

Given an instruction, a screenshot, UI element information, a ground-truth action, and a raw thought (which may contain index-based references to UI elements), rewrite the thought into a clear, single-sentence reasoning process.

Your rewritten thought must:

1. Convert all index-based UI references  
   - Replace any "index x" or "(index x)" with natural-language descriptions of the corresponding UI element, based on the screenshot and UI element info.  
   - If an index cannot be matched reliably, omit it and describe the plausible UI element in context.

2. Maintain a realistic speculative reasoning style  
   - Do not imply knowledge of the ground-truth action.
   - Use first-person and present a logical, step-by-step thought process.

3. Follow the essential reasoning structure  
   - Infer what the next reasonable action(i.e., the action_type of the ground-truth action) should be to proceed with the task.
   - Ensure that the rewritten thought and next action strictly follows the type of the ground-truth action.
   - If the ground truth action type is open_app, only output: To achieve [instruction], I need to open the [xxx] app.

4. Avoid hallucination  
   - Only describe elements that are visibly present or logically inferable.


Instruction:
{instruction}

Screenshot:
<|vision_start|><|image_pad|><|vision_end|>

UI Elements:
{ui_elements}

Raw Thought:
{raw_thought}

Ground Truth Action:
{action}

Examples:
1. To create a new contact, I need to first open the Contacts app with open_app action.
2. To create a new contact, I need to click on the \"Create contact\" button, which is represented by the plus icon on the screen.
3. I will start by entering the first name \"John\" into the appropriate field.
4. The email field is not visible on the current screen. I will scroll down to reveal it and enter the email address.
5. All the required contact details have been entered. Now, I will save the contact by clicking the \"Save\" button.

Output only:
<one-sentence rewritten thought>

"""
