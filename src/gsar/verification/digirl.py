"""
Code based on https://github.com/DigiRL-agent/digirl/blob/master/digirl/environment/android/evaluate.py
"""
from PIL import Image
from typing import Tuple
import numpy as np
import re
import os
from tenacity import retry, stop_after_attempt, wait_fixed, wait_chain
from openai import OpenAI
from ..common.model_client import OPENAI_CONFIG, create_openai_client
from .utils import encode_image_to_data_url


def extract_status(text):
    match = re.search(r'Status:\s*(\w+)', text)
    if match:
        return match.group(1)
    else:
        return None


def build_prompt_webshop(intent) -> Tuple[str, str]:
    system_msg = """You're an expert in evaluating whether the Screenshot successfully completes the Task."""
    prompt = [f"""Task: Go to bestbuy.com
Q: What should I expect to see on the screenshot if I've gone to bestbuy.com?
A: I should expect to see I'm in the Best Buy website, which usually shows the best buy logo with some featured products and categories. The screenshot shows I'm searching for "bestbuy.com" in the Google search (with some search suggestions) instead of being in the Best Buy website.
Status: failure""", #0
f"""Task: Go to costco.com
Q: What should I expect to see on the screenshot if I've gone to costco.com?
A: I should expect to see I'm in the Costco website, which usually shows the homepage with some featured products and categories. The screenshot shows I'm in the Costco website with some featured products and categories.
Status: success""", #1
f"""Task: Go to bestbuy.com, search for "macbook"
Q: What should I expect to see on the screenshot if I've gone to bestbuy.com AND searched for "macbook"?
A: I should expect to see I'm in the Best Buy website and search results for "macbook". The screenshot shows I'm in the Best Buy website and have several search suggestions for "macbook", but it does not show search results of the product, which usually includes price and the product details.
Status: failure""", #2
f"""Task: Go to ebay.com, search for "corsair k70"
Q: What should I expect to see on the screenshot if I've gone to ebay.com AND searched for "corsair k70"?
A: I should expect to see I'm in the eBay website and search results for "corsair k70". The screenshot shows I'm in the eBay ebay website with some search suggestions for "corsair k70", but it does not show search results of the product, which usually includes price and the product details.
Status: failure""", #3
f"""Task: Go to walmart.com, search for "macbook air"
Q: What should I expect to see on the screenshot if I've gone to walmart.com AND searched for "macbook air"?
A: I should expect to see I'm in the Walmart website and search results for "razer huntsman". The screenshot shows I'm in Google search with some search suggestions for "macbook air", not Walmart.
Status: failure""", #4
f"""Task: Go to walmart.com, search for "razer huntsman"
Q: What should I expect to see on the screenshot if I've gone to walmart.com AND searched for "razer huntsman"?
A: I should expect to see I'm in the Walmart website and search results for "razer huntsman". The screenshot shows I'm in the Walmart website, but there's no search results for "razer huntsman", which usually includes the product details and price.
Status: failure""", #5
f"""Task: Go to ebay.com, search for "lenovo thinkpad"
Q: What should I expect to see on the screenshot if I've gone to ebay.com AND searched for "lenovo thinkpad"?
A: I should expect to see I'm in the eBay website and search results for "lenovo thinkpad". The screenshot shows I'm in the eBay website and have several search results for "lenovo thinkpad".
Status: success""", #6
f"""Task: Go to ebay.com, search for "razer thresher", select the first entry
Q: What should I expect to see on the screenshot if I've gone to ebay.com AND going to the first entry of the search results of "razer thresher"?
A: I should expect to see I'm in the eBay website and detailed information of a razer thresher product, like a big image of the product, the price, and the product details. The screenshot shows I'm in the eBay website but with more than one search results for "razer thresher", which means the user has not selected the first entry of the search results.
Status: failure""", #7
f"""Task: Go to target.com, search for "razer kraken", and select the first entry
Q: What should I expect to see on the screenshot if I've gone to target.com AND gone to the first entry of the search results of "razer kraken"?
A: I should expect to see I'm in the Target website and can see detailed information of a razer thresher product, like a big image of the product, the price, and the product details. The screenshot shows I'm in Google Search, not in the Target website.
Status: failure""", #8
f"""Task: Go to ebay.com, search for "acer predator", and select the first entry
Q: What should I expect to see on the screenshot if I've gone to ebay.com AND gone to the first entry of the search results of "acer predator"?
A: I should expect to see I'm in the eBay website with detailed information of an acer predator product, like a big image of the product, the price, and the product details. The screenshot shows I'm in the eBay website and have more than one search results for "acer predator", which means the user has not selected the first entry of the search results.
Status: failure""", #9
f"""Task: Go to bestbuy.com, search for "macbook", select the first entry
Q: What should I expect to see on the screenshot if I've gone to bestbuy.com AND gone to the first entry of the search results of "macbook"?
A: I should expect to see I'm in the eBay website and detailed information of a macbook product, like a big image of the product, the price, and the product details. The screenshot shows I'm in the eBay website and have detailed information of Macbook Air, including the price and the product details.
Status: success""", #10
f"""Task: {intent}
Respond in this format:
Q: What should I expect to see on the screenshot if I've <repeat the task>?
A: I should expect to see <first expectation, then what's in the given screenshot.>
Status: success or failure (don't return anything else)
Start with "Q:"."""]

    image_paths = os.path.join(os.path.dirname(__file__), "images", "digirl")
    cot_image_list = [os.path.join(image_paths, "step1_bestbuy.png"), # 0
                    os.path.join(image_paths, "step1_costco.png"), # 1
                    os.path.join(image_paths, "step2_bestbuy.png"), # 2
                    os.path.join(image_paths, "step2_ebay.png"), # 3
                    os.path.join(image_paths, "step2_walmart.png"), # 4
                    os.path.join(image_paths, "step2_walmart2.png"), # 5
                    os.path.join(image_paths, "step2_ebay2.png"), # 6
                    os.path.join(image_paths, "step3_ebay.png"), # 7
                    os.path.join(image_paths, "step3_target.png"), # 8
                    os.path.join(image_paths, "step3_ebay2.png"), # 9
                    os.path.join(image_paths, "step3_bestbuy.png"), # 10
                    "" # -1
                    ]    
    
    return system_msg, prompt, cot_image_list


def build_prompt_general(intent) -> Tuple[str, str]:
    system_msg = """You're an expert in evaluating whether the Screenshot successfully completes the Task."""
    prompt = [f"""Task: Open the settings.
Q: What should I expect to see on the screenshot if I've opened the settings?
A: I should expect to see I'm in the settings app. The screenshot shows the home screen of a mobile device, with various app icons displayed, including the settings app icon, but the settings app is not opened.
Status: failure""", #0
f"""Task: Find hotels in washington dc
Q: What should I expect to see on the screenshot if I've searched for hotels in Washington, DC?
A: I should expect to see I'm in a search results page for hotels in Washington, DC. The screenshot shows a Google search page with the search field populated with the query "hotels in washington dc" and a list of suggested searches related to hotels in Washington, DC, but it does not show any search results for hotels in Washington, DC.
Status: failure""", #1
f"""Task: What's a good restaurant in Portland?
Q: What should I expect to see on the screenshot if I've searched for a good restaurant in Portland?
A: I should expect to see I'm in a search results page for a good restaurant in Portland. The screenshot shows a Google search page with a search input field for "good restaurant in portland" and a map results preview showing business locations near Portland, like "Li Pigeon", "Portland City Grill", and "Higgins",
Status: success""", #2
f"""Task: What's on the menu at In-N-Out?
Q: What should I expect to see on the screenshot if I've searched for the menu at In-N-Out?
A: I should expect to see a menu page for In-N-Out, including product names, thumbnails and prices. The screenshot shows a Google search page with a search input field for "In-N-Out menu" and some page snippets of In-N-Out indicating potential menu items, but does not actually show the actual menu.
Status: failure""", #3
f"""Task: What's the news in Suriname?
Q: What should I expect to see on the screenshot if I've searched for the news in Suriname?
A: I should expect to see some news in Suriname, such as someone did something or some accident happens in Suriname. The screenshot shows a Google search page with a search input field for "Suriname news today" and some page snippets indicating potential news items, but does not actually show the news.
Status: failure""", #4
f"""Task: What's the weather like in Chicago?
Q: What should I expect to see on the screenshot if I've searched for the weather in Chicago?
A: I should expect to see some exact values like temperature, humidity, wind speed, and weather condition in Chicago. The screenshot shows a Google search page with a search input field for "weather in Chicago" and some page snippets indicating potential weather information. Although one page snippet contains some weather information, the information is not comprehensive enough to determine the weather in Chicago.
Status: failure""", #5
f"""Task: Set an alarm for 6pm.
Q: What should I expect to see on the screenshot if I've set an alarm for 6pm?
A: I should expect to see some alarms including a 6pm alarm activated in the clock app. The screenshot shows an attempt to set an alarm for 6pm in the clock app, but the alarm is not set yet.
Status: failure""", #6
f"""Task: What's the news in French today?
Q: What should I expect to see on the screenshot if I've searched for the news in French today?
A: I should expect to see some news in French today, such as someone did something or some accident happens in French today. The screenshot shows I'm in the website france24.com but blocked with a cookie consent banner.
Status: failure""", #7
f"""Task: What's the news in French today?
Q: What should I expect to see on the screenshot if I've searched for the news in French today?
A: I should expect to see some news in French today, such as someone did something or some accident happens in French today. The screenshot shows I'm in the website france24.com and can see the news, like something about the Olympic flame.
Status: success""", #8
f"""Task: {intent}
Respond in this format:
Q: What should I expect to see on the screenshot if I've <repeat the task>?
A: I should expect to see <first expectation, then what's in the given screenshot.>
Status: success or failure (don't return anything else)
Start with "Q:"."""]
    
    image_paths = os.path.join(os.path.dirname(__file__), "images", "digirl")
    cot_image_list = [
        os.path.join(image_paths, "screenshot_menu.png"), # 0
        os.path.join(image_paths, "screenshot_hotel.png"), # 1
        os.path.join(image_paths, "screenshot_restaurant.png"), # 2
        os.path.join(image_paths, "screenshot_foodmenu.png"), # 3
        os.path.join(image_paths, "screenshot_news.png"), # 4
        os.path.join(image_paths, "screenshot_weather.png"), # 5
        os.path.join(image_paths, "screenshot_alarm.png"), # 6
        os.path.join(image_paths, "screenshot_frenchnews_blocked.png"), # 7
        os.path.join(image_paths, "screenshot_frenchnews_okay.png"), # 8
        "" # -1
    ]
    
    return system_msg, prompt, cot_image_list

@retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] + [wait_fixed(3) for i in range(2)] + [wait_fixed(5)]),
         stop=stop_after_attempt(5))
def call_openai_compatible(client, model_name, system_msg, prompt, image_list, image_path):
    user_content = []

    user_content.append({
        "type": "text",
        "text": "=====Examples====="
    })

    # few-shot examples
    for i in range(len(prompt) - 1):
        user_content.extend([
            {"type": "text", "text": "\nScreenshot:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_to_data_url(image_list[i])
                }
            },
            {"type": "text", "text": prompt[i]}
        ])

    # current turn
    user_content.extend([
        {"type": "text", "text": "=====Your Turn====="},
        {"type": "text", "text": "\nScreenshot:"},
        {
            "type": "image_url",
            "image_url": {
                "url": encode_image_to_data_url(image_path)
            }
        },
        {"type": "text", "text": prompt[-1]}
    ])

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages = messages,
        temperature=0.5,
        top_p=0.5
    )

    return response.choices[0].message.content


class EndResultEvaluator:
    def __init__(self, task_set="general"):
        self.model_name = OPENAI_CONFIG["model_name"]
        self.client = create_openai_client()

        self.img_matrix = None
        self.cache_max = 5
        self.threshold = 0.001 * 255**2
        self.task_set = task_set

    def __call__(self, last_two_images, intent: str) -> bool:
        """
        last_two_images: a list of two image path. [last_image_path, this_image_path]
        intent: a string representing the user's intent

        Returns:
        - True if the task is completed
        - False otherwise

        If there's an error, it will return -1 and print the error message
        """
        with Image.open(last_two_images[0]) as img1_src, Image.open(last_two_images[1]) as img2_src:   
            img1 = np.array(img1_src)
            img2 = np.array(img2_src)
        if np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2) < self.threshold:
            print("skipping evaluation due to same images")
            return -1
        # this is an approximation, but it should be fine to add frequently viewed false negatives
        if self.img_matrix is None:
            self.img_matrix = np.expand_dims(img2, axis = 0)
        # will always trigger after the first time
        else:
            distances = np.mean((self.img_matrix.astype(np.float64) - img2.astype(np.float64))**2, axis = (1,2,3))
            if np.min(distances) < self.threshold:
                print("skipping evaluation due to previously seen image, current img_matrix size: ", self.img_matrix.shape[0])
                return -1
            elif self.img_matrix.shape[0] < self.cache_max:
                self.img_matrix = np.concatenate([self.img_matrix, np.expand_dims(img2, axis = 0)], axis = 0)
        
        # print(f"Task: {intent}, image: {last_two_images[1]}")
        eval_res = self._evaluate(intent, last_two_images[1])
            
        del img1, img2
        return eval_res

    def _evaluate(self, intent: str, image_path: str) -> bool:
        if self.task_set == "general":
            system_msg, prompt, cot_image_list = build_prompt_general(intent)
        elif self.task_set == "webshop":
            system_msg, prompt, cot_image_list = build_prompt_webshop(intent)
        
        max_retry = 3
        for attempt in range(1, max_retry + 1):
            response_text = call_openai_compatible(self.client, self.model_name, system_msg, prompt, cot_image_list, image_path)
            status = extract_status(response_text)
            status = (status or "").lower()

            if "success" in status:
                return 1
            if "failure" in status:
                return 0
            
            if attempt == 3:
                if "success" in response_text:
                    return 1
                if "failure" in response_text:
                    return 0

        print(f"解析失败, 返回-2")
        return -2 # 解析失败    
