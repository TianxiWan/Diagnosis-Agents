import re
import os
import pandas as pd
import json
from openai import OpenAI

TIMEOUT = 100

def validate_message_structure(messages):
    """Verify that the message list structure meets the requirements"""
    required_keys = ['role', 'content']
    for idx, msg in enumerate(messages):
        # Check required fields
        if not all(key in msg for key in required_keys):
            raise ValueError(f"消息 {idx} 缺少必需字段: {msg}")
        # Checking the content type
        if not isinstance(msg['content'], str):
            raise TypeError(f"消息 {idx} 的content必须是字符串类型, 实际类型: {type(msg['content'])}")
        # Check for empty content
        if len(msg['content'].strip()) == 0:
            raise ValueError(f"消息 {idx} 的content为空")

class DoctorCost:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.input_cost = 0.3 / 1000000
        self.output_cost = 0.15 / 1000000
        self.total_cost = 0

    def money_cost(self, prompt_token_num, generate_token_num):
        if self.model_name == 'gpt-4o-mini':
            self.total_cost += prompt_token_num * self.input_cost + generate_token_num * self.output_cost

    def get_cost(self):
        return self.total_cost
    
class PatientCost:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.input_cost = 0.3 / 1000000
        self.output_cost = 0.15 / 1000000
        self.total_cost = 0

    def money_cost(self, prompt_token_num, generate_token_num):
        if self.model_name == 'gpt-4o-mini':
            self.total_cost += prompt_token_num * self.input_cost + generate_token_num * self.output_cost
    
    def get_cost(self):
        return self.total_cost


def gpt4_client_init():
    openai_api_key = "" #Fill in your gpt API key
    api_base = "" #Fill in your gpt API base url

    client = OpenAI(api_key=openai_api_key, base_url=api_base)
    return client

def qwen_client_init():
    openai_api_key = "" #Fill in your qwen API key
    openai_api_base = "" #Fill in your qwen API base url

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def ds_client_init():
    client = OpenAI(
        api_key = os.environ.get("", ""),#Fill in your deepseek API key
        base_url = "", #Fill in your deepseek API base url
    )
    return client

def tool_client_init(model_name):
    if 'gpt' in model_name:
        client = gpt4_client_init()
    elif 'deepseek' in model_name:
        client = ds_client_init()
    else:
        client = qwen_client_init()
    return client

def doctor_client_init(model_name):
    if 'gpt' in model_name:
        client = gpt4_client_init()
    elif 'deepseek' in model_name:
        client = ds_client_init()
    else:
        client = qwen_client_init()
    return client

def patient_client_init(model_name):
    if 'gpt' in model_name:
        client = gpt4_client_init()
    elif 'deepseek' in model_name:
        client = ds_client_init()
    else:
        client = qwen_client_init()
    return client

def api_response_classification(model_name, input_sentence):   #Used to dichotomize patient responses
    messages = []
    messages = []
    client = tool_client_init(model_name)
    prompt = "你需要根据医生和患者的对话判断该患者是否有医生询问的情况发生。\n如果有，请返回“是”，如果没有，请返回“否”。只能有这两种回答，不用输出解释或思考过程\n\n医患对话如下：{}".format(input_sentence)
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，非常善于文本分类"},
                    {"role": "user", "content":prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        top_p=0.05,
        temperature=0.3
    )
    response = chat_response.choices[0].message.content
    
    response_clean = response.replace(" ", "").replace("\n", "").strip()

    if "是" in response_clean:
        return True
    elif "否" in response_clean:
        return False
    else:
      
        raise ValueError(f"无法判断模型回答：{response}")
    
def api_topic_choice(model_name, input_sentence):  #Determines the execution order of the four disorder-specific sub-state machines
    messages = []
    messages = []
    client = tool_client_init(model_name)
    prompt = "你需要根据医生和患者的对话判断该患者患有“抑郁”，“焦虑”，“双相”，“多动”四种疾病的可能性按从大到小的顺序排序，以列表形式输出。\n输出格式：['xx','xx','xx','xx']\n\n医患对话如下：{}".format(input_sentence)
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，非常善于文本分类"},
                    {"role": "user", "content":prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        top_p=0.05,
        temperature=0.3
    )
    response = chat_response.choices[0].message.content
    response_clean = response.replace(" ", "").replace("\n", "").strip()


    match = re.search(r"\[([^\]]+)\]", response_clean)

    if match:
        items = match.group(1).split("','")  
        items = [item.replace("'", "").replace("‘", "").replace("’", "") for item in items]
        return items
    else:
        print("无法识别模型输出的列表")
        return None

    
def api_if_parse(model_name, input_sentence): #Determine whether it is necessary to ask the patient about his or her experience in depth
    messages = []
    client = tool_client_init(model_name)
    prompt = "你需要根据医生和患者的对话判断该患者的回答中是否很可能隐含ta的一些经历，即医生是否应该根据ta的回答继续追问相关经历，请返回“是”或“否”，只能有这两种回答，不用输出解释或思考过程。医生和患者的对话如下：{}".format(input_sentence)
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，非常善于文本解读"},
                    {"role": "user", "content":prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        top_p=0.05,
        temperature=0.2
    )
    response = chat_response.choices[0].message.content
    response_clean = response.replace(" ", "").replace("\n", "").strip()

    if "是" in response_clean:
        return True
    elif "否" in response_clean:
        return False
    else:
        raise ValueError(f"无法判断模型回答：{response}")

def api_load_for_background_gen(model_name, input_sentence):  #generate Fictitious experience of the patient
    messages = []
    client = tool_client_init(model_name)
    prompt = "输入文本是关于精神疾病患者的基本状况和过去经历的关键词，发挥想象力，根据这些信息以第一人称编写一个故事，完整讲述患者过去的经历，这段经历是患者出现精神疾病的主要原因。\n要求1.输出一整段故事，扩充事件的起因、经过、结果，不要使用比喻句，不要使用浮夸的表述。2.不要输出虚拟的患者姓名。3.不允许输出类似“我正在努力走出阴影”，“在医生的指导下”，只需要输出虚构的故事。\n ###输入文本如下：{}".format(input_sentence)
    messages.extend([{"role": "system", "content": "你是一个功能强大，想象力丰富的文本助手，非常善于写故事"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        top_p=0.9,
        timeout=TIMEOUT
    )
    response = chat_response.choices[0].message.content
    return response



def load_background_story(path):
    with open(path, 'r', encoding='utf-8') as f:
        story = f.readlines()
    return story

    

