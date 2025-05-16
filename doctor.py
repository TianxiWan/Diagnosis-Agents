import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import llm_tools_api
import os


class Doctor(llm_tools_api.DoctorCost):
    DOCTOR_PROMPT_EMPATHY = """
    【语言风格要求】
    1. 使用自然的口语化中文，避免任何书面化表达
    2. 问题要结合上一轮患者的回答
    3. 禁止使用以下模式化开头：
    "从您的描述中..."，"从刚才的描述中..."，"根据你提到的..."，"谢谢"，"你的回答很有帮助"，"听到你的描述我很"，"听起来你..."
    4. 适当表达共情，且共情表达要自然，例如：
    - "这一定很不容易"
    - "我理解这种感受"
    - "我能理解你为什么会这样想"
    但禁止和历史对话中的模式重复！
    
    【输出要求】 
    5.禁止使用括号描述动作/表情
    6.禁止与历史对话使用相同开头
    7.输出纯文本，无换行,注意正确使用【标点符号】！！
    8.控制句子长度，输出最好不超60字
    9.对时间的表述尽量丰富且准确，例如（两周，半个月，两个星期）（六个月，半年）（这段时间，这几周，这几个月，那个时候）
    10.根据历史对话，禁止同一个询问模式连续多次使用，例如：“会不会”，“是不是”和“会......吗”等要交叉使用
    11.根据历史对话，禁止“那你...”“那这种情况...”等模式连续多次使用，建议把“这种情况”/“那”替换成患者提到的症状
    """

    DOCTOR_PROMPT = """
    【核心指令】 
    0. 问题最好结合上一轮患者的回答承上启下
    1. 使用简洁的口语提问，避免任何书面化表达
    2. 控制句子长度，输出最好不超50字
    3. 禁止使用以下模式化开头：
    "从您的描述中..."，"从刚才的描述中..."，"根据你提到的..."，"谢谢"，"你的回答很有帮助"，"听到你的描述我很"，"听起来你..."

    【特殊规范】
    4.允许使用少量引导词（每5次对话最多1次） 
    5.禁止任何括号描述动作/表情
    6.禁止与历史对话使用相同开头
    7.输出纯文本，无换行,注意正确使用【标点符号】！！
    8.对时间的表述尽量丰富且准确，例如（两周，半个月，两个星期）（六个月，半年）（这段时间，这几周，这几个月，那个时候）
    9.根据历史对话，禁止同一个询问模式连续多次使用，例如：“会不会”，“是不是”和“会......吗”等要交叉使用
    10.根据历史对话，禁止“那你...”“那这种情况...”等模式连续多次使用，建议把“这种情况”/“那”替换成患者提到的症状
    """

    def __init__(self, patient_template, doctor_prompt_path,  model_path, machine_path, use_api) -> None:
        super().__init__(model_path.split('/')[-1])
        self.patient_template = patient_template
        self.doctor_prompt_path = doctor_prompt_path
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.doctor_model = None
        self.doctor_tokenizer = None
        self.doctor_prompt = None
        self.client = None
        self.messages = []
        self.dialbegin = True
        self.use_api = use_api
        self.current_idx = 0
        self.doctor_persona = None
        self.patient_persona = None
        self.time_map = None
        self.state_contents = {}
        self.diagnosis_lists = None
        self.diagnosis = ['' for i in range(4)]
        self.machine_path = machine_path

    def _load_rules(self, folder: str):
        state_files = [
            "bipolar.json",
            "anxiety.json",
            "adhd.json",
            "depression.json"
        ]
        
   
        for filename in state_files:
            file_path = os.path.join(folder, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"关键状态文件缺失: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    self.state_contents.update(json.load(f))
                except json.JSONDecodeError:
                    raise ValueError(f"文件格式错误: {filename} 不是有效的JSON")

        time_path = os.path.join(folder, "TIME.json")
        if not os.path.exists(time_path):
            raise FileNotFoundError(f"时间映射文件缺失: {time_path}")
        
        with open(time_path, 'r', encoding='utf-8') as f:
            self.time_map = json.load(f)

    def get_question_text(self, group: str, state: str, time: str, subgroup) -> str: # Question text combining time stamp and states
        
        time_text = self.time_map.get(time, "")
        try:
            if subgroup != None:
                state_text = self.state_contents[group][subgroup][state]
            else:
                state_text = self.state_contents[group][state]
        except KeyError:
            state_text = f"[未定义的问题: {group}.{state}]"
        return f"{time_text}{state_text}" if time_text else state_text


    def doctorbot_init(self):
        with open(self.doctor_prompt_path, 'r', encoding='utf-8') as f:
            prompt = json.load(f)
        self._load_rules(self.machine_path)
        doctor_num = random.randint(0, len(prompt)-1)
        self.doctor_prompt = prompt[doctor_num]
        self.doctor_persona = "你是一名{}的{}专业的精神卫生中心临床心理科主任医师，对一名患者进行问诊。注意，你有如下的问诊习惯，你在所有的对话过程中都要记住和保持这些问诊习惯：\
            你尤其擅长诊断{}，你的问诊速度是{}的，你的交流风格是{}的，你{}在适当到时候与患者进行共情对话，你{}向患者解释一些专业名词术语。使用口语化的表达。注意每轮不能有超过两个问题。" \
            .format(self.doctor_prompt['age'], self.doctor_prompt['gender'], self.doctor_prompt['special'], self.doctor_prompt['speed'], self.doctor_prompt['commu'], self.doctor_prompt['empathy'], self.doctor_prompt['explain'])

        self.patient_persona = "患者是一名{}岁的{}性。".format(self.patient_template['年龄'], self.patient_template['性别'])
        final_prompt = self.doctor_persona + self.patient_persona + "现在你与患者的对话开始，通常一开始你会询问患者【为什么来看医生/想解决什么问题/最近情况如何】等问题。使用口语化表达与患者交流，不要输出类似”好的，我会按照您的要求开始问诊“的话。"

        if self.use_api:
            self.client = llm_tools_api.doctor_client_init(self.model_name)
        else:
            self.doctor_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.doctor_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.messages.extend([{"role": "system", "content": self.doctor_persona},
                            {"role": "user", "content": final_prompt}])

        

    def doctor_response_gen(self, dialogue_history, topic_seq=None, is_dialogue_end=False):
        if self.use_api:
            if self.dialbegin == True:
                self.doctorbot_init()
                print("问诊开始---\n")
                self.current_idx += 1
                chat_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    top_p = 0.93
                )
                super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                doctor_response = chat_response.choices[0].message.content
                self.messages.pop()
                self.dialbegin = False
                return doctor_response, None, super().get_cost()
            else:   
                if is_dialogue_end:
                    diag_result = "诊断结束，你的诊断结果为：{}。".format(dialogue_history)
                    return diag_result, None, super().get_cost()
                else:
                    self.current_idx += 1
                    print("**********current_topic ", topic_seq)
                    if self.doctor_prompt['empathy'] == '有':
                        doctor_prompt = self.doctor_persona + self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history[-6:]) + self.DOCTOR_PROMPT_EMPATHY + "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前话题{}。注意输出的问题要符合当前话题并结合上一轮患者的回答，换成口语化的表述方式加上共情策略，不能与对话历史的语言结构重复！！如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。".format(topic_seq)
                            
                    else:
                        doctor_prompt = self.doctor_persona + self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history[-6:]) + self.DOCTOR_PROMPT + "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前话题{}。注意输出的问题要符合当前话题并结合上一轮患者的回答，换成口语化的表述方式，不能与对话历史的语言结构重复！！如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。".format(topic_seq)                             
                    self.messages.append({"role": "user", "content": doctor_prompt})
                    chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,
                        top_p=0.95,
                        temperature = 1,
                        frequency_penalty=0.9
                    )
                    super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                    doctor_response = chat_response.choices[0].message.content
                    self.messages.pop()
                    return doctor_response, None, super().get_cost()
                
        else:
            #Todo
            if self.dialbegin == True:
                self.doctorbot_init()
                text = self.doctor_tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                doctor_model_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self.doctor_model.device)
                generated_ids = self.doctor_model.generate(
                    doctor_model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(doctor_model_inputs.input_ids, generated_ids)
                ]
                doctor_response = self.doctor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.messages.append({"role": "assistant", "content": doctor_response})
                self.dialbegin = False
            else:
                text = self.doctor_tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                doctor_model_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self.doctor_model.device)
                generated_ids = self.doctor_model.generate(
                    doctor_model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(doctor_model_inputs.input_ids, generated_ids)
                ]
                doctor_response = self.doctor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.messages.append({"role": "assistant", "content": doctor_response})
            return doctor_response
        
