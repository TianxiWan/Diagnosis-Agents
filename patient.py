import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import llm_tools_api
import re

class Patient(llm_tools_api.PatientCost):
    PATIENT_PROMPT = """
    回答对话历史中医生的最近一个问题，禁止与对话历史出现重复回复
    【回答风格要求】
    1. 使用真实生活场景中的自然对话方式，禁止书面话表达：
    - 适当使用不完整句："有时候会心慌"
    - 可加入合理停顿："工作的话，大概每天加班"
    2. 禁止多次使用"..."、"嗯"、"呃"等词
    3.使用第一人称回答，非必要情况不生成疑问句，不总以“医生”开头。
    4.对话历史中的内容不要重复提起。
    5.回复内容必须根据病例内容和对话历史。
    6.对于病例中没有记录的症状坚决否认！！

    【内容规范】
    7. 根据病史生成具体细节：
    - 疼痛描述："像有根筋扯着疼"
    - 时间表述："这两周特别明显"
    - 程度量化："十次里有三四次失眠"
    8. 注意回答字数不能超过50字
    9.禁止任何括号描述动作/表情
    10.输出纯文本，无换行,标点符号只能使用逗号和句号
    不要输出思考过程!!
    """

    PATIENT_PROMPT_EXPERIENCE = """
    回答对话历史中医生的最近一个问题，禁止与对话历史出现重复回复
    【回答规范】
    1. 回复问题必须根据病例内容和创伤经历，对于病例和创伤经历中没有记录的默认没有（尤其时长不足的情况），对于症状是否存在和时长是否满足要分开回复！！
    2. 尽量避免直接心理描述，通过行为表现，例如：
    错误："我感到非常恐惧"
    正确："当时就蹲在墙角不敢动"
    3. 禁止多次使用"..."、"嗯"、"呃"等填充词
    4.使用第一人称回答，非必要情况不生成疑问句，不总以“医生”开头。
    5.对话历史中的内容不要重复提起。
    6.使用真实生活场景中的自然对话方式，禁止书面话表达。
    
    【语言控制】
    7. 包含1-2个生活化细节。
    8. 不使用文学化修辞。
    9. 控制回答长度，不能回答超60字。
    10.禁止任何括号描述动作/表情
    11.输出纯文本，无换行，标点符号只能使用逗号和句号
    12.问到创伤经历中存在的症状/经历可适当展开描述
    不要输出思考过程!!
    """

    def __init__(self, patient_template, model_path, use_api, story_path, disease_symptom_map_path) -> None:
        super().__init__(model_path.split('/')[-1])
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.patient_model = None
        self.patient_tokenizer = None
        self.experience = None
        self.patient_template = patient_template
        self.system_prompt = "你是一名{}岁的{}性{}患者，正在和一位精神科医生交流，使用简洁且口语化的表达，要求无空行，回复尽量简短并适当表现出犹豫。".format(self.patient_template['年龄'], self.patient_template['性别'], self.patient_template['诊断结果'])
        self.messages = []
        self.use_api = use_api
        self.client = None
        self.story_path = story_path
        self.dialbegin = True
        self.target_disease = re.split(r"[，,]\s*", patient_template['诊断结果'])
        self.all_diseases = ['抑郁症', '焦虑症', '双相情感障碍', '多动症']
        self.disease_symptom_map = {}
        self.disease_symptom_map_path = disease_symptom_map_path
        


    def patientbot_init(self):
        if self.use_api:
            self.client = llm_tools_api.patient_client_init(self.model_name)
        else:
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.patient_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.messages.append({"role": "system", "content": self.system_prompt})

    def find_unique_symptoms(self): #Find symptoms that belong only to include_diseases and not to exclude_diseases
        with open(self.disease_symptom_map_path, 'r', encoding='utf-8') as f:
            self.disease_symptom_map= json.load(f)
        disease_symptom_map = self.disease_symptom_map
        exclude_diseases = self.target_disease
        include_diseases = list(set(self.all_diseases) - set(exclude_diseases))
        
        include_symptom_sets = [set(disease_symptom_map[d]) for d in include_diseases if d in disease_symptom_map]
        if not include_symptom_sets:
            return set()
        common_include_symptoms = set.union(*include_symptom_sets)

        exclude_symptom_sets = [set(disease_symptom_map[d]) for d in exclude_diseases if d in disease_symptom_map]
        all_excluded_symptoms = set.union(*exclude_symptom_sets) if exclude_symptom_sets else set()
        return common_include_symptoms - all_excluded_symptoms


    def patient_response_gen(self, current_topic, dialogue_history):
        excluded_symptoms = self.find_unique_symptoms()
        if self.use_api:
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            patient_template = {key:val for key, val in self.patient_template.items() if key != '处理意见'} 
            self.experience = llm_tools_api.load_background_story(self.story_path)[0] #Set the patient to answer the experience whenever the doctor asks about it. Can be deleted
            if self.experience is None:               
                patient_prompt = "你是一名{}患者，正在和一位精神卫生中心临床心理科医生进行交流。你的回复要尽量简短精确。".format(self.patient_template['诊断结果'])+ self.PATIENT_PROMPT+"\n你的病例为“{}”，\n你和医生的对话历史为{}，你当前的回复需要围绕话题“{}”展开。注意：你没有除了{}之外的心理疾病的症状，被问到是否存在以下症状时你倾向于否定回答：{}。\n".format(patient_template, dialogue_history[-3:], current_topic, self.patient_template['诊断结果'], excluded_symptoms)
                self.messages.append({"role": "user", "content": patient_prompt})
                chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,
                        top_p=0.85,
                        frequency_penalty=0.8
                    )
                super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                patient_response = chat_response.choices[0].message.content
                self.messages.pop()
            else:
                patient_prompt = "你是一名{}患者，正在和一位心理科医生进行交流。 \
                    \n\n现在请根据下面要求生成对医生的回答:\n"+self.PATIENT_PROMPT_EXPERIENCE+"1.回复内容必须根据：\n  （1）病例：“{}“\n  （2）过去的创伤经历：“{}”\n  （3）对话历史：“{}”。你当前的回复需要围绕话题“{}”展开。注意：你没有除了{}之外的心理疾病的症状，被问到是否存在以下症状时你倾向于否定回答：{}。\n".format(self.patient_template['诊断结果'], patient_template, self.experience, dialogue_history[-3:], current_topic, self.patient_template['诊断结果'], excluded_symptoms)

                self.messages.append({"role": "user", "content": patient_prompt})
                chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,
                        top_p=0.85,
                        frequency_penalty=0.7
                    )
                super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                patient_response = chat_response.choices[0].message.content
                self.messages.pop()
        else:
            #TODO
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            text = self.patient_tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            patient_model_inputs = self.patient_tokenizer([text], return_tensors="pt").to(self.patient_model.device)
            generated_ids = self.patient_model.generate(
                patient_model_inputs.input_ids,
                max_new_tokens=2048
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(patient_model_inputs.input_ids, generated_ids)
            ]
            patient_response = self.patient_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.messages.append({"role": "assistant", "content": patient_response})
        
        return patient_response, super().get_cost()
    