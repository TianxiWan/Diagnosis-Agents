import os
import pandas as pd
import json
import llm_tools_api


PATIENT_CASES_ORIGIN_PATH = './raw_data/patient_info.xlsx'
ORIGIN_JSON_PATH = './raw_data/cases_completed.json'
CASES_JSON_PATH = './raw_data/cases_ready.json'
PROMPT_PATH = './prompts'
OUTPUT_PASTEXP_PATH = './prompts/patient/background_story'
MODELNAME = 'gpt-4o-mini'  
PATIENT_COUNT = 1 # Number of electronic medical records to be used
FicExp_COUNT = 1 # 1 EMR will be used to generate FicExp_COUNT fictitious experiences


class PatientCases():
    def __init__(self, prompt_path, origin_json_path, cases_json_path, use_api) -> None:
        self.origin_json_path = origin_json_path
        self.cases_json_path = cases_json_path
        self.use_api = use_api
        self.prompt_path = prompt_path    # root path of prompt
        self.gender_mode = None
        self.age_mode = None
    
    def patient_json2json(self, patient_count, conversation_count):
        with open(self.origin_json_path, 'r', encoding='utf-8') as f:
            origin_patient_data = json.load(f)
        patient_data = origin_patient_data[:patient_count]
        combine_list = []
        if conversation_count <=5:
            for i in range(conversation_count):
                combine_list.append([i+1, i+1])
        elif conversation_count > 5 and conversation_count <= 50:
            for i in range(5):
                for k in range(conversation_count):
                    combine_list.append([i+1, k%5+1])
        else:
            return("conversation_count should be less than 50")

        output_list = []
        for case in patient_data:
  
            for [i,k] in combine_list:
                output_dict = {}
 
                output_dict['患者'] = str(case['id']) + f'com{i}_{k}'
                output_dict['年龄'] = case['年龄']
                output_dict['性别'] = case['性别'] 
                output_dict['职业'] = case['职业']
                output_dict['婚姻状况'] = case['婚姻状况']
                output_dict['教育背景'] = case['教育背景']
                output_dict['诊断结果'] = case['初步诊断']
                output_dict['主诉'] = case['主诉']
                output_dict['病情状况'] = case['病情状况']
                output_dict['既往史'] = case['既往史']
                output_dict['家族史'] = case['家族史']
                output_dict['个人史'] = case['个人史'][str(i)]

                if self.use_api:                        

                    if output_dict['个人史'] == None:
                        print(f"警告1：病例 {output_dict['患者']} 的个人史内容为空，使用默认值")
                    else:
                        
                        print(f"个人史：{output_dict['个人史']}")
                story = self.gen_background_story(case, [i, k])
                if story != '故事生成失败':
                    output_path = os.path.join(
                        OUTPUT_PASTEXP_PATH,
                        f"patient_{case['id']}",
                        f'story_com{i}_{k}.txt'
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(story.replace("\n", ""))
                output_list.append(output_dict)
        with open (self.cases_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, indent=2, ensure_ascii=False)   

    def gen_background_story(self, patient, combine_list):
        [i,k] = combine_list
        with open(os.path.join(self.prompt_path, 'patient', 'patient_background.txt'), 'r', encoding='utf-8') as f:
            text_prompt = f.readlines()[0]
        text_prompt = text_prompt.format(age=patient['年龄'],gender=patient['性别'],diagnosis=patient['初步诊断'],illness=patient['病情状况'],work=patient['职业'],personal_history=patient['个人史'][str(i)], experience=patient['经历'][str(k)])
    
        response = llm_tools_api.api_load_for_background_gen(MODELNAME, text_prompt)
        if response is not None:
            return response
        else:
            print(f"病例 {patient['患者']} 的背景故事生成失败")
            return '故事生成失败'
                    



patient = PatientCases(PROMPT_PATH, ORIGIN_JSON_PATH, CASES_JSON_PATH, use_api=True)
patient.patient_json2json(PATIENT_COUNT, FicExp_COUNT)
