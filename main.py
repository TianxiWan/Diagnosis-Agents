from doctor import Doctor
from patient import Patient
from DiagStateMachine import HierarchicalStateMachine
import json
import os
from tqdm import tqdm
import random
import re
import glob
import shutil
import os
import llm_tools_api


DOCTOR_PROMPT_PATH = './prompts/doctor/doctor_persona.json'
PATIENT_INFO_PATH = './raw_data/cases_ready.json'
MODEL_NAME = 'gpt-4o-mini'    
NUM = 1 #generate 1 conversation for each patient(1 FicExp corresponds to 1 conversation)
OUTPUT_DATA_PATH = './Dial_data'
OUTPUT_PASTEXP_PATH = './prompts/patient/background_story'
DIAGNOSIS_LIST_PATH = './prompts/diagstatemachine/diagnosis_list.json'
TOPIC_ORDER_PATH = './prompts/diagstatemachine/topic_order_dict.json'
MACHINE_PATH = "prompts/diagstatemachine"
DIAGNOSIS_DICT_PATH = './prompts/diagstatemachine/diagnosis_dict.json'
DISEASE_SYMPTOM_MAP_PATH = './prompts/diagstatemachine/disease_symptom_map.json'

total_cost = 0

with open(PATIENT_INFO_PATH, 'r', encoding='utf-8') as f:
    patient_info = json.load(f)

with open(DIAGNOSIS_LIST_PATH, 'r', encoding='utf-8') as f:
    diag_list = json.load(f)

with open(TOPIC_ORDER_PATH, 'r', encoding='utf-8') as f:
    topic_order_dict = json.load(f)

with open(DIAGNOSIS_DICT_PATH, 'r', encoding='utf-8') as f:
    diag_dict = json.load(f)

original_order = ['抑郁', '焦虑', '双相', '多动']
order_list = [random.sample(original_order, len(original_order))]


for patient_template in tqdm(patient_info):
    total_output_list = []
    number_before_com = patient_template['患者'].split("com")[0]
    com_and_after = "com" + patient_template['患者'].split("com", 1)[1]
    patient_dir = os.path.join(OUTPUT_PASTEXP_PATH, f'patient_{number_before_com}')
    original_diagnosis = patient_template["诊断结果"]


    if not os.path.isdir(patient_dir):
        raise FileNotFoundError(f"患者目录 {patient_dir} 不存在")


    story_paths = sorted(
        glob.glob(os.path.join(patient_dir, '*.txt')),
        key=lambda x: (
            int(re.search(r'com(\d+)', x).group(1)),  
            int(os.path.basename(x).split('_')[2].split('.')[0])  
        )
    )
    
    for i in range(NUM):
        dialogue_history = []
        output_list = []
        output_dict = {}
        story_path = story_paths[i]
        doc = Doctor(patient_template, DOCTOR_PROMPT_PATH,  MODEL_NAME, MACHINE_PATH, True)
        pat = Patient(patient_template, MODEL_NAME, True, story_path, DISEASE_SYMPTOM_MAP_PATH)
        doctor_response, current_topic, doctor_cost = doc.doctor_response_gen(None)
        output_dict['doctor'] = doctor_response
        dialogue_history.append('医生：' + doctor_response)
        print("医生：", doctor_response)
        current_topic = '患者的近况'
        patient_response, patient_cost = pat.patient_response_gen(current_topic, dialogue_history)
        output_dict['patient'] = patient_response
        dialogue_history.append('患者：' + patient_response)
        output_list.append(output_dict)
        output_dict = {}
        print("患者：", patient_response)
        if i == 0:
            order = llm_tools_api.api_topic_choice(MODEL_NAME, dialogue_history)
        else:
            order = order_list[i-1]
        state_transition_process = []
        for m in range(4):
            current_topic = order[m]
            states = topic_order_dict[current_topic]
            group = states[0]
            state = states[1]
            machine = HierarchicalStateMachine(states[0], states[1], MACHINE_PATH)
            current_topic=doc.get_question_text(group, state, "time0",machine.current_subgroup)
            doctor_response, current_topic, doctor_cost = doc.doctor_response_gen(dialogue_history,topic_seq=current_topic)
            output_dict['doctor'] = doctor_response
            dialogue_history.append('医生：' + doctor_response)
            print("医生：", doctor_response)
            patient_response, patient_cost = pat.patient_response_gen(current_topic, dialogue_history)
            output_dict['patient'] = patient_response
            dialogue_history.append('患者：' + patient_response)
            output_list.append(output_dict)
            print("患者：", patient_response)
            output_dict = {}
            parse_number = 0
            while machine.current_state not in diag_list:
                parse = llm_tools_api.api_if_parse(MODEL_NAME, dialogue_history[-2:])
                
                if parse and parse_number<4:
                    Current_topic_doctor = "请根据患者的回答深入询问经历"
                    doctor_response, current_topic_doctor, doctor_cost = doc.doctor_response_gen(dialogue_history, topic_seq=Current_topic_doctor)
                    output_dict['doctor'] = doctor_response
                    dialogue_history.append('医生：' + doctor_response)
                    print("医生：", doctor_response)
                    Current_topic_patient = "请根据医生的提问回答,多给相应的经历描述"
                    patient_response, patient_cost = pat.patient_response_gen(Current_topic_patient, dialogue_history)
                    output_dict['patient'] = patient_response
                    dialogue_history.append('患者：' + patient_response)
                    output_list.append(output_dict)
                    print("患者：", patient_response)
                    output_dict = {}
                    parse_number+=1
                else:
                    transfer = llm_tools_api.api_response_classification(MODEL_NAME, dialogue_history[-2:])
                    machine.get_next_state(transfer)
                    if machine.current_state not in diag_list:
                        Current_topic = doc.get_question_text(machine.current_group, machine.current_state, machine.current_time, machine.current_subgroup)
                        doctor_response, current_topic, doctor_cost = doc.doctor_response_gen(dialogue_history, topic_seq=Current_topic)
                        output_dict['doctor'] = doctor_response
                        dialogue_history.append('医生：' + doctor_response)
                        print("医生：", doctor_response)
                        current_topic_pat="根据医生的问题回答"
                        patient_response, patient_cost = pat.patient_response_gen(current_topic_pat, dialogue_history)
                        output_dict['patient'] = patient_response
                        dialogue_history.append('患者：' + patient_response)
                        output_list.append(output_dict)
                        print("患者：", patient_response)
                        output_dict = {}
                    elif machine.current_state in diag_list:
                        disease = machine.current_state
                        if "depression" in disease:
                            doc.diagnosis[0]=disease
                        elif "bipolar" in disease:
                            doc.diagnosis[1]=disease
                        elif "anxiety" in disease:
                            doc.diagnosis[2]=disease
                        elif "adhd" in disease:
                            doc.diagnosis[3]=disease
            state_transition_process.append(machine.state_history)
        if all(doc.diagnosis):
            diagnosis_result_list = doc.diagnosis
        else:
            print("还有疾病未诊断")
            print(doc.diagnosis)
            break

        if diagnosis_result_list[1] not in ["bipolar2", "bipolar6", "bipolar8"]:
            diagnosis_result_list[1] = "bipolar9"
        elif diagnosis_result_list[1] != "bipolar6" and diagnosis_result_list[0] != "depression4":
            diagnosis_result_list[1] = "bipolar10"
        else:
            diagnosis_result_list[1] = "bipolar6"
        final_diagnosis = "、".join([diag_dict[disease] for disease in diagnosis_result_list if diag_dict[disease] != ""])
        #Diagnostic Context Tree
        if patient_template['性别'] == "女":
            topiclist = ["有无家人患精神疾病或遗传病", "是否抽烟喝酒",  "是否有喝咖啡习惯", "有无不良嗜好", "月经情况（是否规律/痛经）", "是否有运动习惯"]
        else:
            topiclist = ["有无家人患精神疾病或遗传病", "是否抽烟喝酒",  "是否有喝咖啡习惯", "有无不良嗜好", "是否有运动习惯"]
        random.shuffle(topiclist)
        for topic in topiclist:
            print("当前话题：", topic)
            doctor_response, current_topic, doctor_cost = doc.doctor_response_gen(dialogue_history[-2:], topic_seq=topic)
            output_dict['doctor'] = doctor_response
            dialogue_history.append('医生：' + doctor_response)
            print("医生：", doctor_response)
            topic_patient = "请根据医生的提问回答,多给相应的经历描述"
            patient_response, patient_cost = pat.patient_response_gen(topic_patient, dialogue_history)
            output_dict['patient'] = patient_response
            dialogue_history.append('患者：' + patient_response)
            output_list.append(output_dict)
            print("患者：", patient_response)
            output_dict = {}
        doctor_response, current_topic, doctor_cost = doc.doctor_response_gen(final_diagnosis, is_dialogue_end=True)
        output_dict['doctor'] = doctor_response
        dialogue_history.append('医生：' + doctor_response)
        output_list.append(output_dict)
        print("医生：", doctor_response)
        total_output_list.append({"doctor":i, "topic_order":order, "original_diagnosis":original_diagnosis, "diagnosis_result":final_diagnosis,"experience_combination":com_and_after , "conversation":output_list, "state transition process":state_transition_process })
        total_cost += doctor_cost+patient_cost    
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)        
    with open(os.path.join(OUTPUT_DATA_PATH, 'patient_{}.json'.format(patient_template['患者'])), 'w', encoding='utf-8') as f:
        json_data = json.dump(total_output_list, f, indent=2, ensure_ascii=False)

print("********总价格*********:", total_cost)
source_dirs = [
    r"raw_data",
    r"./Dial_data",
    r"prompts/patient/background_story"
]
target_base = r"already_runned/1th_run"

os.makedirs(target_base, exist_ok=True)

for src in source_dirs:
    dest = os.path.join(target_base, os.path.basename(src))
    if os.path.exists(dest):
        shutil.rmtree(dest)  
    shutil.copytree(src, dest)
    print(f"已复制 {src} 到 {dest}")

print("所有文件夹复制完成。")