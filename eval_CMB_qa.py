import os
import torch
import platform
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import re

def init_model():
    print("init model ...")
    model_path = './export'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype='auto'
    ).eval()
    return model, tokenizer

def main():
    model, tokenizer = init_model()
    
    with open('CMB-test-choice-question-merge.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    results = []
    for i in tqdm(range(len(data))):
        test_id = data[i]['id']
        exam_type = data[i]['exam_type']
        exam_class = data[i]['exam_class']
        question_type = data[i]['question_type']
        question = data[i]['question']
        option_str = 'A. ' + data[i]['option']['A'] + '\nB. ' + data[i]['option']['B']+ '\nC. ' + data[i]['option']['C']+ '\nD. ' + data[i]['option']['D']
        
        prompt = f'以下是中国{exam_type}中{exam_class}考试的一道{question_type}，不需要做任何分析和解释，直接输出答案选项。\n{question}\n{option_str}'
        messages = [{"role": "user", "content": prompt}]
        
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(response)
        matches = re.findall("[ABCDE]", response)
        final_result = "".join(matches)
        
        info = {
                "id": test_id,
                "model_answer": final_result
            }
        results.append(info)
        
        messages.clear()

    with open('output.json', 'w', encoding="utf-8") as f1:
        json.dump(results, f1, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()