from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import random
import pandas as pd
import torch
import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from data_processing import load_and_process_data
import random

# 시드 고정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 고정
set_seed(42)

# Config 파일 로드
with open("config.json", "r") as f:
    config = json.load(f)

model_name = "Qwen/Qwen2.5-7B-Instruct"

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto",
    # trust_remote_code=True,
    device_map="auto",  # 양자화 지원 장치에 자동 매핑
    # quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 아래의 지문, 질문, 선택지를 바탕으로 수험생이 질문에 답하는 데 도움이 되는 사전지식을 250~350자 내외의 한국어로 제공하세요.
#     사전지식은 정답을 직접적으로 드러내지 말고, 학생들이 스스로 사고하여 답을 도출할 수 있도록 유도하세요.
def generate_additional_info_with_huggingface(paragraph, question, choices, tokenizer, model):
    prompt = f"""
    아래의 지문, 질문, 선택지를 바탕으로 수험생이 질문에 답하기에 지문에 부족한 사전지식을 ㄹ 한국어로 제공하세요.

    지문:
    {paragraph}
    
    질문:
    {question}

    선택지:
    {choices}
    
    사전지식:
    """
    
    messages = [
        {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # 필요에 따라 조정
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def enrich_test_data_with_huggingface(test_df, tokenizer, model):
    """
    테스트 데이터에 대해 Hugging Face 모델을 사용하여 'hint' 열을 추가합니다.
    """
    enriched_data = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing hint"):
        paragraph = row["paragraph"]
        question = row["question"]
        choices = row["choices"]
        
        # Hugging Face 모델을 사용하여 hint 생성
        generated_hint = generate_additional_info_with_huggingface(paragraph, question, choices, tokenizer, model)
        
        # row에 hint 열 추가
        row["hint"] = generated_hint
        enriched_data.append(row)
    
    return pd.DataFrame(enriched_data)

# 'id' 열에서 끝 숫자 추출 및 조건 필터링
def filter_and_sample(df, n=3):
    # 'id'의 끝 숫자를 기준으로 필터링
    df['id_number'] = df['id'].str.extract(r'(\d+)$').astype(int)
    filtered_df = df[(df['id_number'] >= 450) & (df['id_number'] <= 1380)]
    
    # 무작위로 10개의 행 샘플링
    sampled_df = filtered_df.sample(n=n, random_state=random.randint(0, 1000))
    
    # 'id_number' 열 삭제 후 반환
    return sampled_df.drop(columns=['id_number'])


# ### 단순히 3개만 실험
# sample_data 폴더를 만들어야함
# test_df = pd.read_csv("../data/test_known.csv")

# sampled_test_data = filter_and_sample(test_df)

# enriched_test_df = enrich_test_data_with_huggingface(sampled_test_data, tokenizer, model)

# enriched_test_df.to_csv('../sample_data/qwen_hint.csv')
# ###


### 실행
test_df = pd.read_csv("../data/test_known.csv")

# Hugging Face 모델을 사용하여 question_plus 추가
enriched_test_df = enrich_test_data_with_huggingface(test_df, tokenizer, model)

enriched_test_df.to_csv('../data/qwen_hint.csv')
###