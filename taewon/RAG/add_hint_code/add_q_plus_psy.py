import csv
import ast
import openai
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42) # magic number :)

# OpenAI API 키 설정
openai.api_key = "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"

# GPT 모델을 사용하여 지문과 질문 처리
def process_paragraph_and_question(paragraph, question):
    prompt = f"""당신은 심리학 전문가입니다. 다음 지문과 질문을 읽고, 질문에 답하기 위해 필요한 중요한 정보와 관련된 추가적인 심리학적 맥락이나 배경 지식을 제공해주세요. 특히 지문에 언급된 인간 행동, 인지 과정, 감정, 발달 단계, 심리적 장애, 치료 방법 등에 대해 더 자세한 설명이 필요할 경우 그 정보를 포함해주세요.

    <지문>
    {paragraph}

    <질문>
    {question}

    <힌트>
    (지문의 내용과 질문을 바탕으로 중요한 심리학적 정보와 추가적인 맥락을 제공해주세요. 답변은 간결하고 명확하게 300자 이내로 작성해주세요. 필요한 경우 관련 심리학 이론, 연구 결과, 실험 데이터, 또는 사례 연구에 대한 정보를 포함할 수 있습니다.)
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a Korean history expert."},
                  {"role": "user", "content": prompt}]
    )

    # Clean up the output to replace spaces with a single space and remove newlines
    result = response['choices'][0]['message']['content'].strip()
    result = ' '.join(result.split())  # Replace multiple spaces with a single space

    return result

# CSV 파일 읽기 및 처리
def process_csv(file_path):
    df = pd.read_csv(file_path)
    df['hint'] = ""  # 새로운 'hint' 열 추가
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        paragraph = row['paragraph']
        problems = ast.literal_eval(row['problems'])
        question = problems['question']
        hint = process_paragraph_and_question(paragraph, question)
        df.at[index, 'hint'] = hint

    return df

# CSV 파일 처리 및 결과 저장
input_file = './raw_data/mmmlu_psy.csv'
output_file = './hint/mmmlu_psy_add_hint.csv'

processed_df = process_csv(input_file)
processed_df.to_csv(output_file, index=False)

print(f"처리가 완료되었습니다. 결과가 {output_file}에 저장되었습니다.")