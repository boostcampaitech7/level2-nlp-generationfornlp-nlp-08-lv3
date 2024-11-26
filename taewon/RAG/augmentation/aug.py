import pandas as pd
import openai
import re
from tqdm import tqdm
import torch
import numpy as np
import random
import time
import ast
import logging

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42) # magic number :)

logging.basicConfig(filename='data_augmentation_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# OpenAI API 설정
api_key = "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"
openai.api_key = api_key

# Prompt (간결화된 한국어)
def generate_question(paragraph, original_question, original_choices, original_answer):
    prompt = [
        {"role": "system", "content": "당신은 단락과 그에 대한 예시 문제들을 참고하여 사전지식 없이 주어진 단락 내용을 읽고 추론하여 풀 수 있는 오지선다형 문제를 새롭게 생성하는 AI입니다."},
        {"role": "user", "content": f"""
        주어진 단락을 바탕으로 새로운 질문을 만들어주세요. 

        단락:
        {paragraph}

        원래 질문: {original_question}
        원래 선택지: {original_choices}
        원래 정답: {original_answer}

        요청 사항:
        - 한국어로 생성하며 원래 질문, 원래 선택지, 원래 정답을 참고하여 원래 내용과 다른 질문, 선택지, 정답을 생성하세요.
        - 단락의 주요 내용을 확인합니다.
        - 단락의 주요 내용을 바탕으로 원래 질문의 내용과 다른 질문을 생성합니다.
        - 5개의 선택지를 생성하되, 정답은 반드시 1개입니다.
        - 나머지 선택지의 경우 그럴듯한 오답을 생성합니다.
        - 문제를 푸는 데 필요한 간단한 해설(힌트)을 추가로 200자 이내로 작성합니다.
        - 한국어로 작성하며, 다음 형식으로 작성합니다:

        질문: [새 질문]
        선택지:
        1. [선택지 1]
        2. [선택지 2]
        3. [선택지 3]
        4. [선택지 4]
        5. [선택지 5]
        정답: [정답 번호]
        힌트: [문제 해설]

        - 이 때 정답 번호는 생성한 질문 중 정답인 선택지 번호 입니다. 
        """}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=prompt,
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"질문 생성 중 오류 발생: {str(e)}")
        return None

df = pd.read_csv('need_knowledge.csv')

# 결과를 저장할 새로운 DataFrame
results = []
failed_rows = []

# 각 행에 대해 새로운 문제 생성
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="새로운 질문 생성 중"):
    #print(index)
    paragraph = row['paragraph']
    problems = ast.literal_eval(row['problems'])
    
    original_question = problems['question']
    original_choices = problems['choices']
    original_answer = problems['answer']
    
    success = False
    for attempt in range(2): 
        generated_content = generate_question(paragraph, original_question, original_choices, original_answer)
        #print(f"generate: {generated_content}")


        if generated_content:
            # 응답을 수동으로 파싱하여 필요한 정보를 추출합니다.
            lines = generated_content.split('\n')
            new_question = ""
            new_choices = []
            new_answer = None
            reading_choices = False
            count = 0
            
            for line in lines:
                line = line.strip()
                #print(line)
                
                if line.startswith("질문"):
                    new_question = line.replace("질문: ", "").strip()
                
                elif line.startswith("선택지"):
                    reading_choices = True
                    continue
                
                elif reading_choices:
                    if line.strip().startswith(tuple(str(i) for i in range(1, 6))):
                        choice_text = line.strip().split(". ", 1)[-1]
                        new_choices.append(choice_text)
                        if len(new_choices) == 5:
                            reading_choices = False
                
                elif line.startswith("정답"):
                    try:
                        new_answer = int(line.split(":", 1)[-1].strip())
                    except ValueError:
                        logging.error(f"Invalid answer format: {line}")
                        
                elif line.startswith("힌트"):
                    hint = line.replace("힌트: ", "").strip()
            
            if (new_question) and (len(new_choices) == 5) and (new_answer is not None) and hint:
                new_row = {
                    'id': f"{row['id']}_aug",
                    'paragraph': paragraph,
                    'problems': {
                        'question': new_question,
                        'choices': new_choices,
                        'answer': new_answer
                    },
                    'question_plus': '',
                    'hint': hint
                }
                results.append(new_row)
                success = True
                break  # 성공적으로 생성되었으므로 추가 시도 중단
        
        if not success:
            time.sleep(2)  # 재시도 전 잠시 대기
    
    if not success:
        failed_rows.append(row)
        logging.warning(f"{index}번 행에 대한 질문 생성 실패. 2번의 시도 후 건너뜁니다.")
    

# 결과를 새로운 DataFrame으로 변환
result_df = pd.DataFrame(results)
# 결과를 CSV 파일로 저장
result_df.to_csv('need_knowledge_aug.csv', index=False)

# 실패한 행들을 CSV 파일로 저장
failed_df = pd.DataFrame(failed_rows)
failed_df.to_csv('failed_rows.csv', index=False)

print(f"데이터 증강이 완료되었습니다. 결과가 'new_augmented_questions.csv' 파일에 저장되었습니다.")