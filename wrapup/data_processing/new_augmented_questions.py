import pandas as pd
import openai
import time
from tqdm import tqdm
import ast
import logging
import re

logging.basicConfig(filename='data_augmentation_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# OpenAI API 설정
api_key = "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"
openai.api_key = api_key

def clean_text(text):
    # 괄호와 괄호 안의 모든 문자 제거
    text = re.sub(r'\([A-Za-z\s]*\)', '', text)
    text = text.replace('*', '')
    return text

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
        - 5개의 선택지를 포함하되, 정답은 1개만 포함하고 대부분 선택지 5에 위치하게 생성합니다. 
        - 한국어로 작성하며, 다음 형식으로 작성합니다:

        질문: [새 질문]
        선택지:
        1. [선택지 1]
        2. [선택지 2]
        3. [선택지 3]
        4. [선택지 4]
        5. [선택지 5]
        정답: [정답 번호]

        - 이 때 정답 번호는 생성한 질문 중 정답인 선택지 번호 입니다. 
        """}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=400
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"질문 생성 중 오류 발생: {str(e)}")
        return None

df = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/classified_train_processed_filtered.csv')

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
                
                if line.startswith("질문"):
                    new_question = line.replace("질문 : ", "").strip()
                
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
            
            if (new_question) and (len(new_choices) == 5) and (new_answer is not None):
                new_row = {
                    'id': f"{row['id']}_aug",
                    'paragraph': paragraph,
                    'problems': {
                        'question': new_question,
                        'choices': new_choices,
                        'answer': new_answer
                    },
                    'question_plus': ''
                }
                results.append(new_row)
                success = True
                break  # 성공적으로 생성되었으므로 추가 시도 중단
        
        if not success:
            time.sleep(2)  # 재시도 전 잠시 대기
    
    if not success:
        failed_rows.append(row)
        logging.warning(f"{index}번 행에 대한 질문 생성 실패. 2번의 시도 후 건너뜁니다.")
    
    time.sleep(4)

# 결과를 새로운 DataFrame으로 변환
result_df = pd.DataFrame(results)
# 결과를 CSV 파일로 저장
result_df.to_csv('../data/new_augmented_questions.csv', index=False)

# 실패한 행들을 CSV 파일로 저장
failed_df = pd.DataFrame(failed_rows)
failed_df.to_csv('failed_rows.csv', index=False)

print(f"데이터 증강이 완료되었습니다. 결과가 'new_augmented_questions.csv' 파일에 저장되었습니다.")