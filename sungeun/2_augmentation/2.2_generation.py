import time
import random

import pandas as pd
import google.generativeai as genai

from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# 데이터셋 로드
test_kiqu = load_dataset("maywell/test_kiqu")
ko_wikidata_QA = load_dataset("maywell/ko_wikidata_QA")

# 데이터 합치기
data = concatenate_datasets([test_kiqu['train'], ko_wikidata_QA['train']])

df = pd.DataFrame(data)
print(df.head())

# 새로운 데이터프레임 형식으로 변환
converted_data = {
    "id": [f"index-{i}" for i in df.index],
    "paragraph": df["output"],
    "question": df["instruction"],
    "choices": [[] for _ in df.index],  # 빈 리스트로 초기화
    "answer": [None for _ in df.index],  # None으로 초기화
    "question_plus": ["N/A" for _ in df.index]  # "N/A"로 초기화
}

converted_df = pd.DataFrame(converted_data)
print(converted_df.head())

GOOGLE_API_KEY="AIzaSyBZcAT2pLZUiTEo5lk2Dbm-zaHCgulW-2c"

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')


# 보기 및 정답 생성 함수
def generate_choices_and_answer(row):
    prompt = (
        f"아래 질문을 참고하여 문맥 기반으로 올바른 보기를 4개와 틀린 보기를 1개 생성해줘.\n"
        f"보기는 ['보기 1', '보기 2', '보기 3', '보기 4', '보기 5'] 형식으로 출력해줘.\n"
        f"틀린 보기는 반드시 답이어야 하며, 보기 중 몇 번째인지도 알려줘.\n\n"
        f"문맥:\n{row['paragraph']}\n\n"
        f"질문:\n{row['new_question']}\n\n"
        "새로운 보기와 정답을 생성해줘."
    )
    
    retries = 3  # 재시도 횟수
    for attempt in range(retries):
        try:
            # API 호출
            response = model.generate_content(prompt)
            
            # 응답 처리
            if response and hasattr(response, "text"):
                result = response.text.strip()
                parts = result.split("\n")
                
                # 보기를 추출
                choices = parts[:5] if len(parts) >= 5 else ["보기 생성 실패"] * 5
                
                # 정답을 추출
                last_line = parts[-1]
                try:
                    # 정답 번호가 "보기 5"와 같은 형식으로 포함되어 있다고 가정
                    if "보기" in last_line:
                        answer = int(last_line.split("보기")[-1].strip()[0])  # '보기 X'에서 숫자만 추출
                    else:
                        raise ValueError(f"Unexpected answer format: {last_line}")
                except ValueError:
                    print(f"Invalid answer format in response: {last_line}")
                    answer = -1  # 기본값
                
                return choices, answer
        except Exception as e:
            if "429" in str(e) or "TooManyRequests" in str(e):
                print(f"Rate limit exceeded. Retrying... ({attempt + 1}/{retries})")
                time.sleep(3 + random.uniform(0, 2))  # 고정 + 랜덤 지연
            elif "Connection" in str(e):
                print(f"Connection error. Retrying... ({attempt + 1}/{retries})")
                time.sleep(5)
            else:
                print(f"Unhandled error: {e}")
                break
    
    return ["보기 생성 실패"] * 5, -1

# 반복문으로 보기를 생성
choices_list = []
answers = []

for index, row in converted_df.iterrows():
    print(f"Processing row {index + 1}/{len(converted_df)}: {row['new_question']}")
    choices, answer = generate_choices_and_answer(row)
    choices_list.append(choices)
    answers.append(answer)
    time.sleep(3)  # 요청 간 고정 지연 추가

# DataFrame에 결과 저장
converted_df["choices"] = choices_list
converted_df["answer"] = answers

# 결과 확인
print(converted_df.head())
converted_df.to_csv("final_questions_with_choices.csv", index=False, encoding="utf-8-sig")
print("결과가 'final_questions_with_choices.csv'로 저장되었습니다.")