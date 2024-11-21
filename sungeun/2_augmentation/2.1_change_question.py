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

# # 질문 변환 함수
# def transform_question_with_genai(question):
#     prompt = (
#         f"아래 문장을 참고하여 질문을 '~에 대한 설명으로 옳지 않은 것은?' 형식으로 바꿔줘:\n\n"
#         f"질문: {question}\n\n"
#         "새로운 질문:"
#     )

#     retries = 3  # 재시도 횟수
#     for attempt in range(retries):
#         try:
#             # API 호출
#             response = model.generate_content(prompt)

#             # 응답 처리
#             if response and hasattr(response, "text"):
#                 return response.text.strip()
#         except Exception as e:
#             if "429" in str(e) or "TooManyRequests" in str(e):
#                 print(f"Rate limit exceeded. Retrying... ({attempt + 1}/{retries})")
#                 time.sleep(3 + random.uniform(0, 2))  # 고정 + 랜덤 지연
#             elif "Connection" in str(e):
#                 print(f"Connection error. Retrying... ({attempt + 1}/{retries})")
#                 time.sleep(5)
#             else:
#                 print(f"Unhandled error: {e}")
#                 break

#     return "변환 실패"

# new_questions = []
# for index, row in tqdm(converted_df.iterrows(), total=len(converted_df), desc="Processing rows"): # tqdm 추가
#     # 진행 상황과 함께 출력되는 부분
#     print(f"Processing row {index + 1}/{len(converted_df)}: {row['question']}")
#     new_question = transform_question_with_genai(row['question'])
#     new_questions.append(new_question)
#     time.sleep(3)  # 요청 간 고정 지연 추가

# # 결과 저장
# converted_df["new_question"] = new_questions

# # 결과 확인
# print(converted_df.head())
# converted_df.to_csv("transformed_questions.csv", index=False, encoding='utf-8-sig')
# print("결과가 'transformed_questions.csv'로 저장되었습니다.")

################

# 질문 변환 함수 (기존과 동일)
def transform_question_with_genai(question):
    prompt = (
        f"아래 문장을 참고하여 질문을 '~에 대한 설명으로 옳지 않은 것은?' 형식으로 바꿔줘:\n\n"
        f"질문: {question}\n\n"
        "새로운 질문:"
    )
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            if response and hasattr(response, "text"):
                return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "TooManyRequests" in str(e):
                time.sleep(3 + random.uniform(0, 2))
            elif "Connection" in str(e):
                time.sleep(5)
            else:
                print(f"Unhandled error: {e}")
                return "변환 실패"
    return "변환 실패"

# 병렬 처리 함수
def process_row(index, question):
    new_question = transform_question_with_genai(question)
    return index, new_question

# 데이터 변환 및 병렬 처리
new_questions = [""] * len(converted_df)  # 결과를 저장할 리스트
with ThreadPoolExecutor(max_workers=10) as executor:  # 병렬로 10개 작업
    futures = [executor.submit(process_row, index, row['question']) for index, row in converted_df.iterrows()]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
        index, new_question = future.result()
        new_questions[index] = new_question  # 결과 저장

# 결과 추가
converted_df["new_question"] = new_questions

# 결과 저장
converted_df.to_csv("transformed_questions.csv", index=False, encoding='utf-8-sig')
print("결과가 'transformed_questions.csv'로 저장되었습니다.")
