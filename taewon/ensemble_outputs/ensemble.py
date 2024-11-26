import pandas as pd

# 기본 양식 파일 읽기
base_df = pd.read_csv("../data/output.csv")

# ODQA 모델과 Known 모델의 결과 파일 읽기
odqa_df = pd.read_csv("../output_odqa_model/output_odqa.csv")
known_df = pd.read_csv("../output_known_model/output_known.csv")

# id를 기준으로 세 데이터프레임 병합, 열 이름 명시적 지정
merged_df = pd.merge(base_df[['id']], odqa_df[['id', 'answer']], on='id', how='left', suffixes=('', '_odqa'))
merged_df = pd.merge(merged_df, known_df[['id', 'answer']], on='id', how='left', suffixes=('_odqa', '_known'))

# 새로운 answer 컬럼 생성
merged_df['answer'] = merged_df['answer_odqa'].combine_first(merged_df['answer_known'])
merged_df['answer'] = merged_df['answer'].astype(int)

# 최종 데이터프레임 생성 (id와 answer만 포함)
final_df = merged_df[['id', 'answer']]

# CSV 파일로 저장
final_df.to_csv("output_ensemble.csv", index=False, encoding="utf-8-sig")