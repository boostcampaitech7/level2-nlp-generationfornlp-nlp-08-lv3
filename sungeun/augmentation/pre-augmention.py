import pandas as pd
import ast

# CSV 파일 읽기
df = pd.read_csv('classified_train_processed.csv')

# prior_knowledge_needed가 "필요 없음"인 행만 선택
df_filtered = df[df['prior_knowledge_needed'] == "필요 없음"]

# problems 열 생성
df_filtered['problems'] = df_filtered.apply(lambda row: {
    'question': row['question'],
    'choices': ast.literal_eval(row['choices']) if isinstance(row['choices'], str) else row['choices'],
    'answer': row['answer']
}, axis=1)

# 기존 열 삭제
df_filtered = df_filtered.drop(columns=['question', 'choices', 'answer'])

# 결과 확인
print(df_filtered.head())

# 새로운 CSV 파일로 저장
df_filtered.to_csv('classified_train_processed_filtered.csv', index=False)
