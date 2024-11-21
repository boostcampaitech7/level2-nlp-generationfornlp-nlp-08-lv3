# 1. answer = -1인 행 제거
filtered_df = converted_df[converted_df['answer'] != -1]

# 2. choices에서 "보기 1:", "보기 2:" 등의 텍스트 제거
filtered_df['choices'] = filtered_df['choices'].apply(
    lambda x: [choice.replace('보기 1:', '').replace('보기 2:', '').replace('보기 3:', '').replace('보기 4:', '').replace('보기 5:', '').strip() for choice in x]
)

# 3. choices의 길이가 4 미만인 행 제거
filtered_df = filtered_df[filtered_df['choices'].apply(len) >= 4]

# 4. 기존 question 컬럼 제거 후, new_question을 question으로 대체하고 위치 이동
filtered_df = filtered_df.drop(columns=['question'])  # 기존 question 삭제
filtered_df = filtered_df.rename(columns={'new_question': 'question'})  # new_question -> question

# question을 paragraph 옆으로 이동
columns = ['id', 'paragraph', 'question', 'choices', 'answer', 'question_plus']
filtered_df = filtered_df[columns]

# 결과 확인
print(filtered_df.head())

# (옵션) 결과 저장
filtered_df.to_csv("processed_dataset.csv", index=False, encoding="utf-8-sig")