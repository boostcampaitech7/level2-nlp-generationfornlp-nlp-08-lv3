import pandas as pd

# CSV 파일 읽기
file_path = '../data/train.csv'  # 파일 경로 수정 필요
df = pd.read_csv(file_path)

# 길이에 따른 기준 설정
length_threshold = 100  # 임의의 길이 기준, 원하는 기준으로 수정 가능

# paragraph 길이에 따라 두 개의 데이터프레임으로 나누기
short_paragraph_df = df[df['paragraph'].str.len() <= length_threshold]
long_paragraph_df = df[df['paragraph'].str.len() > length_threshold]

# CSV 파일로 각각 저장
short_paragraph_df.to_csv('short_paragraphs.csv', index=False, encoding='utf-8-sig')
long_paragraph_df.to_csv('long_paragraphs.csv', index=False, encoding='utf-8-sig')

print("CSV 파일이 성공적으로 저장되었습니다.")