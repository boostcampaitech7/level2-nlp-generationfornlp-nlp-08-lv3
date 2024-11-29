import pandas as pd

# 두 개의 CSV 파일 로드
csv1 = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test.csv')
csv2 = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test_known_hint_final.csv')

# 'id' 열을 기준으로 두 DataFrame을 병합
merged_df = pd.merge(csv1, csv2[['id', 'hint']], on='id', how='left')

# NaN인 경우 "이 문제의 답변은 주어진 지문 안에 있습니다."를 넣도록 수정
merged_df['hint'] = merged_df['hint'].fillna("이 문제의 답변은 주어진 지문 안에 있습니다.")

# 'paragraph'와 'hint' 결합
merged_df['paragraph'] = merged_df['paragraph'] + '\n' + merged_df['hint']

# 불필요한 'hint' 열 삭제
merged_df = merged_df.drop(columns=['hint'])

# 결과를 새로운 CSV로 저장
merged_df.to_csv('merged_file_with_filled_hint.csv', index=False)

