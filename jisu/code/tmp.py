# # from transformers import AutoModelForCausalLM
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# print(model)

# import pandas as pd
# import re

# # CSV 파일 읽기
# file_path = './test_odqa.csv'  # 파일 경로를 적절히 수정하세요.
# data = pd.read_csv(file_path)

# # 1380 이상인 ID를 저장할 리스트
# filtered_ids = []
# count = 0

# # 데이터를 순회하며 ID의 뒷부분 숫자를 확인
# for index, row in data.iterrows():
#     id_str = row['id']
#     match = re.search(r'(\d{1,4})$', id_str)  # ID의 마지막 숫자 추출
#     if match:
#         number = int(match.group(1))  # 숫자로 변환
#         if number < 1380:
#             filtered_ids.append(id_str)
#             count += 1

# # 결과 출력
# print("ID의 뒷부분 숫자가 1380 이상인 ID:")
# for filtered_id in filtered_ids:
#     print(filtered_id)
# print(count)

import pandas as pd

# 두 CSV 파일 읽기 (한글 인코딩 처리)
df1 = pd.read_csv('../data/js_new_train3_cleaned286.csv', encoding='utf-8-sig')
df2 = pd.read_csv('../data/new_quiz_format.csv', encoding='utf-8-sig')

# 데이터프레임 수직 연결
combined_df = pd.concat([df1, df2], ignore_index=True)

# 연결된 데이터프레임 저장 (한글 인코딩 처리)
combined_df.to_csv('../data/combined_file.csv', index=False, encoding='utf-8-sig')

# import pandas as pd

# # CSV 파일 읽기 (한글 인코딩 처리)
# df = pd.read_csv('../data/new_train3_cleaned286.csv', encoding='utf-8-sig')

# # 첫 번째 열 제거
# df = df.iloc[:, 1:]  # 모든 행과 첫 번째 열 제외

# # 변경된 데이터프레임 저장
# df.to_csv('../data/js_new_train3_cleaned286.csv', index=False, encoding='utf-8-sig')
