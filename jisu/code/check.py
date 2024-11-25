"""
csv 파일 안에 id가 1380 이상/이하인 데이터의 개수를 출력해주는 파이썬 코드입니다.

csv 파일, 이상, 이하, 1380 등은 사용자가 원하시는 대로 바꿔서 사용하면 됩니다.
"""

import pandas as pd
import re

# CSV 파일 읽기
file_path = './test_known.csv'  # 파일 경로를 적절히 수정하세요.
data = pd.read_csv(file_path)

# 1380 이상인 ID를 저장할 리스트
filtered_ids = []
count = 0

# 데이터를 순회하며 ID의 뒷부분 숫자를 확인
for index, row in data.iterrows():
    id_str = row['id']
    match = re.search(r'(\d{1,4})$', id_str)  # ID의 마지막 숫자 추출
    if match:
        number = int(match.group(1))  # 숫자로 변환
        if number < 1380:
            filtered_ids.append(id_str)
            count += 1

# 결과 출력
print("ID의 뒷부분 숫자가 1380 이상인 ID:")
for filtered_id in filtered_ids:
    print(filtered_id)
print(count)