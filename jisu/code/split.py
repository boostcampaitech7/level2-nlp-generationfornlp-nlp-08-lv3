import pandas as pd

# 엑셀 파일 경로 설정
file_path = '../data/qa_dataset.xlsx'

# 엑셀 파일 읽기
data = pd.read_excel(file_path)

# 데이터 확인
print(data.head())  # 데이터프레임의 상위 5개 행 출력

