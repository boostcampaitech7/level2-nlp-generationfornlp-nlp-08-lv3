import pandas as pd
import ast

# CSV 파일 읽기
file_path = '../data/test.csv'  # 파일 경로를 적절히 수정하세요.
data = pd.read_csv(file_path)

# 일치 및 불일치 데이터를 저장할 리스트
matching_rows = []
non_matching_rows = []

# 데이터를 순회하며 확인
for index, row in data.iterrows():
    paragraph = row['paragraph']
    choices_dict = ast.literal_eval(row['problems'])  # 문자열을 딕셔너리로 변환
    choices = choices_dict.get('choices', [])  # 'choices' 키의 값 가져오기

    # choices의 문장들이 paragraph에 포함되는지 확인
    is_matching = any(choice in paragraph for choice in choices)  # 하나라도 포함되면 True

    if is_matching:
        matching_rows.append(row)  # 일치하는 데이터 추가
    else:
        non_matching_rows.append(row)  # 불일치 데이터 추가

# 일치하는 데이터프레임 생성
matching_data = pd.DataFrame(matching_rows)
non_matching_data = pd.DataFrame(non_matching_rows)

# CSV 파일로 저장
matching_file = 'test_odqa.csv'
non_matching_file = 'test_known.csv'

matching_data.to_csv(matching_file, index=False, encoding='utf-8-sig')
non_matching_data.to_csv(non_matching_file, index=False, encoding='utf-8-sig')

print(f"일치하는 데이터가 {matching_file}로 저장되었습니다!")
print(f"일치하지 않는 데이터가 {non_matching_file}로 저장되었습니다!")