import pandas as pd
import ast

# 단어 단위 포함 비율 계산 함수
def calculate_coverage(choice, paragraph):
    words = choice.split()  # choice를 단어 단위로 나눔
    matched_words = [word for word in words if word in paragraph]  # paragraph에 포함된 단어
    return len(matched_words) / len(words) if words else 0  # 포함된 비율 반환

# 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ'이 choices에 포함되었는지 확인
def contains_special_chars(choices):
    special_chars = {'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ'}
    for choice in choices:
        if any(char in choice for char in special_chars):
            return True
    return False

# CSV 파일 읽기
file_path = './test_known.csv'  # 파일 경로를 적절히 수정하세요.
data = pd.read_csv(file_path)

# 80% 이상 포함 여부에 따라 데이터를 저장할 리스트
matching_rows = []
non_matching_rows = []
perenct = 0.65

# 데이터를 순회하며 확인
for index, row in data.iterrows():
    paragraph = row['paragraph']
    choices_dict = ast.literal_eval(row['problems'])  # 문자열을 딕셔너리로 변환
    choices = choices_dict.get('choices', [])  # 'choices' 키의 값 가져오기

    # 1. choices에 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ'이 포함된 경우
    if contains_special_chars(choices):
        non_matching_rows.append(row)
        continue  # 다음 데이터로 넘어감

    # 2. paragraph 길이가 100자보다 짧은 경우
    if len(paragraph) < 100:
        non_matching_rows.append(row)
        continue  # 다음 데이터로 넘어감

    # 3. 각 choice의 포함 비율 확인
    is_matching = any(calculate_coverage(choice, paragraph) >= perenct for choice in choices)

    if is_matching:
        matching_rows.append(row)  # percent 이상 포함된 데이터 추가
    else:
        non_matching_rows.append(row)  # 그렇지 않은 데이터 추가

# 일치하는 데이터프레임 생성
matching_data = pd.DataFrame(matching_rows)
non_matching_data = pd.DataFrame(non_matching_rows)

# CSV 파일로 저장
matching_file = 'odqa_matching.csv'
non_matching_file = 'unknown.csv'

matching_data.to_csv(matching_file, index=False, encoding='utf-8-sig')
non_matching_data.to_csv(non_matching_file, index=False, encoding='utf-8-sig')

print(f"{perenct} 이상 포함된 데이터가 {matching_file}로 저장되었습니다!")
print(f"100자 미만 포함 및 {perenct} 미만 데이터가 {non_matching_file}로 저장되었습니다!")
