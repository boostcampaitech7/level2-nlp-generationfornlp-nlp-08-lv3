import pandas as pd
import ast

# CSV 파일 읽기
unc = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/new_train3_cleaned286.csv')
c = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/adjusted_data.csv')

# 'problems' 컬럼에서 JSON 문자열로 저장된 데이터를 파싱
c_ans = c['problems'].apply(ast.literal_eval)

# 'answer' 키의 값을 추출
c_ans = c_ans.apply(lambda x: x['answer'])

# 고유한 answer 값의 개수 확인
answer_counts = c_ans.value_counts()

# 결과 출력
print(answer_counts)
print(len(unc))
