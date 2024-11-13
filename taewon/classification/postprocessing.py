import pandas as pd
import re

# CSV 파일 읽기
df = pd.read_csv('classified_train_raw.csv')

# ID에서 숫자를 추출하여 정렬하는 함수 정의
def extract_number_from_id(id_str):
    # ID에서 숫자 부분만 추출 (정수형으로 변환)
    match = re.search(r'\d+', id_str)
    return int(match.group()) if match else float('inf')  # 숫자가 없으면 무한대 반환

# 후처리 함수 정의 (topic 처리)
def clean_topic(topic):
    # 괄호와 괄호 안의 모든 문자 제거
    topic = re.sub(r'\(.*?\)', '', topic)
    # 양끝의 '*' 문자 제거
    topic = topic.strip('*')
    # 주어진 카테고리 리스트
    categories = ['문학', '독해', '언어와 매체', '사회', '한국사', '세계사', '경제', '지리', '심리',
                  '교육산업', '국제', '부동산', '생활', '책마을']
    
    # 카테고리 중 하나만 남기기, 없으면 "ETC"로 처리
    for category in categories:
        if category in topic:
            return category
    return "ETC"  # 해당하는 카테고리가 없으면 "ETC" 반환

# prior_knowledge_needed 열 후처리
df['prior_knowledge_needed'] = df['prior_knowledge_needed'].apply(lambda x: '필요 없음' if '필요 없음' in x else '필요함')

# topic 열 후처리
df['topic'] = df['topic'].apply(clean_topic)

# ID를 숫자 기준으로 정렬
df = df.sort_values(by='id', key=lambda x: x.apply(extract_number_from_id))

# 결과 확인 (첫 5개 행 출력)
#print(df.head())

# 필요시 결과를 새로운 CSV 파일로 저장할 수 있습니다.
df.to_csv('classified_train_processed.csv', index=False)