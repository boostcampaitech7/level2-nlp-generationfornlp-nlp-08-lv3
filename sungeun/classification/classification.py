import pandas as pd
import google.generativeai as genai
import time
from tqdm import tqdm
import ast
import logging

logging.basicConfig(filename='classification_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')
# Gemini API 설정

# API 키 직접 입력
api_key = "key"  # 여기에 API 키를 입력하세요

# Gemini API 설정
genai.configure(api_key=api_key)

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    }
]

model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings)

# 주제 카테고리 정의
categories = {
    'Korean': ['문학', '독해', '언어와 매체'],
    'Social Studies': {
        '윤리': ['정의', '공정성', '인권'],
        '정치': ['정치제도', '정당', '선거'],
        '사회': ['사회구조', '문화', '경제']
    },
    'Datasets': {
        'KMMLU': ['한국사'],
        'MMMLU': ['역사', '경제', '정치', '지리', '심리'],
        'KLUE MRC': ['경제', '교육산업', '국제', '부동산', '사회', '생활', '책마을']
    }
}

def classify_question(paragraph, question, choices, question_plus):
    system_prompt = """
    당신은 한국 수능 문제 분석 전문가입니다. 주어진 문제를 분석하여 정확한 주제를 파악하고, 
    사전지식 필요 여부를 판단해야 합니다. 다음 지침을 따라주세요:

    1. 주제 분류: 문학, 독해, 언어와 매체, 사회, 한국사, 세계사, 경제, 지리, 심리, 
       교육산업, 국제, 부동산, 생활, 책마을 중 하나를 선택하세요.
    2. 사전지식 필요 여부: 
       - '필요 없음': 문단을 읽고 답을 유추할 수 있는 경우
       - '필요함': 문단 외의 추가적인 지식이 필요한 경우
    3. 응답 형식을 정확히 지켜주세요.

    분석 시 지문, 문제, 선택지, 보기를 모두 고려하세요.
    """

    user_prompt = f"""
    다음 수능 문제를 분석해주세요:

    지문: {paragraph}
    문제: {question}
    선택지: {choices}
    보기: {question_plus}

    위 문제에 대해 다음 정보를 제공해주세요:
    1. 이 문제의 주제
    2. 이 문제를 풀기 위해 사전지식이 필요한지 여부 (문단만으로 답변 가능한지 고려)

    답변 형식:
    주제: [주제]
    사전지식 필요: [필요함/필요 없음]
    """

    response = model.generate_content([
        {"role": "user", "parts": [system_prompt]},
        {"role": "user", "parts": [user_prompt]}
    ])
    return response.text

# CSV 파일 읽기
df = pd.read_csv('train.csv')

# 결과를 저장할 리스트
results = []

# 각 행에 대해 분류 수행
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="문제 분류 중"):
    try:
        paragraph = row['paragraph']
        problems = ast.literal_eval(row['problems'])
        question = problems['question']
        choices = problems['choices']
        answer = problems['answer']
        question_plus = row['question_plus']
        
        classification = classify_question(paragraph, question, choices, question_plus)
        
        if classification:
            try:
                # 결과 파싱
                topic = classification.split('\n')[0].split(': ')[1]
                prior_knowledge = classification.split('\n')[1].split(': ')[1]
            except IndexError:
                logging.error(f"Error parsing classification for row {index}: {classification}")
                topic = "분류 실패"
                prior_knowledge = "분류 실패"
        else:
            topic = "분류 실패"
            prior_knowledge = "분류 실패"
        
        results.append({
            'id': row['id'],
            'topic': topic,
            'prior_knowledge_needed': prior_knowledge,
            'paragraph': paragraph,
            'question': question,
            'choices': choices,
            'answer': answer,
            'question_plus': question_plus
        })
        # 5초 대기
        time.sleep(5)
    
    except Exception as e:
        logging.error(f"Error processing row {index}: {str(e)}")
        continue  # 오류 발생 시 다음 row로 넘어감

result_df = pd.DataFrame(results)

# 최종 결과 저장
result_df.to_csv('classified_train.csv', index=False)

print("분류 작업이 완료되었습니다.")