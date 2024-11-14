import pandas as pd
import google.generativeai as genai
import time
from tqdm import tqdm
import ast
import logging
import re

logging.basicConfig(filename='data_augmentation_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Gemini API 설정
api_key = "AIzaSyD0IrkE8UxmhUtAp25NiqiyBNhpuKomCdE"
genai.configure(api_key=api_key)

safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
]

model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings)

# Few-shot 예제 (한국어)
few_shot_example = """
단락: "공산주의라는 유령 하나가 유럽을 떠돌고 있다. 교황과 차르, 메테르니히와 기조, 프랑스 급진주의자와 독일 경찰 스파이를 포함한 구 유럽의 모든 강대국은 이 유령을 쫓아내고자 신성한 동맹을 맺었다. 이 사실로 두 결과가 발생한다. I. 이미 유럽 모든 강대국이 공산주의 자체가 하나의 세력임을 인정하고 있다. II. 이제 공산주의자는 전 세계에 맞서 견해, 목표, 경향을 공개 발표하고 공산당 자체 선언문으로 공산주의 유령이라는 동화에 대응해야 한다."

새로운 질문과 선택지를 생성하는 과정:
1. 단락을 읽고 주요 내용을 파악합니다.
2. 질문을 만들어 주요 내용을 테스트할 수 있도록 합니다.
3. 5개의 선택지와 그 중 정답을 만듭니다.

생성된 결과:
{
    "question": "이 글에서 공산주의에 대한 유럽 강대국들의 반응은 어떠한가?",
    "choices": [
        "공산주의를 환영하고 있다",
        "공산주의에 대해 무관심하다",
        "공산주의를 억압하려 동맹을 맺었다",
        "공산주의와 협력 관계를 맺고 있다",
        "공산주의의 존재를 부정하고 있다"
    ],
    "answer": 3
}
"""

def generate_question(paragraph, original_question, original_choices, original_answer):
    prompt = f"""
    Based on the given paragraph, create a multi-choice problem with 5 choices and give the correct answer. Describe the process of creating problems, options, and answers using a Chain of Thought (CoT) approach.

    Few-shot example:
    {few_shot_example}

    Now, generate a question for the following paragraph:
    
    Paragraph: {paragraph}

    Original Question: {original_question}
    Original Choices: {original_choices}
    Original Answer: {original_answer}

    Please follow these steps:
    1. Analyze the main points of the paragraph.
    2. Consider the original question and how it relates to the paragraph.
    3. Formulate a new question that tests understanding of the paragraph in a different way.
    4. Create 5 answer choices, including the correct answer and 4 plausible distractors.
    5. Choose the "ONLY ONE" correct answer (1, 2, 3, 4, or 5).
    6. Question, Choices, Answer MUST BE Korean
    7. Provide your response just like in the following format.:
    
    New Question: [Your new question here]
    New Choices: 
    1. [Choice 1]
    2. [Choice 2]
    3. [Choice 3]
    4. [Choice 4]
    5. [Choice 5]
    New Answer: [Correct answer number]
    
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"질문 생성 중 오류 발생: {str(e)}")
        return None


# CSV 파일 읽기
df = pd.read_csv('classified_train_processed_filtered.csv')

# 결과를 저장할 새로운 DataFrame
results = []

# 각 행에 대해 새로운 문제 생성
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="새로운 질문 생성 중"):
    paragraph = row['paragraph']
    problems = ast.literal_eval(row['problems'])
    
    original_question = problems['question']
    original_choices = problems['choices']
    original_answer = problems['answer']
    
    generated_content = generate_question(paragraph, original_question, original_choices, original_answer)
    
    if generated_content:
        # 응답을 수동으로 파싱하여 필요한 정보를 추출합니다.
        lines = generated_content.split('\n')
        new_question = ""
        new_choices = []
        new_answer = None
        reading_choices = False
        
        for line in lines:
            if line.startswith("**New Question:**"):
                new_question = line.replace("**New Question:**", "").strip()
            elif line.startswith("**New Choices:**"):
                reading_choices = True
                continue
            elif reading_choices:
                if line.strip().startswith(tuple(str(i) for i in range(1, 6))):
                    choice_text = line.strip().split(". ", 1)[-1]
                    new_choices.append(choice_text)
                elif line.strip() == "":
                    reading_choices = False  # Stop reading choices if we encounter an empty line
            elif line.startswith("**New Answer:**"):
                try:
                    new_answer = int(line.replace("**New Answer:**", "").strip())
                except ValueError:
                    logging.error(f"Invalid answer format in row {index}: {line}")
        
        if new_question and new_choices and new_answer is not None:
            new_row = {
                'id': f"{row['id']}_aug",
                'paragraph': paragraph,
                'problems': {
                    'question': new_question,
                    'choices': new_choices,
                    'answer': new_answer
                },
                'question_plus': ''
            }
            results.append(new_row)
        else:
            logging.warning(f"Failed to parse generated content for row {index}.")
    
    else:
        logging.warning(f"{index}번 행에 대한 질문 생성 실패. 다음 행으로 넘어갑니다.")
    
    # API 호출 제한을 고려한 대기 시간
    time.sleep(4)

# 결과를 새로운 DataFrame으로 변환
result_df = pd.DataFrame(results)

# 결과를 CSV 파일로 저장
result_df.to_csv('augmented_questions.csv', index=False)

print("데이터 증강이 완료되었습니다. 결과가 'augmented_questions.csv' 파일에 저장되었습니다.")
