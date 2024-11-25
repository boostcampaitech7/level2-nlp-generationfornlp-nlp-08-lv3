from datasets import load_dataset
import pandas as pd

# KMMLU 데이터셋 로드
dataset = load_dataset("HAERAE-HUB/KMMLU", "Korean-History")

def convert_answer_to_number(answer):
    return {'A': 1, 'B': 2, 'C': 3, 'D': 4}.get(answer, 0)
# 데이터 전처리 함수
def preprocess_kmmlu(example, idx):
    # '?' 를 기준으로 question을 split
    split_question = example['question'].split('?', 1)
    
    # split 결과에 따라 question과 paragraph 설정
    if len(split_question) > 1:
        question = split_question[0] + '?'
        paragraph = split_question[1].strip()
    else:
        question = example['question']
        paragraph = ""
    
    # 선택지 생성
    choices = [example['A'], example['B'], example['C'], example['D']]
    
    return {
        'id': f"kmmlu_korean_history_{idx}",
        'paragraph': paragraph,
        'question': question,
        'choices': choices,
        'answer': example['answer']  # 숫자를 알파벳으로 변환
    }

# 모든 split(train, validation, test)의 데이터를 하나로 합치기
all_data = []
for split in dataset.keys():
    all_data.extend(dataset[split])

# 데이터셋 전처리
processed_data = [preprocess_kmmlu(example, idx) for idx, example in enumerate(all_data)]

# DataFrame으로 변환
df = pd.DataFrame(processed_data)

# CSV 파일로 저장
df.to_csv('kmmlu_korean_history_processed.csv', index=False)

print("처리된 데이터 샘플:")
print(df.head())
print(f"\n총 데이터 수: {len(df)}")