from datasets import load_dataset
import pandas as pd

# MMMLU 한국어 데이터셋 로드
dataset = load_dataset("openai/MMMLU", "KO_KR")

# 선택할 고등학교 수준 과목들
selected_subjects = [
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history"
]

def convert_answer_to_number(answer):
    return {'A': 1, 'B': 2, 'C': 3, 'D': 4}.get(answer, 0)

# 데이터 전처리 함수
def preprocess_mmlu(example, idx):
    choices = [example['A'], example['B'], example['C'], example['D']]
    
    # 마지막 줄바꿈을 기준으로 분리
    parts = example['Question'].rsplit('\n', 1)
    
    if len(parts) > 1:
        paragraph, question = parts
    else:
        paragraph = ""
        question = parts[0]
    
    return {
        'id': f"mmlu_{example['Subject']}_{idx}",
        'paragraph': paragraph.strip(),
        'question': question.strip(),
        'choices': choices,
        'answer': convert_answer_to_number(example['Answer'])
    }

# 선택된 과목의 데이터만 필터링하고 전처리
all_data = []
idx = 0

for split in ['test']:  # MMMLU KO_KR 데이터셋은 test 분할만 있습니다
    filtered_data = [ex for ex in dataset[split] if ex['Subject'] in selected_subjects]
    for example in filtered_data:
        all_data.append(preprocess_mmlu(example, idx))
        idx += 1

# DataFrame으로 변환
df = pd.DataFrame(all_data)

# CSV 파일로 저장
df.to_csv('mmmlu_history.csv', index=False)

print("처리된 데이터 샘플:")
print(df.head())
print(f"\n총 데이터 수: {len(df)}")