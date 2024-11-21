from datasets import load_dataset
import pandas as pd

# KLUE MRC 데이터셋 로드
dataset = load_dataset("klue", "mrc")

# 필터링할 news_category 목록
selected_categories = ['경제', '교육산업', '국제', '부동산', '사회', '생활', '책마을']

# 데이터 전처리 함수
def preprocess_klue_mrc(example, idx):
    return {
        'id': f"klue_mrc_{idx}",
        'paragraph': example['context'],
        'question': example['question'],
        'choices': [],  # 비워둠
        'answer': example['answers']['text'][0] if example['answers']['text'] else ""
    }

# 데이터 필터링 및 전처리
processed_data = []
idx = 0

for split in ['train', 'validation']:
    for example in dataset[split]:
        if example['news_category'] in selected_categories:
            processed_data.append(preprocess_klue_mrc(example, idx))
            idx += 1

# DataFrame으로 변환
df = pd.DataFrame(processed_data)

# CSV 파일로 저장
df.to_csv('klue_mrc_processed.csv', index=False)

print("처리된 데이터 샘플:")
print(df.head())
print(f"\n총 데이터 수: {len(df)}")