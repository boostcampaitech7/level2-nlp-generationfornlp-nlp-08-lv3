import pandas as pd

# 파일 경로 설정
GPT_file_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/GPT_Answers.csv'
answer_file_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/output_5829.csv'

# 데이터 로드
true_data = pd.read_csv(GPT_file_path)
answer_data = pd.read_csv(answer_file_path)

# true_answer와 answer 컬럼 설정
true_data['true_answer'] = true_data['nlp_answer']  # 실제 정답을 true_answer로 설정
answer_data['answer'] = answer_data['nlp_answer']  # 예측 정답을 answer로 설정 (예시로 nlp_answer 사용)

# NLP 기반 예측 정답과 실제 정답 비교를 위한 데이터 병합
merged_data = pd.merge(
    true_data[['id', 'true_answer']], 
    answer_data[['id', 'answer']], 
    on='id'
)

# 정확도 계산
correct_predictions = (merged_data['true_answer'] == merged_data['answer']).sum()
total_predictions = len(merged_data)
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

# 결과 출력
print(f"Accuracy: {accuracy:.2f}")
