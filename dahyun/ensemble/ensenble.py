import pandas as pd
import os
from collections import Counter

# 폴더 내 CSV 파일 경로
folder_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/ensemble'

# CSV 파일 리스트 가져오기
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 모델 예측 결과들을 저장할 리스트
predictions = []
ids = []  # id들을 저장할 리스트

# 각 CSV 파일에서 데이터를 불러와서 예측 결과를 predictions 리스트에 저장
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    print(f"file path: {file_path}")
    
    # 첫 번째 파일에서 id 값을 저장 (모든 파일에 동일한 id 열이 있다고 가정)
    if not ids:
        ids = df['id'].values.tolist()
    
    # 모델의 예측 결과가 'answer' 열에 있다고 가정
    predictions.append(df['answer'].values)

# 각 데이터에 대해 다수결을 적용해 최종 예측 값 계산
final_predictions = []
for i in range(len(predictions[0])):  # 첫 번째 파일의 예측 개수로 반복
    # 각 모델의 예측값을 모은 리스트
    vote = [pred[i] for pred in predictions]
    
    # 다수결로 최종 예측 값 결정
    final_prediction = Counter(vote).most_common(1)[0][0]
    final_predictions.append(final_prediction)

# 결과 DataFrame에 id와 final_answer 열을 추가
final_df = pd.DataFrame({'id': ids, 'answer': final_predictions})

# 최종 예측 값을 CSV로 저장
final_df.to_csv('final_predictions.csv', index=False)