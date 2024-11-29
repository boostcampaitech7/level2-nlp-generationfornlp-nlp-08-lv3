import pandas as pd

# 파일 불러오기
first_file = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test_known_hint_final.csv')
second_file = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/ensemble/output_6521.csv')
third_file = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/ensemble/final_predictions.csv')

# 첫 번째 파일에서 id 열만 추출
first_file_ids = first_file[['id']]

# 두 번째 파일의 id 값에 해당하는 행을 찾아서 첫 번째 파일의 answer로 덮어쓰기
second_file_updated = second_file.copy()
third_file_updated = third_file.copy()
for idx, row in second_file.iterrows():
    id_value = row['id']
    # 첫 번째 파일에서 해당 id의 값이 있으면 덮어쓰기
    if id_value in first_file_ids['id'].values:
        matched_row = first_file[first_file['id'] == id_value]
        second_file_updated.loc[idx, 'answer'] =third_file_updated['answer'].values[0]



# 결과 저장
second_file_updated.to_csv('updated.csv', index=False)
third_file_updated.to_csv('third_file_updated.csv', index=False)

print("파일 덮어쓰기가 완료되었습니다.")
