import pandas as pd

original_df = pd.read_csv("/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test_known_hint_fix.csv")
updated_hints_df = pd.read_csv("/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test_known_139_fix.csv")

# `id` 열을 기준으로 두 데이터프레임 병합하여 `hint` 열 업데이트
original_df.set_index('id', inplace=True)  # 기존 데이터 인덱스를 `id`로 설정
updated_hints_df.set_index('id', inplace=True)  # 업데이트 데이터 인덱스를 `id`로 설정

# `hint` 열 업데이트
original_df.update(updated_hints_df[['hint']])

# 결과를 CSV 파일로 저장
original_df.reset_index(inplace=True)  # 인덱스 다시 초기화
original_df.to_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test_known_hint_final.csv', index=False)