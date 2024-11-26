import os
import pandas as pd

# 디렉토리 경로 설정
directory_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/aug_hint'

# 디렉토리 내 모든 CSV 파일을 순차적으로 처리
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        # 파일 경로 생성
        file_path = os.path.join(directory_path, filename)

        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 'hint' 열에서 "힌트: " 삭제
        if 'hint' in df.columns:
            df['hint'] = df['hint'].str.replace('힌트: ', '', regex=False)

        # 수정된 DataFrame을 다시 CSV로 저장 (덮어쓰기)
        df.to_csv(file_path, index=False)

        print(f"Processed {filename}")
