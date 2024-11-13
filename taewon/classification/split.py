# split.py

import pandas as pd
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('train.csv')

split_1 = int(len(df) * 1/3)
split_2 = int(len(df) * 2/3)

df1 = df[:split_1]
df2 = df[split_1:split_2]
df3 = df[split_2:]

# 분할된 데이터 저장
df1.to_csv('train1.csv', index=False)
df2.to_csv('train2.csv', index=False)
df3.to_csv('train3.csv', index=False)

print("train.csv가 train1.csv, train2.csv, train3.csv로 분할되었습니다.")
print(f"train1.csv: {len(df1)} 행")
print(f"train2.csv: {len(df2)} 행")
print(f"train3.csv: {len(df3)} 행")