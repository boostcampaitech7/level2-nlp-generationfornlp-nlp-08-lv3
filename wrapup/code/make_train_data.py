import pandas as pd

# CSV 파일 로드
df1 = pd.read_csv('../data/aug_train.csv')
df2 = pd.read_csv('../data/new_augmented_questions.csv')

# 데이터 병합 및 저장
new_train3 = pd.concat([df1, df2])
new_train3.to_csv("../data/new_train3.csv", index=False)  # 병합된 데이터 저장

# 저장된 파일 다시 로드
new_train3 = pd.read_csv("../data/new_train3.csv")

# 조건에 맞는 데이터 필터링 및 저장
new_train3_cleaned286 = new_train3[new_train3['paragraph'].apply(len) > 286]
new_train3_cleaned286.to_csv('../data/new_train3_cleaned286.csv')
