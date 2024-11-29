import pandas as pd

# CSV 파일 로드
# CSV 파일 로드
df1 = pd.read_csv('../data/aug_train.csv')
df2 = pd.read_csv('../data/new_augmented_questions.csv')

new_train3 = pd.concat([df1, df2])

# 결과 저장
new_train3_cleaned286 = new_train3[new_train3['paragraph'].apply(len) > 286]
new_train3_cleaned286.to_csv('../data/new_train3_cleaned286.csv')