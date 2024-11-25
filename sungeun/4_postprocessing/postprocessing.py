# 1. test을 랜덤화하고
#     1-(1) output_odqa 분리한다. (435)
#     1-(2) output_known 분리한다. (434)

# import pandas as pd

# data = pd.read_csv("test.csv")
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# # 데이터 분리
# output_odqa = data.iloc[:435]  # 상위 435개
# output_known = data.iloc[435:869]  # 그 다음 434개

# # 결과 저장 (옵션)
# output_odqa.to_csv("output_odqa.csv", index=False, encoding = "utf-8-sig")
# output_known.to_csv("output_known.csv", index=False, encoding = "utf-8-sig")

# 2. 기존 test 버전에 맞추는지 확인
import pandas as pd
test_df = pd.read_csv("test.csv")
odqa_df = pd.read_csv("output_odqa.csv")
known_df = pd.read_csv("output_known.csv")

merged_df = pd.merge(test_df, odqa_df, on='id', how='left')
merged_df = pd.merge(merged_df, known_df, on='id', how='left')

merged_df.to_csv("try1.csv", index=False, encoding = "utf-8-sig")

# 3. 찐막 확인
# import pandas as pd
# try1 = pd.read_csv("try1.csv")
# test = pd.read_csv("test.csv")

# if 'id' in try1.columns and 'id' in test.columns:
#     comparison = try1['id'].eq(test['id']).astype(int).replace(1, 0).replace(0, -1)
#     try1['comparison'] = comparison
# else:
#     print("Error: Both DataFrames must have an 'id' column.")

# try1.to_csv("try1_comparison.csv", index=False)
# print("Comparison result saved to 'try1_comparison.csv'")

# 4. 한놈이라도 -1 인 나왔다면???