import csv
import ast
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42) # magic number :)

# CSV 파일 읽기 및 처리
def process_csv(file_path):
    df = pd.read_csv(file_path)
    df['hint'] = ""  # 새로운 'hint' 열 추가
    
    # 고정된 힌트 메시지
    fixed_hint = "이 문제의 답변은 주어진 지문 안에 있습니다."
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        df.at[index, 'hint'] = fixed_hint

    return df

# CSV 파일 처리 및 결과 저장
input_file = './raw_data/klue_mrc.csv'
output_file = './hint/klue_mrc_add_hint.csv'

processed_df = process_csv(input_file)
processed_df.to_csv(output_file, index=False)

print(f"처리가 완료되었습니다. 결과가 {output_file}에 저장되었습니다.")