import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from data_processing import load_and_process_data, format_test_data_for_model  # 데이터 로드 및 전처리 함수
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Config 파일 로드
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

checkpoint_path = config["checkpoint_path"]

# 모델 및 토크나이저 로드 시 low_cpu_mem_usage 적용
model = AutoPeftModelForCausalLM.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,  # V100 32GB에 적합한 dtype 설정
    low_cpu_mem_usage=True  # CPU 메모리 최적화
)
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True  # CPU 메모리 최적화
)

# 테스트 데이터 로드 및 전처리
test_df = load_and_process_data(config["test_data_path"])
test_dataset = format_test_data_for_model(test_df)

# 추론 및 결과 저장
infer_results = []
pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

model.eval()
with torch.inference_mode():
    for data in tqdm(test_dataset):
        _id = data["id"]
        messages = data["messages"]
        len_choices = data["len_choices"]

        # 템플릿 적용 및 토큰화
        # tokenizer.apply_chat_template이 Tensor를 반환하는 경우, 이를 dict 형태로 변환
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # 'input_ids'가 반환되지 않는다면, 'input_ids' 키를 추가하여 dict 형태로 만듦
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.to("cuda")
            attention_mask = torch.ones_like(input_ids).to("cuda")  # 모든 토큰에 대해 마스크 설정
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        elif isinstance(inputs, dict):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        else:
            raise ValueError("Tokenizer output is neither Tensor nor dict.")

        # 모델에 입력 전달
        outputs = model(**inputs)

        # logits 추출 및 정답 예측
        logits = outputs.logits[:, -1, :]  # 마지막 토큰의 logits
        # '1', '2', '3', '4', '5' 토큰의 인덱스를 가져옴
        target_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, 6)]
        target_logits = logits[:, target_token_ids].squeeze(0)  # shape: (5,)

        # 확률 계산
        probs = torch.nn.functional.softmax(target_logits, dim=-1).cpu().numpy()

        # 정답 선택
        predict_idx = np.argmax(probs)
        predict_value = pred_choices_map.get(predict_idx, "1")  # 기본값 '1' 설정

        infer_results.append({"id": _id, "answer": predict_value})

# CSV 파일로 결과 저장
output_file_path = os.path.join(config["output_dir"], "output.csv")
pd.DataFrame(infer_results).to_csv(output_file_path, index=False)
