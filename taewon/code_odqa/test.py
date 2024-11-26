import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from data_processing import load_and_process_test_data, format_test_data_for_model# 데이터 로드 및 전처리 함수
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import random

# 시드 고정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 고정
set_seed(42)

# Config 파일 로드
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

checkpoint_path = config["checkpoint_path"]

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoPeftModelForCausalLM.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    # torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
)

# 테스트 데이터 로드 및 전처리
test_df = load_and_process_test_data(config["test_data_path"])
test_dataset = format_test_data_for_model(test_df)

# 추론 및 결과 저장
infer_results = []
pred_choices_map = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    #5: lambda: random.choice(["1", "2", "3", "4", "5"])
}

model.eval()
with torch.inference_mode():
    for data in tqdm(test_dataset):
        _id = data["id"]
        messages = data["messages"]
        len_choices = data["len_choices"]

        outputs = model(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
        )

        logits = outputs.logits[:, -1].flatten().cpu()
        target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(target_logit_list, dtype=torch.float32)
            )
            .detach()
            .cpu()
            .numpy()
        )

        predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
        infer_results.append({"id": _id, "answer": predict_value})
        
'''        predict_idx = np.argmax(probs, axis=-1)
        predict_value = pred_choices_map[predict_idx]
        
        # lambda 함수일 경우 호출하여 랜덤 선택
        if callable(predict_value):
            predict_value = predict_value()
        
        infer_results.append({"id": _id, "answer": predict_value})'''

# CSV 파일로 결과 저장
output_file_path = os.path.join(config["output_dir"], "output_odqa.csv")
pd.DataFrame(infer_results).to_csv(output_file_path, index=False)