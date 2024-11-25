import json
import random
import numpy as np
import torch
import os
from utils import get_peft_config
from data_processing import (
    load_and_process_data,
    #concat_question_and_question_plus,
    process_dataset_with_prompts,
    process_and_tokenize_dataset,
    filter_and_split_dataset, 
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import evaluate  # 메트릭 평가 라이브러리
#import bitsandbytes as bnb

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class RingAttention(nn.Module):
    def __init__(self, embed_size, ring_radius):
        super(RingAttention, self).__init__()
        self.embed_size = embed_size
        self.ring_radius = ring_radius
        
        # Query, Key, Value의 linear transformation
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Attention scores 계산: Q와 K의 내적을 구한 후,
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)

        # Ring 범위 내의 토큰만 고려하기 위한 mask 생성
        mask = torch.zeros_like(attention_scores)

        for i in range(seq_len):
            # 해당 위치에서 반경 내의 인덱스를 선택
            start_idx = max(0, i - self.ring_radius)
            end_idx = min(seq_len, i + self.ring_radius + 1)
            mask[:, i, start_idx:end_idx] = 1  # 범위 내의 값에만 1을 부여

        # mask를 attention scores에 적용
        attention_scores = attention_scores * mask - 1e9 * (1 - mask)  # 마스크된 부분은 매우 작은 값으로 설정

        # Softmax를 적용하여 Attention weight 계산
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Attention output 계산
        output = torch.matmul(attention_weights, V)
        output = self.out(output)

        return output

class RingGPT2Attention(GPT2Attention):
    def __init__(self, config, ring_radius=3):
        super().__init__(config)
        self.ring_attention = RingAttention(config.hidden_size, ring_radius)
        
    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        # 기존 방식대로 forward() 호출을 이어가고,
        # 대신 RingAttention을 호출하여 계산합니다.
        return self.ring_attention(hidden_states, hidden_states, hidden_states)

class RingQwenModel(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # GPT2의 Attention을 RingAttention으로 교체
        self.transformer.h[0].attn = RingGPT2Attention(config)  # 첫 번째 레이어만 교체 (전체 레이어를 바꾸려면 반복문 사용)


# Config 파일 로드
with open("config.json", "r") as f:
    config = json.load(f)

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42) # magic number :)

# 데이터 로드 및 전처리
train_data = load_and_process_data(config["train_data_path"]) # DataFrame으로 변환
#train_data = concat_question_and_question_plus(train_data) 

# 프롬프트 적용 데이터셋 생성
processed_train_data = process_dataset_with_prompts(train_data)

# 첫 번째 데이터셋의 내용을 출력하는 코드
first_sample = processed_train_data[0]
print("=== First Sample ===")
for message in first_sample["messages"]:
    print(f"Role: {message['role']}, Content: {message['content']}")


# # 4bit 양자화 설정
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     config["model_name"], 
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
#     device_map="auto",  # 양자화 지원 장치에 자동 매핑
#     quantization_config=bnb_config
# )
# tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True,)

model = RingQwenModel.from_pretrained(config["model_name"], torch_dtype=torch.float16, trust_remote_code=True,)
tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True,)

# model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.float16, trust_remote_code=True,)
# tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True,)

# 채팅 템플릿 설정
#tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + content + '<end_of_turn>\\n<start_of_turn>model\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\\n' }}{% endif %}{% endfor %}"
#tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 데이터셋 토큰화
tokenized_train_data = process_and_tokenize_dataset(processed_train_data, tokenizer)

# 첫 번째 토큰화된 데이터셋을 디코딩해서 출력하는 코드
print("\n=== Decoded First Tokenized Sample ===")
first_tokenized_sample = tokenized_train_data[0]
decoded_input_ids = tokenizer.decode(first_tokenized_sample["input_ids"], skip_special_tokens=False)
print(decoded_input_ids)

max_length = config['max_length']
# 데이터 분리 및 필터링
train_dataset, eval_dataset = filter_and_split_dataset(tokenized_train_data, max_length=max_length, test_size=0.1, seed=config["random_seed"])

print("=== Train Dataset Sample ===")
print(train_dataset[0])

print("\n=== Evaluation Dataset Sample ===")
print(eval_dataset[0])

# Completion 부분만 학습하기 위한 Data Collator 설정
response_template = "<|im_start|>assistant"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# pad token 설정
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

# config 파일의 sft_config 파라미터 사용하여 SFTConfig 초기화
sft_config = SFTConfig(**config["sft_config"])

### Metric 설정
def preprocess_logits_for_metrics(logits, labels):
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"], tokenizer.vocab["5"]]
    logits = logits[:, -2, logit_idx]
    logits = logits.float()
    return logits

# metric 로드
acc_metric = evaluate.load("accuracy")

# 정답 토큰 매핑
int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

# metric 계산 함수
def compute_metrics(evaluation_result):
    logits, labels = evaluation_result
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # labels = list(map(lambda x: x.split("<|im_end|>")[0].strip(), labels))
    # labels = list(map(lambda x: int_output_map[x], labels))
    
    # 응답에서 마지막 <|im_end|> 기준으로 자르기
    labels = list(map(lambda x: x.rsplit("<|im_end|>", 1)[0].strip(), labels))  
    # 디코딩한 응답을 정수형 매핑
    labels = list(map(lambda x: int_output_map.get(x, -1), labels))  # KeyError 방지

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc

# LoRA 설정 가져오기
peft_config = get_peft_config()

### Training history storage ###
training_history = {
    "Epoch": [],
    "Training Loss": [],
    "Validation Loss": [],
    "Accuracy": []
}

### Callback to store metrics after each epoch ###
class SaveMetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_loss = None  # 학습 손실을 저장할 변수

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 학습 중간에 로깅되는 값에서 'loss' 추출
        if logs is not None and "loss" in logs:
            self.training_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Store metrics after every evaluation step (i.e., after each epoch)
        epoch = int(state.epoch) if state.epoch is not None else "N/A"
        validation_loss = metrics.get("eval_loss", None)
        accuracy = metrics.get("eval_accuracy", None)

        # Store the metrics in the training_history dictionary
        training_history["Epoch"].append(epoch)
        training_history["Training Loss"].append(self.training_loss)
        training_history["Validation Loss"].append(validation_loss)
        training_history["Accuracy"].append(accuracy)

        # Create the formatted output for each epoch
        result_str = f"Epoch : {epoch},\n"
        result_str += f"Training Loss : {self.training_loss:.6f},\n"
        result_str += f"Validation Loss : {validation_loss:.6f},\n"
        result_str += f"Accuracy : {accuracy:.6f}\n"
        result_str += "#" * 25 + "\n"

        # Save the formatted output to a file
        output_file_path = os.path.join(config["output_dir"], "training_results.txt")
        with open(output_file_path, "a") as f:  # 'a' mode to append each epoch's results
            f.write(result_str)

# SFTTrainer 정의 및 학습 시작
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    peft_config=peft_config,
    args=sft_config,
    callbacks=[SaveMetricsCallback()],
)

# 체크포인트에서 이어서 학습
checkpoint_path = config.get("checkpoint_path")
if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    torch.cuda.empty_cache()
    trainer.train()  # 처음부터 학습 시작