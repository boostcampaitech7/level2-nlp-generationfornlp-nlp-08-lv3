# 라이브러리 임포트
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import numpy as np
import random
import evaluate
import bitsandbytes as bnb

# Hugging Face Hub 로그인
hf_token = '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'  # 실제 토큰으로 대체하세요
login(token=hf_token)

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # 멀티 GPU 사용 시
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42)

# KorQuAD 데이터셋 로드
korquad = load_dataset('squad_kor_v1')

# 데이터 전처리
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    answers = examples["answers"]

    inputs = []
    targets = []

    for question, context, answer in zip(questions, contexts, answers):
        input_text = f"지문:\n{context}\n\n질문:\n{question}\n\n답변을 작성하세요:"
        target_text = answer['text'][0] if len(answer['text']) > 0 else ''
        inputs.append(input_text)
        targets.append(target_text)

    return {"inputs": inputs, "targets": targets}

tokenized_dataset = korquad.map(
    preprocess_function,
    batched=True,
    remove_columns=korquad["train"].column_names,
)

# 모델 및 토크나이저 로드
model_name = "beomi/gemma-ko-2b"

# 4비트 양자화된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    quantization_config=bnb.QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

# LoRA 설정 및 적용
peft_config = LoraConfig(
    r=8,  # Rank를 8로 설정하여 학습 가능한 파라미터 수를 적절하게 유지
    lora_alpha=16,  # 스케일링 팩터를 16으로 설정하여 학습 안정성 향상
    lora_dropout=0.1,  # 드롭아웃을 0.1로 설정하여 과적합 방지
    target_modules=['q_proj', 'v_proj'],  # LoRA를 적용할 모듈 지정
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# 토큰화
max_seq_length = 1024

def tokenize_function(examples):
    full_texts = [inp + "\n" + tgt for inp, tgt in zip(examples["inputs"], examples["targets"])]
    tokenized = tokenizer(
        full_texts,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )
    labels = tokenized["input_ids"].copy()
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }

tokenized_dataset = tokenized_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["inputs", "targets"],
)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

# 데이터셋 분할
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# 평가 메트릭 설정
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)

    mask = labels != tokenizer.pad_token_id
    labels = labels[mask]
    predictions = predictions[mask]

    return accuracy.compute(predictions=predictions, references=labels)

# 훈련 인자 설정
output_dir = "gemma-ko-2b-korquad-finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

# 트레이너 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 학습 시작
trainer.train()

# 모델 저장 및 Hugging Face에 업로드
trainer.save_model(output_dir)

# 모델 업로드
repo_id = 'beaver-zip/gemma-ko-2b-korquad'

model.push_to_hub(repo_id, use_auth_token=True)
tokenizer.push_to_hub(repo_id, use_auth_token=True)