from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1. 데이터셋 불러오기
dataset = load_dataset("maywell/korean_textbooks", "mmlu_high_school_psychology")
dataset = dataset['train'].select(range(30)) # train이라고 해줘야함!

# 2. 일정 길이 이하인 행 제거 (30자 이하 제거)
def filter_short_examples(example):
    return len(example["0"]) > 30

filtered_dataset = dataset.filter(filter_short_examples)

# 3. Tokenizer 및 모델 로드
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"  # 사전 학습 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4. 데이터셋 토크나이징
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = filtered_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 5. 데이터 준비
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 6. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./pretrained_model",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
)

# 7. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 8. Pre-training 시작
trainer.train()

# 9. 모델 저장
trainer.save_model("./pretrained_model")
tokenizer.save_pretrained("./pretrained_model")
