from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import re

# 모델 및 토크나이저 설정
model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # FP16으로 메모리 최적화
    device_map="auto"          # GPU에 자동으로 매핑
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# 특수 토큰 설정 확인 및 수정
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id + 1
    model.config.pad_token_id = tokenizer.pad_token_id

print("Pad Token ID:", tokenizer.pad_token_id)
print("EOS Token ID:", tokenizer.eos_token_id)

# 입력 길이 제한
MAX_INPUT_LENGTH = 4096  # Llama 기반 모델은 긴 입력을 처리 가능

# 프롬프트 템플릿 생성 함수
def apply_chat_template(messages, add_generation_prompt=True):
    """
    메시지를 'role'과 'content' 형태로 받아 프롬프트 템플릿 생성.
    """
    prompt = ""
    for msg in messages:
        prompt += f"{msg['role']}: {msg['content']}\n\n"
    if add_generation_prompt:
        prompt += "assistant: "
    return prompt.strip()

# 사전지식 생성 함수
def generate_additional_info(paragraph, question, choices):
    PROMPT = "당신은 유능한 AI 문제풀이 어시스턴트입니다. 사용자의 요구에 맞게 힌트를 생성하세요."

    # 메시지 템플릿 작성
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f""" 문제의 지문, 질문, 선택지의 내용을 바탕으로 문제풀이에 도움이 되는 사전지식을 힌트로써 한국어로 생성하세요.
        사전지식은 간단명료하게 요약하여 핵심 내용만 생성하세요 문단으로 작성하세요. 
        불필요한 세부사항이나 반복된 내용을 포함하지 마세요.
        
        ### 힌트를 생성해야 될 문제 ###
        지문: {paragraph}

        질문: {question}

        선택지: {choices}

        한국어 줄글 문장으로 끊기는 문장 없게 요약하여 200자 이내의 문단으로 힌트만 생성하세요."""}
    ]

    # 프롬프트 생성
    prompt = apply_chat_template(messages)

    # 입력 데이터 생성 및 모델 디바이스로 이동
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH, padding=True)
    device = model.device  # 모델 디바이스 확인
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 입력 데이터를 모델 디바이스로 이동

    try:
        # 텍스트 생성
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=300,  # 생성할 텍스트 길이 제한
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return parse_hint(generated_text)  # '힌트:' 이후만 반환
    except Exception as e:
        print("Error during generation:", str(e))
        return "Generation Failed"

def parse_hint(generated_text):
    """
    생성된 텍스트에서 'assistant:' 다음 모든 내용을 추출
    """
    match = re.search(r"assistant:\s*(.*)", generated_text, re.DOTALL)  # assistant 이후 모든 텍스트 추출
    if match:
        return match.group(1).strip()  # 추출한 텍스트 반환
    return "Hint not found"

# 데이터 처리
test_df = pd.read_csv("/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test_known_hint_fix.csv")  # 데이터 파일 경로
# 'Hint not found' 행만 슬라이싱하고 복사본 생성
hint_not_found_df = test_df[test_df["hint"] == "Hint not found"].copy()



batch_size = 1  # 배치 크기
new_hints = []  # 새로운 힌트를 저장할 리스트

# 힌트 생성
for i in tqdm(range(0, len(hint_not_found_df), batch_size), desc="Processing"):
    batch = hint_not_found_df.iloc[i : i + batch_size]
    paragraphs = batch["paragraph"].tolist()
    questions = batch["question"].tolist()
    choices_list = batch["choices"].tolist()

    # 힌트 생성
    batch_hints = [generate_additional_info(p, q, c) for p, q, c in zip(paragraphs, questions, choices_list)]
    new_hints.extend(batch_hints)

    # 생성된 내용 출력
    print(f"Batch {i // batch_size + 1}:")
    for p, q, c, hint in zip(paragraphs, questions, choices_list, batch_hints):
        print(f"Paragraph: {p}")
        print(f"Question: {q}")
        print(f"Choices: {c}")
        print(f"Hint: {hint}")
        print("=" * 80)

# 힌트 저장
hint_not_found_df["hint"] = new_hints
hint_not_found_df.to_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test_known_139.csv', index=False)  # 결과 파일 저장
