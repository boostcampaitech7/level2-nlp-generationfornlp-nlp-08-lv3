import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging

# ===========================
# 1. 설정 변수
# ===========================

INPUT_FILE = '../data/need_knowledge.csv'
OUTPUT_FILE = 'train_hint.csv'
MODEL_NAME = 'beomi/Qwen2.5-7B-Instruct-kowiki-qa-context'
SPECIAL_TOKENS_FILE = 'special_tokens_map.json'
BATCH_SIZE = 1
MAX_NEW_TOKENS = 256

# ===========================
# 2. 로깅 설정
# ===========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===========================
# 3. 모델 및 토크나이저 로드
# ===========================

def load_special_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    for token in ['eos_token', 'pad_token']:
        if token in tokens and isinstance(tokens[token], dict):
            tokens[token] = tokens[token].get('content', '')
    return tokens

# Load special tokens
logger.info("Loading tokenizer and model...")
special_tokens = load_special_tokens(SPECIAL_TOKENS_FILE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, padding_side='left')
tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.info(f"Model loaded to {device}.")

# ===========================
# 4. 공통 함수: 모델 예측 실행 (배치 처리)
# ===========================

def generate_background_info_batch(prompts):
    """
    Generate background information for prompts in batch.
    """
    system_prompt = "당신은 학생들이 지문의 배경 지식을 이해하도록 돕는 AI 튜터입니다."

    full_prompts = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for prompt in prompts
    ]

    inputs = tokenizer(
        full_prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1
        )

    responses = [
        tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
        for i in range(len(prompts))
    ]

    return responses

# ===========================
# 5. 데이터 처리 및 파일 생성
# ===========================

def main():
    logger.info("Reading input CSV file...")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')

    def parse_question(problems_str):
        try:
            problems_dict = json.loads(problems_str.replace("'", '"'))
            return problems_dict.get('question', '')
        except json.JSONDecodeError:
            return ''

    df['question'] = df['problems'].astype(str).apply(parse_question)

    def construct_user_prompt(row):
        paragraph = str(row.get('paragraph', '')).strip()
        question_plus = str(row.get('question_plus', '')).strip()
        question = row.get('question', '')

        if question_plus:
            return (
                f"아래 지문과 질문을 참고하여, 학생들이 질문을 해결하기 위한 배경 지식을 제공해주세요.\n\n"
                f"지문:\n{paragraph}\n\n<보기>\n{question_plus}\n\n질문:\n{question}"
            )
        else:
            return (
                f"아래 지문과 질문을 참고하여, 학생들이 질문을 해결하기 위한 배경 지식을 제공해주세요.\n\n"
                f"지문:\n{paragraph}\n\n질문:\n{question}"
            )

    df['user_prompt'] = df.apply(construct_user_prompt, axis=1)

    logger.info(f"Processing {len(df)} rows in batches...")
    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing batches"):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()
        prompts = batch_df['user_prompt'].tolist()

        try:
            hints = generate_background_info_batch(prompts)
        except Exception as e:
            logger.error(f"Error processing batch {start_idx}-{end_idx}: {e}")
            hints = [''] * len(prompts)

        df.loc[start_idx:end_idx - 1, 'hint'] = hints
        logger.info(f"Processed batch {start_idx}-{end_idx - 1}.")

    logger.info("Dropping intermediate columns...")
    df.drop(columns=['user_prompt', 'question'], inplace=True)

    logger.info(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, encoding='utf-8', index=False)
    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
