import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
import random
import numpy as np

# ===========================
# 설정 변수
# ===========================

INPUT_FILE = '../data/test_known.csv'
OUTPUT_FILE = 'test_hint.csv'
MODEL_NAME = 'beomi/Llama-3-Open-Ko-8B'
BATCH_SIZE = 1
MAX_NEW_TOKENS = 200
CHECKPOINT_INTERVAL = 2  # n개 데이터 처리 후 자동저장

# ===========================
# 로깅 설정
# ===========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===========================
# 시드 고정
# ===========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ===========================
# 모델 및 토크나이저 로드
# ===========================

logger.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, padding_side='left')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("pad_token was not found in tokenizer. Set pad_token to eos_token.")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.info(f"Model loaded to {device}.")

# ===========================
# 모델 예측 실행
# ===========================

def generate_background_info_batch(prompts):
    """
    Generate background information for prompts in batch.
    """
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,          # 샘플링 비활성화
            num_beams=3,              # 빔 서치 사용
            no_repeat_ngram_size=3,   # 반복 방지
            early_stopping=True
        )

    generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
    responses = [
        tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip()
        for i in range(len(prompts))
    ]

    # 글자 수 제한 없이 모델 출력에 의존 (200자 이내의 한 문장으로 유도)
    return responses

# ===========================
# 데이터 처리 및 파일 생성
# ===========================

def main():
    logger.info("Reading input CSV file...")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')

    def extract_question(problems_str):
        """
        Extract question from the 'problems' JSON string.
        """
        try:
            problems_dict = json.loads(problems_str.replace("'", '"'))
            return problems_dict.get('question', '').strip()
        except (json.JSONDecodeError, AttributeError):
            return ''

    def construct_user_prompt(row):
        """
        Construct prompt for the model using paragraph, question, and question_plus.
        """
        paragraph = str(row.get('paragraph', '')).strip()
        question_plus = str(row.get('question_plus', '')).strip()
        question = extract_question(row.get('problems', ''))

        prompt = (
            "당신은 학생들의 문제 해결을 돕는 뛰어난 AI 튜터입니다. "
            "다음 지문과 질문을 읽고, 질문에 답하기 위해 필요한 핵심적인 배경 지식을 "
            "200자 이내의 한 문장으로 작성해주세요. "
            "배경 지식 외의 다른 내용은 생성하지 마세요.\n\n"
            "<지문>\n"
            f"{paragraph}\n\n"
        )

        if question_plus and question_plus.lower() != 'nan':
            prompt += f"<보기>\n{question_plus}\n\n"

        prompt += f"<질문>\n{question}\n\n"

        # 모델이 한 문장으로 응답하도록 유도
        return prompt

    df['user_prompt'] = df.apply(construct_user_prompt, axis=1)

    logger.info(f"Processing {len(df)} rows in batches...")
    hints = []

    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing batches"):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        prompts = batch_df['user_prompt'].tolist()

        try:
            batch_hints = generate_background_info_batch(prompts)
        except Exception as e:
            logger.error(f"Error processing batch {start_idx}-{end_idx}: {e}")
            batch_hints = [''] * len(prompts)

        hints.extend(batch_hints)

        # 중간 저장: n개의 데이터를 처리할 때마다 저장
        if (len(hints) % CHECKPOINT_INTERVAL == 0) or (end_idx == len(df)):
            df.loc[:len(hints) - 1, 'hint'] = hints
            df[['id', 'paragraph', 'problems', 'question_plus', 'hint']].to_csv(
                OUTPUT_FILE, encoding='utf-8', index=False
            )
            logger.info(f"Checkpoint saved at row {len(hints)}.")

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
