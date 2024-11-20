# test.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. 파인튜닝된 모델과 토크나이저 로드
model_name = "beaver-zip/gemma-ko-2b-qa-finetuned"  # 업로드한 모델의 경로
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 2. 테스트용 파이프라인 설정 (device 인수 제거)
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# 3. 테스트 질문과 맥락 설정
test_data = [
    {"question": "대한민국의 수도는 어디인가요?", "context": "대한민국의 수도는 서울입니다."},
    {"question": "아인슈타인의 상대성 이론이란 무엇인가요?", "context": "상대성 이론은 아인슈타인이 제안한 물리학 이론으로, 공간과 시간이 상대적인 개념임을 설명합니다."},
    {"question": "태양계에서 가장 큰 행성은 무엇인가요?", "context": "태양계에서 가장 큰 행성은 목성입니다."}
]

# 4. 모델 테스트 함수 정의
def test_model(qa_pipeline, test_data):
    for idx, data in enumerate(test_data):
        question = data["question"]
        context = data["context"]

        # 프롬프트 생성
        prompt = (
            f"<bos><start_of_turn>user\n"
            f"질문: {question}\n\n"
            f"맥락: {context}\n\n"
            f"답변:"
        )

        # 모델 응답 생성
        response = qa_pipeline(
            prompt,
            max_length=len(tokenizer.encode(prompt)) + 100,  # 충분한 길이 설정
            do_sample=True,
            temperature=0.7,      # 다양성 조절
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")  # 생성 중단 토큰 설정
        )

        # 응답을 <end_of_turn> 이전 텍스트로만 추출
        generated_text = response[0]['generated_text']
        answer_start = len(prompt)
        answer_end = generated_text.find("<end_of_turn>")
        if answer_end == -1:
            answer = generated_text[answer_start:].strip()
        else:
            answer = generated_text[answer_start:answer_end].strip()

        # 결과 출력
        print(f"테스트 {idx + 1}")
        print("질문:", question)
        print("맥락:", context)
        print("모델의 응답:", answer)
        print("=" * 50)

# 5. 모델 테스트 실행
if __name__ == "__main__":
    test_model(qa_pipeline, test_data)
