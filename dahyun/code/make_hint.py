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

# 사전지식 생성 함수
def generate_additional_info(paragraph, question, choices):
    prompt = f"""예시를 참고하여 아래의 지문, 선택지를 바탕으로 질문에 답하기에 부족한 사전지식을 힌트로써 한국어로만 반드시 생성하세요. 선택지와 관련된 주요 정보만 제공하면 됩니다.


    ### 예시 ###
    지문: 
    다국적 기업은 운영을 분산하여...

    질문: 
    다국적 기업의 운영 방식에 대한 설명으로 옳은 것은?

    선택지: ['생산비용이 가장 높은 제품을 제조한다', '경제적인 곳에서 회계 및 연구 서비스를 수행한다', '비교우위를 바탕으로 한다', '본사는 저개발국에 소재한다']

    힌트: 지문은 다국적 기업의 운영 방식을 언급하며, 관련된 지리적 맥락으로는 지역적 특성이 중요한 역할을 합니다. 예를 들어, 다국적 기업은 비용 절감을 위해 인건비가 저렴한 아시아 국가나 생산 인프라가 좋은 국가에 공장을 두는 경향이 있습니다. 또한, 기후와 자원 분포도 고려되어, 특정 지역에서 원자재를 적극적으로 활용할 수 있습니다. 인구 밀도가 높은 도시 지역은 소비 시장으로서 가치가 커지므로, 마케팅 전략 수립에 중요한 요소로 작용합니다. 전 세계적으로 물류와 통신의 발달로 인해 기업의 글로벌 운영이 용이해졌으며, 이는 환경 문제나 규제에 대한 대응을 요구하기도 합니다."

    ### 힌트를 생성해야 될 문제 ###
    지문:
    {paragraph}
    
    질문:
    {question}

    선택지:
    {choices}
    
    예시의 힌트 형태처럼 힌트를 300토큰 내로 생성하세요.
    힌트:"""

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
    생성된 텍스트에서 '힌트:' 이후 내용만 추출
    """
    match = re.search(r"힌트:\s*(.*)", generated_text, re.DOTALL)  # '힌트:' 이후 텍스트만 추출
    if match:
        return match.group(1).strip()  # 힌트 내용만 반환
    return "Hint not found"

# 데이터 처리
test_df = pd.read_csv("/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test_known.csv")  # 데이터 파일 경로
test_df = test_df[:4]
batch_size = 2  # 배치 크기
hints = []

for i in tqdm(range(0, len(test_df), batch_size), desc="Processing"):
    batch = test_df.iloc[i : i + batch_size]
    paragraphs = batch["paragraph"].tolist()
    questions = batch["question"].tolist()
    choices_list = batch["choices"].tolist()

    # 힌트 생성 및 저장
    batch_hints = [generate_additional_info(p, q, c) for p, q, c in zip(paragraphs, questions, choices_list)]
    hints.extend(batch_hints)
    
    # 생성된 내용 출력
    print(f"Batch {i // batch_size + 1}:")
    for p, q, c, hint in zip(paragraphs, questions, choices_list, batch_hints):
        print(f"Paragraph: {p}")
        print(f"Question: {q}")
        print(f"Choices: {c}")
        print(f"Hint: {hint}")
        print("=" * 80)

# 힌트 저장
test_df["hint"] = hints
test_df.to_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/test_known_hint.csv', index=False)  # 결과 파일 저장
