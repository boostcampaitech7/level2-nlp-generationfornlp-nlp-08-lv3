# "1d5e0826-8d37-446d-894a-68e61cd5f8a6:fx"
import os
import json
import requests

# DeepL API 설정
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_API_KEY = "1d5e0826-8d37-446d-894a-68e61cd5f8a6:fx"  # 여기에 DeepL API 키를 입력하세요.

# 경로 설정
BASE_DIR = "/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/sungeun/2_GAOKAO"
INPUT_FILE = os.path.join(BASE_DIR, "Objective_Questions/2010-2013_English_MCQs.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "translated.json")

def translate_text(text, source_lang="EN", target_lang="KO"):
    """
    DeepL API를 사용하여 텍스트를 번역합니다.
    """
    try:
        params = {
            "auth_key": DEEPL_API_KEY,
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        response = requests.post(DEEPL_API_URL, data=params)
        response.raise_for_status()  # 요청 실패 시 예외 발생
        translated_text = response.json()["translations"][0]["text"]
        return translated_text
    except Exception as e:
        print(f"번역 오류: {e}")
        return text  # 오류 발생 시 원본 텍스트 반환

def translate_json(input_path, output_path):
    """
    JSON 파일을 읽고, 'question' 및 'analysis' 필드를 번역한 후 새 JSON 파일로 저장합니다.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        
        # 번역 작업
        for item in data.get("example", []):
            if "question" in item:
                print(f"Translating 'question' at index {item['index']}...")
                item["question"] = translate_text(item["question"], source_lang="EN", target_lang="KO")
            if "analysis" in item:
                print(f"Translating 'analysis' at index {item['index']}...")
                item["analysis"] = translate_text(item["analysis"], source_lang="EN", target_lang="KO")
        
        # 번역된 결과 저장
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        
        print(f"Translation completed. Translated file saved to: {output_path}")
    except FileNotFoundError:
        print(f"입력 파일을 찾을 수 없습니다: {input_path}")
    except json.JSONDecodeError:
        print("입력 파일이 올바른 JSON 형식이 아닙니다.")
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")

if __name__ == "__main__":
    translate_json(INPUT_FILE, OUTPUT_FILE)
