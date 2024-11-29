import json
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset

def load_and_process_data(data_path):
    # 데이터셋 로드
    dataset = pd.read_csv(data_path)
        
    # JSON 형식의 데이터를 펼쳐서 새로운 DataFrame 생성
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        problems['choices'].append('정답 없음') # 추가
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            'question_plus': problems.get('question_plus', None),
        }
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        records.append(record)
        
    # DataFrame으로 변환
    df = pd.DataFrame(records)

    return df


def concat_question_and_question_plus(df):
    # 'question'과 'question_plus' 컬럼 결합
    df['question_plus'] = df['question_plus'].fillna('')
    df['full_question'] = df.apply(lambda x: x['question'] + ' ' + x['question_plus'] if x['question_plus'] else x['question'], axis=1)

    # 각 질문의 길이를 계산
    df['question_length'] = df['full_question'].apply(len)
    
    return df


def compute_tfidf_features(df, max_features=1000):
    # TF-IDF 벡터화 수행
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['full_question'])
    
    # DataFrame 형태로 변환하여 반환
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    return tfidf_df


# Config 파일에서 프롬프트 경로 불러오기
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def load_prompt_template(filepath):
    """지정된 파일에서 프롬프트 템플릿을 불러옴."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

# 프롬프트 템플릿 로드
PROMPT_NO_QUESTION_PLUS = load_prompt_template(config["prompts"]["PROMPT_NO_QUESTION_PLUS_PATH"])
PROMPT_QUESTION_PLUS = load_prompt_template(config["prompts"]["PROMPT_QUESTION_PLUS_PATH"])

def process_dataset_with_prompts(df):
    dataset = Dataset.from_pandas(df)
    processed_dataset = []
    for i in range(len(dataset)):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])
        
        # question_plus 여부에 따라 프롬프트 템플릿 선택
        if dataset[i]["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                question_plus=dataset[i]["question_plus"],
                choices=choices_string,
            )
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                choices=choices_string,
            )
        
        processed_dataset.append(
            {
                "id": dataset[i]["id"],
                "messages": [
                    {"role": "system", "content": "질문의 답을 구하세요."},
                    # 지문을 읽고 질문의 답을 구하세요.
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"{dataset[i]['answer']}"}
                ],
                "label": dataset[i]["answer"],
            }
        )
    return Dataset.from_pandas(pd.DataFrame(processed_dataset))


def formatting_prompts_func(example, tokenizer):
    output_texts = []
    for i in range(len(example["messages"])):
        output_texts.append(
            tokenizer.apply_chat_template(
                example["messages"][i],
                tokenize=False,
            )
        )
    return output_texts


def tokenize(element, tokenizer):
    outputs = tokenizer(
        formatting_prompts_func(element, tokenizer),
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }


def process_and_tokenize_dataset(processed_dataset, tokenizer):
    # 데이터셋 토큰화
    tokenized_dataset = processed_dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=list(processed_dataset.features),
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Tokenizing",
    )
    return tokenized_dataset

def filter_and_split_dataset(tokenized_dataset, max_length=1024, test_size=0.1, seed=42):
    # max_length 토큰 이하의 데이터만 필터링
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    # 훈련 및 검증 데이터셋 분리
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']
    
    return train_dataset, eval_dataset


def format_test_data_for_model(test_df):
    # 테스트 데이터셋 가공
    test_dataset = []
    for _, row in test_df.iterrows():
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])

        # <보기>가 있을 때
        if row["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )
    return test_dataset