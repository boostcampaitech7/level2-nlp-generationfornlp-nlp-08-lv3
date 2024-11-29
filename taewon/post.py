import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('quiz_korean.csv', encoding='utf-8-sig')

# 새로운 데이터프레임 생성
new_df = pd.DataFrame(columns=['id', 'paragraph', 'problems', 'question_plus'])

for index, row in df.iterrows():
    # id 생성
    new_id = f"quiz_korean_{row['번호(Q_no)']}"
    
    # paragraph 설정
    paragraph = row['정답 근거(AP_sent)']
    
    # problems 문자열 생성
    choices = [
        row['보기()E_string'],
        row['보기()E_string.1'],
        row['보기()E_string.2'],
        row['보기()E_string.3']
    ]
    choices_str = "['" + "', '".join(choices) + "']"
    
    # 정답 찾기
    answer = next((i+1 for i, choice in enumerate(choices) if choice == row['정답(A_ans)']), None)
    
    problems = f"{{'question': '{row['질문(Q_string)']}', 'choices': {choices_str}, 'answer': {answer}}}"
    
    # 새로운 행 추가
    new_df = new_df._append({
        'id': new_id,
        'paragraph': paragraph,
        'problems': problems,
        'question_plus': ''  # 필요한 경우 여기에 추가 정보를 넣을 수 있습니다
    }, ignore_index=True)

# 새로운 CSV 파일로 저장
new_df.to_csv('new_quiz_format.csv', index=False, encoding='utf-8-sig')