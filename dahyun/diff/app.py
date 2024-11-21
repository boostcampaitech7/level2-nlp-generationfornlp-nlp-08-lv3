import streamlit as st
import pandas as pd

# 데이터 로드
@st.cache_data
def load_data():
    test_df = pd.read_csv('../data/test.csv')
    output1_df = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/diff/output_shuffle.csv') # Sota
    output2_df = pd.read_csv('/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/diff/output_shuffle2.csv') # 현재 결과
    return test_df, output1_df, output2_df

test_df, output1_df, output2_df = load_data()

# 제목
st.title('수능 모델 정답 비교')

# 정답이 다른 문항 찾기
different_answers = output1_df[output1_df['answer'] != output2_df['answer']]

# 결과 표시
st.write(f"총 {len(different_answers)}개의 문항에서 정답이 다릅니다.")

# 드롭다운 메뉴로 문항 선택
selected_question = st.selectbox(
    '확인할 문항을 선택하세요:',
    options=different_answers['id'].tolist(),
    format_func=lambda x: f"문항 ID: {x}"
)

if selected_question:
    question_id = selected_question
    output1_answer = different_answers.loc[different_answers['id'] == question_id, 'answer'].values[0]
    output2_answer = output2_df.loc[output2_df['id'] == question_id, 'answer'].values[0]
    
    st.subheader(f"문항 ID: {question_id}")
    
    # test.csv에서 해당 문항 정보 가져오기
    question_info = test_df[test_df['id'] == question_id].iloc[0]
    
    # 문단 표시
    st.write("**문단:**")
    st.write(question_info['paragraph'])
    
    # 문제 표시
    st.write("**문제:**")
    problem_dict = eval(question_info['problems'])
    st.write(problem_dict['question'])
    
    # 선택지 표시
    st.write("**선택지:**")
    for i, choice in enumerate(problem_dict['choices'], 1):
        st.write(f"{i}. {choice}")
    
    # 모델 답안 비교
    st.write("**모델 답안:**")
    st.write(f"Sota Output : {output1_answer}")
    st.write(f"Your Output:  {output2_answer}")