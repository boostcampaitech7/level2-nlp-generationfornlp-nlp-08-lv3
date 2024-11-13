import streamlit as st
import pandas as pd

# CSV 파일 로드
@st.cache_data
def load_data():
    return pd.read_csv('classified_train_processed.csv')

df = load_data()

# 앱 제목
st.title('수능 문제 분석 대시보드')

# 사이드바에 필터 옵션 추가
st.sidebar.header('필터 옵션')

# 주제 선택
topics = ['모두'] + list(df['topic'].unique())
selected_topic = st.sidebar.selectbox('주제 선택', topics)

# 사전 지식 필요 여부 선택
prior_knowledge = ['모두'] + list(df['prior_knowledge_needed'].unique())
selected_prior_knowledge = st.sidebar.selectbox('사전 지식 필요 여부', prior_knowledge)

# 데이터 필터링
if selected_topic != '모두':
    df = df[df['topic'] == selected_topic]
if selected_prior_knowledge != '모두':
    df = df[df['prior_knowledge_needed'] == selected_prior_knowledge]

# 필터링된 데이터 표시
st.write(f'표시된 문제 수: {len(df)}')
st.dataframe(df)

# 문제 상세 보기
st.header('문제 상세 보기')
selected_id = st.selectbox('문제 ID 선택', df['id'].unique())

if selected_id:
    problem = df[df['id'] == selected_id].iloc[0]
    st.subheader(f'문제 ID: {problem["id"]}')
    st.write(f'주제: {problem["topic"]}')
    st.write(f'사전 지식 필요 여부: {problem["prior_knowledge_needed"]}')
    st.write('지문:')
    st.text(problem['paragraph'])
    st.write('질문:')
    st.text(problem['question'])
    st.write('선택지:')
    st.text(problem['choices'])
    st.write(f'정답: {problem["answer"]}')
    if problem['question_plus']:
        st.write('추가 정보:')
        st.text(problem['question_plus'])

# 간단한 통계
st.header('간단한 통계')
st.write('주제별 문제 수:')
st.bar_chart(df['topic'].value_counts())

st.write('사전 지식 필요 여부별 문제 수:')
st.bar_chart(df['prior_knowledge_needed'].value_counts())
