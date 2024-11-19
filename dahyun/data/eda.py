import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기 (예시 경로)
file_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/juhyun/output_gemma_cleaned/output.csv'
data = pd.read_csv(file_path)

# 'problems' 열이 문자열로 저장되어 있을 경우, eval로 복원
#data['problems'] = data['problems'].apply(eval)

# 각 문제의 answer 값을 추출


# answer 값의 빈도 계산 (오름차순 정렬)
answer_counts = data['answer'].value_counts().sort_index()

# 그래프 그리기
plt.figure(figsize=(8, 5))
answer_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('output data: Distribution of Answers', fontsize=14)
plt.xlabel('Answer', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 그래프 이미지 저장
image_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/image3.png'
plt.savefig(image_path, format='png')
plt.close()

print(f"그래프가 저장되었습니다: {image_path}")
