# Level 2: 수능형 문제 풀이 모델 생성 (Generation for NLP)

## 📝 Abstract (수정 필요)
- [📈 Project Wrap-Up Report](https://github.com/user-attachments/files/17674538/DataCentric_13.Wrap-Up.pdf)
- 이 프로젝트는 부스트캠프 AI Tech 7기 NLP track 경진 대회로, Dacon, Kaggle과 유사한 대회형 방식으로 진행되었다.
- 소형 언어 모델을 사용해 한국어 및 수능 시험에 최적화된 AI 모델을 개발하여 GPT, Claude, Gemini와 같은 대형 모델을 능가하는 성능을 목표로 하였다.

<br>

## 🏆 Leader Board
### Public
<img width="700" alt="public_leader_board" src="https://github.com/user-attachments/assets/e2f86836-f39f-461a-beab-c2aa01076874">

### Private
<img width="700" alt="private_leader_board" src="https://github.com/user-attachments/assets/4f4edb2b-7e48-468e-90a0-e484f464741a">

<br>

## 👔 Team Introduction 

> **Team NLP크민**

### 👨🏼‍💻 Members
권지수 | 김성은 | 김태원 | 이다현 | 이한서 | 정주현
:-: | :-: | :-: | :-: | :-: | :-:
<img src='https://github.com/user-attachments/assets/ab4b7189-ec53-41be-8569-f40619b596ce' height=125 width=100></img> | <img src='https://github.com/user-attachments/assets/49dc0e59-93ee-4e08-9126-4a3deca9d530' height=125 width=100></img> | <img src='https://github.com/user-attachments/assets/a15b0f0b-cd89-412b-9b3d-f59eb9787613' height=125 width=100></img> | <img src='https://github.com/user-attachments/assets/4064f03a-a1dc-4dd1-ac84-d9ac8636418a' height=125 width=100></img> | <img src='https://github.com/user-attachments/assets/11b2ed88-bf94-4741-9df5-5eb2b9641a9b' height=125 width=100></img> | <img src='https://github.com/user-attachments/assets/3e2d2a7e-1c64-4cb7-97f6-a2865de0c594' height=125 width=100></img>
<a href="mailto:wltn80609@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a> | <a href="mailto:sunny020111@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a> | <a href="mailto:chris40461@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a> | <a href="mailto:dhdh09290929@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a> | <a href="mailto:beaver.zip@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a> | <a href="mailto:peter520416@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|
<a href="https://github.com/Kwon-Jisu" target="_blank"><img src="https://img.shields.io/badge/GitHub-Kwon--Jisu-181717?style&logo=GitHub&logoColor=white" /></a> | <a href="https://github.com/ssungni" target="_blank"><img src="https://img.shields.io/badge/GitHub-ssungni-181717?style&logo=GitHub&logoColor=white" /></a> | <a href="https://github.com/chris40461" target="_blank"><img src="https://img.shields.io/badge/GitHub-chris40461-181717?style&logo=GitHub&logoColor=white" /></a> | <a href="https://github.com/dhl0929" target="_blank"><img src="https://img.shields.io/badge/GitHub-dhl0929-181717?style&logo=GitHub&logoColor=white" /></a> | <a href="https://github.com/beaver-zip" target="_blank"><img src="https://img.shields.io/badge/GitHub-beaver--zip-181717?style&logo=GitHub&logoColor=white" /></a> | <a href="https://github.com/peter520416" target="_blank"><img src="https://img.shields.io/badge/GitHub-peter520416-181717?style&logo=GitHub&logoColor=white" /></a>

### 🧑🏻‍💻 Role (수정 필요)

| 이름 | 역할 |
| :---: | --- |
| **`권지수`** | **Text/Label Noise Split**, **Text Cleaning**, **Prompt-Based Generation** |
| **`김성은`** | **Text/Label Noise Split**, **Back-Translation**(Google Translate), **Prompt-Based Generation** |
| **`김태원`** | **EDA**, **Re-Labeling**, **Evol-Instruct LLM for Augmentation**  |
| **`이다현`** | **EDA**, **Re-Labeling**, **Evol-Instruct LLM for Augmentation**  |
| **`이한서`** | **Text/Label Noise Split**, **Text Cleaning**, **Back-translation**(DeepL), **Prompt-Based Generation** |
| **`정주현`** | **Re-Labeling**, **ML Model Searching** |

<br>

## 🖥️ Project Introduction 

| **주제** | 소형 언어 모델을 사용해 대형 모델을 능가하는 한국어 및 수능 시험 최적화 AI 모델 개발 |
| :---: | --- |
| **구현 내용** | 주어진 수능형 문제 데이터셋을 전처리 및 증강하여 Train Dataset을 구성하고, 사전 학습된 소형 모델을 Train Dataset에 맞게 파인튜닝 후 학습을 진행한다. 이후 학습된 모델로 Test Dataset에 대해 추론을 수행한다. 이 과정에서 최적의 데이터셋, 모델, 파라미터 및 프롬프트를 탐구한다. |
| **개발 환경** | **• `GPU`:** NVIDIA Tesla V100 32GB 서버 4개 <br> **• `Tool`:** VS Code, Jupyter Notebook |
| **협업 환경** | **• `Zoom`:** 실시간 비대면 회의 <br> **• `Github`:** 코드, 데이터 공유 및 버전 관리 <br> **• `Notion`:** 역할 분담, 실험 가설 설정 및 경과 공유 |

<br>

## 📁 Project Structure

```
📂code
  ┣ 📜requirements.txt
  ┣ 📜config.json
  ┣ 📜make_train_dataset.ipynb
  ┣ 📜data_processing.py
  ┣ 📜utils.py
  ┣ 📜train.py
  ┗ 📜test.py

 📂data
  ┣ 📜aug_gemini.csv
  ┣ 📜aug_gpt.csv
  ┣ 📜classified_train_raw.csv
  ┣ 📜classified_train_processed.csv
  ┣ 📜classified_train_processed_filtered.csv
  ┣ 📜train_augmented.csv
  ┗ 📜train_final.csv

📂prompts
  ┣ 📜prompt_no_question_plus.txt
  ┗ 📜prompt_question_plus.txt
 ```

<br>

## 📐 Project Ground Rule
> 효율적인 협업을 위해 매일 아침 ZOOM 화상 회의를 통해 각자의 진행 상황과 당일 실험 계획을 공유하였으며, 오후 4시부터 7시까지 다시 ZOOM 회의를 통해 실험 결과를 실시간으로 공유하고 피드백을 주고받았다.
- **`Server`:** 모든 서버를 Git으로 연동해 버전 관리, 유휴 상태의 서버에서 실험 진행
- **`Git`:** 개인별 branch에 작업한 코드 및 데이터 공유
- **`Notion`:** 가설, 구현 및 실험 결과를 기록해 진행 상황 공유
- **`Submission`:** 리더보드 마감 5일 전까지 자유롭게 제출, 이후 인당 1일 2회 제출

<br>

## 🗓 Project Time Line
> 2024.11.11.(월)-11.28.(목) (17일간)
- **1~5 일차:** 강의 수강, Baseline Code 분석 및 데이터 EDA
- **6~12 일차:** 모델 탐색 및 실험, 데이터 정제
- **13~17 일차:** 데이터 증강, 파라미터 및 프롬프트 실험

<br>

## 💻 Getting Started

### ⌨️ How To Install Requirements
```bash
pip3 install -r requirements.txt
```

### ⌨️ How To Make Train Dataset
```bash
make_train_dataset.ipynb
```

### ⌨️ How To Train and Test
```bash
python3 train.py
python3 test.py
```
