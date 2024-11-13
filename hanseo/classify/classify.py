### LIBRARY ###
import pandas as pd
import os
import time
import re
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import csv

### SETTINGS ###
API_KEY = 'AIzaSyCi7mh8iSC5JTycmqb39ypK0sLtPJkJ7R4'  # 실제 Google API 키로 대체하세요.
INPUT_FILE = './code/train.csv'
OUTPUT_FILE = 'classified.csv'
BATCH_SIZE = 3
MAX_RETRIES = 5
RETRY_DELAY = 10  # 재시도 대기 시간(초)
REQUEST_DELAY = 2  # 요청 간 대기 시간(초)
MAX_PARAGRAPH_LENGTH = 1000
MAX_PROBLEMS_LENGTH = 500

# API 키 설정
os.environ["GOOGLE_API_KEY"] = API_KEY

# 데이터 로드
try:
    # CSV 파일 읽기 옵션 설정
    data = pd.read_csv(
        INPUT_FILE,
        quoting=csv.QUOTE_ALL,  # 모든 필드를 인용 부호로 감싸는 경우
        encoding='utf-8',       # 인코딩 설정
        dtype=str               # 모든 컬럼을 문자열로 읽기
    )
    print(f"{INPUT_FILE}에서 총 {len(data)}개의 데이터를 로드했습니다.")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {INPUT_FILE}")
    exit(1)
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    exit(1)

# 데이터프레임 컬럼명 확인 및 처리
expected_columns = {'id', 'paragraph', 'problems', 'question_plus'}
missing_columns = expected_columns - set(data.columns)
if missing_columns:
    print(f"누락된 컬럼이 있습니다: {missing_columns}")
    # 필요한 경우 컬럼을 추가하거나 기본값을 설정
    for col in missing_columns:
        data[col] = ''

# 모델 초기화
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# 프롬프트 템플릿 정의
prompt_template = PromptTemplate(
    input_variables=["batch"],
    template="""
당신은 주어진 문제의 '지문'과 <보기>에 주어진 내용만으로 해당 문제에 답변할 수 있는지 평가하는 작업을 수행합니다.

각 문제는 대한민국 수능 시험과 유사한 형식으로, 다음과 같이 구성되어 있습니다:
- id: 문제의 id
- paragraph: 지문
- problems: 문제; question(질문), choices(선지), answer(정답) key를 가진 dictionary
- question_plus: <보기>

<판단 기준>
1. True: 
- 주어진 지문과 <보기>만을 읽고 문제의 정답을 정확하게 선택할 수 있는 경우. 즉, 문제를 풀기 위해 지문 외부의 추가적인 배경 지식(해당 시대의 역사적 배경, 사회적 상황, 관련 인물의 생애, 전문적인 지식 등 지문에 명시적으로 제시되지 않은 정보)이 필요하지 않고, 지문에 제시된 정보를 바탕으로 추론, 연역, 유추 등의 논리적 과정을 통해 답을 도출할 수 있는 경우.
- 예시 데이터: 
generation-for-nlp-1930,"기준금리 연 1%대 시대를 맞아 수익형 부동산 중 하나인 소형 오피스텔이 인기를 끌고 있다. 임대 수요가 꾸준한 지하철 역세권 인근 오피스텔의 경우 공실(空室) 가능성도 낮아 안정적인 월세 수입을 얻을 수 있어서다. 경기 수원 광교신도시에서 분양 중인 ‘광교 엘포트 아이파크’는 역세권 내 ‘아파트형 오피스텔’이다. 오피스텔로는 드물게 욕조를 따로 설치하고 수납공간도 넉넉하게 마련했다. 학습지 업체인 노벨과개미의 첫 시행 사업이다.○욕조·수납공간 넣은 오피스텔광교신도시 중심상업지역 1-2블록에 들어서는 이 오피스텔은 지하 8층~지상 최고 20층 규모로 지어진다. 총 1750실(전용면적 21~47㎡) 가운데 소형(전용 21~29㎡)이 전체의 83%인 1457실에 달한다. 3.3㎡당 분양가는 층과 방향에 따라 735만~947만원까지 다양하다. 1차 계약금은 500만원 정액제이고 중도금은 무이자(1~5차) 대출 지원이 이뤄진다.활용 공간이 상대적으로 좁은 오피스텔의 단점을 최대한 줄이는 방향으로 설계했다. 일부 타입 수납장 사이에 인출식 빨래건조대와 식탁 등을 설치, 공간 활용도를 높였다. 빨래건조대는 강선을 사용해 휨 발생을 줄였다. 식탁은 최대 80㎏의 하중을 견딜 수 있도록 했다. 또 모든 욕실에는 욕조를 넣어 기존 오피스텔과 차별화했다. PVC(염화비닐수지) 2중 창호로 마감, 결로(이슬 맺힘) 현상과 환기 문제도 해결했다. 전자레인지, 빌트인 김치냉장고, 에어컨, 천연 화강석 주방 상판, 일체형 비데 등을 기본으로 제공한다. 거실과 방이 나뉜 투룸에는 드레스룸 화장대 광파오븐 등을 설치한다.빌딩 4층에는 꽃과 다양한 나무들로 꾸민 정원과 외부 내방객을 위한 3개의 게스트룸을 마련한다. 피트니스센터, 실내골프연습장, 사우나 스크린골프부스 등 커뮤니티시설도 갖춘다. 주차장 비상콜과 홈오토시스템도 적용한다. 이동희 노벨과개미 분양관리팀장은 “설계에만 3년이 걸릴 정도로 심혈을 기울였다”며 “시행사가 자재 마감 인테리어 등의 부분까지 직접 챙겼다”고 말했다.○광교 행정타운 도보로 5분이 오피스텔은 경기도청 광교청사가 들어서는 행정타운까지 걸어서 5분이 채 안 걸린다. 10층 스카이북카페(면적 265㎡)에서 청사와 일산 호수공원의 2배 규모인 광교호수공원도 내려다볼 수 있다. 경기도청 주변으로 2019년까지 수원고등법원과 고등검찰청이 이전할 계획이다. 주변에 경기도청 및 법조타운, 경기대 아주대 등이 있어 배후 임대 수요가 많다는 게 회사 측 설명이다. 교통 여건도 좋아지고 있다. 내년에 신분당선 경기도청역(가칭)이 개통되면 강남역까지 30분대에 진입할 수 있다. 경기도청역환승센터(가칭)가 마련되면 도로 교통을 이용하기도 편해질 전망이다. 또 2018년 개통 예정인 경부고속도로와 용인서울고속도로를 연결하는 1단계 사업이 마무리되면 양재IC까지 18분 정도면 갈 수 있다. 모델하우스는 수원시 영통구 하동 1016에 마련돼 있다. 김진수 기자/김하나 한경닷컴 기자 true@hankyung.com", "{{'question': '광교 엘포트 아이파크 오피스텔의 특징 중 하나로, 기존 오피스텔과 차별화된 점은 무엇인가?', 'choices': ['욕조를 설치한 점', '전용면적이 21~47㎡인 점', '지하 8층~지상 20층 규모인 점', '무이자 대출 지원이 있는 점', '피트니스센터가 있는 점'], 'answer': 1}}",
generation-for-nlp-1931,(사)한국프로골프협회(이하 KPGA) 구자철 회장이 임성재(22.CJ대한통운)에게 ‘제84회 마스터스 토너먼트’ 준우승을 축하하는 축전을 보냈다. 17일 구자철 회장은 ‘’제84회 마스터스 토너먼트’에서 준우승이라는 쾌거를 달성한 것을 진심으로 축하한다”며 “첫 출전한 대회에서 아시아 선수 역대 최고 순위를 거뒀기에 그 의미는 더할 것”이라고 축하의 말을 건넸다. 이어 “한국 남자골프 사상 최초로 ‘마스터스 토너먼트’ 최종라운드 챔피언조에서 우승 경쟁을 펼친 임성재 선수의 플레이는 6천여 KPGA 회원들을 비롯해 밤새 중계를 지켜본 모두에게 잊지 못할 장면으로 남을 것”이라며 “임성재 선수의 강인한 도전 정신과 포기하지 않는 끈기로 이뤄낸 이번 성과는 우리 국민들에게 큰 자부심을 일깨워줬다”라고 전했다. 임성재는 지난 16일 끝난 ‘제84회 마스터스 토너먼트’에서 최종합계 15언더파 273타로 공동 2위로 대회를 마쳤다. 이는 ‘마스터스 토너먼트’에서 역대 아시아 국적 선수가 거둔 최고 성적으로 종전 기록은 2004년 대회에서 최경주(50.SK telecom)가 기록한 단독 3위였다. 우승 후 임성재는 “’마스터스 토너먼트’에 첫 출전했기 때문에 컷통과가 목표였다”며 “준우승은 믿을 수 없는 성적이다. 두고두고 기억에 남을 것”이라는 소감을 밝히기도 했다. ‘마스터스 토너먼트’ 준우승을 통해 임성재는 세계랭킹을 25위에서 18위까지 7계단이나 끌어올렸다. 임성재가 세계랭킹 20위 이내에 진입한 것은 이번이 처음이다. 2018~2019 시즌 PGA투어에 입성한 임성재는 데뷔 시즌에 ‘신인상(아널드파머 어워드)’을 수상했고 2019~2020 시즌 ‘혼다 클래식’에서 PGA투어 첫 우승컵을 들어올렸다. 임성재는 20일부터 나흘간 미국 조지아주 시아일랜드 리조트에서 열리는 ‘더 RSM 클래식’에 출전해 2020~2021 시즌 첫 승에 도전한다., "{{'question': '임성재가 ‘제84회 마스터스 토너먼트’에서 달성한 성적은 무엇인가?', 'choices': ['준우승', '단독 3위', '1위', '공동 2위', '4위'], 'answer': 1}}",
generation-for-nlp-1975,"김기현 울산시장이 17개 광역시·도 자치단체장에 대한 직무수행평가에서 가장 높은 평가를 받았다. ...", "{{'question': '김기현 울산시장이 직무수행평가에서 받은 긍정적인 평가 비율은 얼마인가?', 'choices': ['73%', '67%', ...], 'answer': 1}}",,True

2. False: 지문과 <보기>만으로는 문제의 정답을 정확하게 선택할 수 없고, 지문 외부의 추가적인 배경 지식이 필요한 경우. 즉, 지문만으로는 답을 알 수 없고, 문제를 풀기 위해 지문에 제시되지 않은 외부 지식이 필요하거나, 지문에 필요한 정보가 부족하거나 암시적으로 제시되어 일반적인 상식을 바탕으로 해석해야 하는 경우.
- 예시 데이터: 
generation-for-nlp-425,"상소하여 아뢰기를 , “신이 좌참 찬 송준길이 올린 차자를 보았는데 , 상복(喪服) 절차에 대하여 논한 것이 신과는 큰 차이가 있었습니다 . 장자를 위하여 3년을 입는 까닭은 위로 ‘정체(正體)’가 되기 때문이고 또 전 중(傳重: 조상의 제사나 가문의 법통을 전함)하기 때문입니다 . …(중략) … 무엇보다 중요한 것은 할아버지와 아버지의 뒤를 이은 ‘정체’이지, 꼭 첫째이기 때문에 참 최 3년 복을 입는 것은 아닙니다 .”라고 하였다 .－현종실록 －ㄱ.기 사환국으로 정권을 장악하였다 .ㄴ.인 조반정을 주도 하여 집권세력이 되었다 .ㄷ.정조 시기에 탕평 정치의 한 축을 이루었다 .ㄹ.이 이와 성혼의 문인을 중심으로 형성되었다.", "{{'question': '상소한 인물이 속한 붕당에 대한 설명으로 옳은 것만을 모두 고르면?', 'choices': ['ㄱ, ㄴ', 'ㄱ, ㄷ', 'ㄴ, ㄹ', 'ㄷ, ㄹ'], 'answer': 2}}",
generation-for-nlp-426,"(가)은/는 의병계열과 애국계몽 운동 계열의 비밀결사가 모여 결성된 조직으로, 총사령 박상진을 중심으로 독립군 양성을 목적으로 하였다.", "{{'question': '(가)에 대한 설명으로 옳지 않은 것은?', 'choices': ['고려 문종 때에 남경(南京)으로 승격되었다.', '종루(鐘樓), 이현, 칠패 등에서 상업활동이 이루어졌다.', '정도전은 궁궐 전각(殿閣)과도성성문 등의 이름을 지었다.', '성곽은 거중기 등을 이용하여 약 2년 만에 완성되었다.'], 'answer': 1}}",
generation-for-nlp-427,나는 삼한(三韓) 산천의 음덕을 입어 대업을 이루었다.(가)는/은 수덕(水德)이 순조로워 우리나라 지맥의 뿌리가 되니 대업을 만대에 전할 땅이다. 왕은 춘하 추동네 계절의 중간달에 그곳에 가 100일 이상 머물러서 나라를 안녕케 하라. － 고려사－, "{{'question': '(가) 지역에 대한 설명으로 옳은 것은?', 'choices': ['이곳에 대장도감을 설치하여 재조대장경을 만들었다.', '지눌이 이곳에서 수선사 결사운동을 펼쳤다.', '망이 ․망소이가 이곳에서 봉기하였다.', '몽골이 이곳에 동녕부를 두었다.'], 'answer': 4}}",
generation-for-nlp-428,"이 날 소정방이 부총관 김인문 등과 함께 기 벌포에 도착하여 백제 군사와 마주쳤다. …(중략) …소정방이 신라군이 늦게 왔다는 이유로 군문에서 신라 독군 김문영의 목을 베고자 하니, 그가 군사들 앞에 나아가 “황산 전투를 보지도 않고 늦게 온 것을 이유로 우리를 죄 주려 하는구나. 죄도 없이 치욕을 당할 수는 없으니, 결단코 먼저 당나라 군사와 결전을 한 후에 백제를 쳐야겠다.”라고 말하였다.", "{{'question': '밑줄 친 ‘그’에 대한 설명으로 옳은 것은?', 'choices': ['살수에서 수의 군대를 물리쳤다 .', '김춘추 의 신라 왕위 계승을 지원하였다 .', '청해진을 설치하고 해상 무역을 전개하였다 .', '대가야를 정벌하여 낙동강 유역을 확보하였다 .'], 'answer': 2}}",
generation-for-nlp-429,"선비들 수만 명이 대궐 앞에 모여 만 동묘와 서원을 다시 설립할 것을 청하니, (가)이/가 크게 노하여 한성부의 조례(皂隷)와 병졸로 하여 금 한 강 밖으로 몰아내게 하고 드디어 천여 곳의 서원을 철폐하고 그 토지를 몰수하여 관에 속하게 하였다.－대한계년사 －", "{{'question': '(가) 인물이 추진한 정책으로 옳지 않은 것은?', 'choices': ['사창제를 실시하였다 .', '대전회통을 편찬하였다 .', '비변사의 기능을 강화하였다 .', '통상 수교 거부 정책을 추진하였다 .'], 'answer': 3}}",
generation-for-nlp-431,"(가)의 사신 고제덕 등이 일본에 와서 왕이 보낸 국서를 전하였다. 그 국서에 이르기를 “나(대무 예)는 큰 나라를 맡아 여러 주변국을 다스렸으며, 고구려의 옛 땅을 회복하였고 부여의 풍속을 이었다.”라고 하였다.", "{{'question': '(가) 국가에 대한 설명으로 옳은 것은?', 'choices': ['나 당연합군의 공격으로 멸망하였다 .', '9주 5소경의 지방 행정 구역을 두었다 .', '중앙 행정 기구로 3성 6부를 설치하였다 .', '고구려의 수도였던 평양을 서경으로 삼았다 .'], 'answer': 3}}",
generation-for-nlp-432,(가)신라의 한강 유역 확보 (나)관산성 전투(다) 백제의 웅진 천도 (라)고구려의 평양 천도, "{{'question': '다음 사건을 시기 순으로 바르게 나열한 것은?', 'choices': ['(가)→(라)→(나)→(다)', '(나)→(다)→(가)→(라)', '(다)→(나)→(가)→(라)', '(라)→(다)→(가)→(나) '], 'answer': 4}}",
generation-for-nlp-433,"신돈이 (가)을/를 설치하자고 요청하자, …(중략)…이 제 도감이 설치되었다. …(중략)… 명령이 나가자 권세가 중에 전민을 빼앗은 자들이 그 주인에게 많이 돌려주었으며, 전국에서 기뻐하였다.－고려사－", "{{'question': '(가)에 대한 설명으로 옳은 것은?', 'choices': ['시 전의 물가를 감독하는 임무를 담당하였다 .', '국가재정의 출납과 회계 업무를 총괄하였다 .', '불법적으로 점유된 토지와 노비를 조사하였다 .', '부족한 녹봉을 보충하고자 관료에게 녹과 전을 지급하였다 .'], 'answer': 3}}",
generation-for-nlp-434,(가) 황제가 영원히 가시던 길에 엎드려 크게 통곡하던 우리는 …(중략) … 우리민족의 새로운 기백과 책동이 발발하기를 간절히 기대하는 바이다.－동아일보 1926년 6월 12일－, "{{'question': '(가) 재위 기간에 있었던 사실이 아닌 것은?', 'choices': ['일본은 동양척식 주식회사를 설립하였다.', '일본이 간도를 청에 귀속하는 협약을 체결하였다.', '유생의 병장 중심으로 13도 창의군을 결성하였다.', '대한제국의 외교권을 박탈하고 통감부를 설치하였다.'], 'answer': 1}}",
generation-for-nlp-435,"올해 초가을에 비로소 저는 책을 완성하여 그 이름을 성학집요 라고 하였습니다. 이 책에는 임금이 공부해야 할 내용과 방법, 정치하는 방법, 덕을 쌓아 실천하는 방법과 백성을 새롭게 하는 방법이 실려 있습니다. 또한 작은 것을 미루어 큰 것을 알게 하고 이것을 미루어 저것을 밝혔으니, 천하의 이치가 여기에서 벗어나 지 않을 것입니다. 따라서 이것은 저의 글이 아니라 성현의 글이 옵니다.", "{{'question': '밑줄 친 ‘저’에 대한 설명으로 옳은 것은?', 'choices': ['예안향약을 만들었다 .', '동호문답 을 저술하였다 .', '백운동 서원을 건립하였다 .', '왕자의 난 때 죽임을 당했다 .'], 'answer': 2}}",

<출력 형식> 
- 각 문제에 대해 아래 형식으로만 답변해 주세요:
id: [문제의 id] answerable: [True 또는 False]
- 추가적인 설명이나 텍스트는 포함하지 말아주세요.

<데이터>: 
{batch}

각 문제를 평가하고, 지문만으로 문제에 답을 제공할 수 있는지 판단해 주세요.
"""
)

# 기존 출력 파일 삭제
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print(f"기존 {OUTPUT_FILE} 파일을 삭제하고 새로 생성합니다.")

def truncate_text(text, max_length):
    return text if len(text) <= max_length else text[:max_length] + '...'

def process_batch(batch_df):
    batch_texts = []
    for _, row in batch_df.iterrows():
        id = row['id']
        paragraph = truncate_text(row['paragraph'], MAX_PARAGRAPH_LENGTH)
        problems = truncate_text(row['problems'], MAX_PROBLEMS_LENGTH)
        batch_texts.append(f"id: {id}\nparagraph: {paragraph}\nproblems: {problems}\n")

    batch_prompt = "\n".join(batch_texts)
    prompt = prompt_template.format(batch=batch_prompt)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.invoke(prompt)
            response_text = response.content.strip()
            print("모델의 응답:")
            print(response_text)
            return response_text
        except Exception as e:
            print(f"모델 호출 중 오류 발생: {e}")
            if attempt < MAX_RETRIES:
                print(f"{RETRY_DELAY}초 후 재시도합니다... (시도 {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                print("최대 재시도 횟수를 초과했습니다. 이 배치를 건너뜁니다.")
                return None

def parse_response(response_text, batch_df):
    results = []
    pattern = r'id:\s*(.*?)\s*answerable:\s*(True|False)'
    matches = re.findall(pattern, response_text, re.DOTALL)

    if len(matches) != len(batch_df):
        print("경고: 응답 개수와 입력된 문제 개수가 일치하지 않습니다.")
        print("순서대로 매칭하여 처리합니다.")

    for idx, row in batch_df.reset_index(drop=True).iterrows():
        id = row['id']
        paragraph = row['paragraph']
        problems = row['problems']
        question_plus = row.get('question_plus', '')
        if idx < len(matches):
            matched_id, answerable = matches[idx]
            answerable = answerable.strip()
        else:
            answerable = 'False'  # 응답이 없을 경우 False로 처리
            print(f"id {id}에 대한 응답이 없어 False로 설정합니다.")

        results.append({
            "id": id,
            "paragraph": paragraph,
            "problems": problems,
            "question_plus": question_plus,
            "answerable": answerable
        })
        print(f"id {id} 처리 완료: answerable = {answerable}")

    return results

def save_results(results):
    df = pd.DataFrame(results)
    # 첫 번째 배치인지 확인하여 헤더 포함 여부 결정
    write_header = not os.path.exists(OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, mode='a', index=False, encoding='utf-8-sig', header=write_header)
    print(f"{len(results)}개의 결과를 {OUTPUT_FILE}에 저장했습니다.")

# 메인 처리 루프
total_records = len(data)
for start_idx in range(0, total_records, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, total_records)
    batch_df = data.iloc[start_idx:end_idx]
    print(f"{start_idx + 1}번째부터 {end_idx}번째 데이터 처리 중...")

    response_text = process_batch(batch_df)
    if response_text is None:
        continue

    results = parse_response(response_text, batch_df)
    save_results(results)

    # 요청 간 딜레이
    time.sleep(REQUEST_DELAY)

print("모든 데이터 처리를 완료했습니다.")
