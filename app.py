import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
import re

import os
import nltk

# NLTK 데이터 경로 설정
if os.name == 'nt':  # Windows
    nltk_data_path = os.path.join(os.getenv('APPDATA'), 'nltk_data')
else:  # Linux/Mac
    nltk_data_path = os.path.expanduser('~/nltk_data')

nltk.data.path.append(nltk_data_path)

# NLTK 데이터 다운로드
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)

# NLTK 데이터 다운로드 경로 설정
nltk_data_path = os.path.join(os.getenv('APPDATA'), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

# 질문과 답변 데이터베이스
qa_dict = {
    "기숙사": "이리고등학교 기숙사는 2개의 건물이 있으며 각 학년별 30명씩 총 90명을 선발합니다. 총 선발 인원의 20%는 원거리 배려 선발, 나머지는 성적을 기준으로 일반 선발합니다. 성적을 볼 때 내신 뿐만 아니라 모의고사 성적도 함께 포함됩니다. 이리고등학교의 기숙사는 총 4분기로 구성되어 있으며 3개월이 1분기입니다. 분기마다 입사 신청이 가능하니 홈페이지를 잘 확인해주시기 바라며 더 자세한 사항은 학교 홈페이지를 확인해주시기 바랍니다.",
    "학생회": "학생회는 전교 회장단의 선거가 끝난 후 매년 8월 말에서 9월 면접을 통해 선출되며, 학생들의 권익을 대변하고 다양한 행사를 주관합니다. 총 10개의 부서가 있으며 원하는 부서에 신청을 해 면접을 통해 선발됩니다. 임기는 1년입니다.(해당년도 2학기 + 다음년도 1학기)",
    "급식실": "급식실은 교문에서 들어와서 왼쪽에 존재하며 급식실 건문 2층엔 강당이 있습니다.",
    "등교": "등교할 때는 크록스나 슬리퍼를 신으면 안되고 운동화를 신고 교복을 입은 채로 등교해야 합니다. 교문 앞에서 학생회 선생님께서 매일매일 검사하시니 주의해주세요.",
    "동아리": "매년 3월 중순에서 말 동아리를 선정하기 위해 동아리를 홍보하는 시간을 가질 예정입니다. 각 동아리의 기장이 돌아다녀 본인 동아리에 대한 설명 등을 할 예정입니다.",
    "축제": "이리고등학교의 축제는 2년에 한번씩 홀수 해에 진행합니다. 축제는 1학기 말 7월 중 열릴 예정이며 원광대학교에서 대관을 하여 진행합니다.",
    "보건실":"이리고등학교의 보건실은 본관 1층 오른쪽 맨 끝에 위치합니다. 다양한 이벤트도 진행하오니 많은 관심 가져주시기 바랍니다.",
    "수학여행":"수학여행은 1학년 2학기 10월 중으로 많이 가는 편입니다.",
    "체육대회":"체육대회는 1학기 1차고사가 끝난 5월 초중에 많이 하며 1,2,3학년 전체가 함께 2일 동안 진행합니다.",
    "인사":"안녕하세요! 무엇을 알려드릴까요?",
    "운동장":"운동장은 인조잔디로 되어있어 축구화를 신고 운동장에 들어가야합니다. 과학동에 갈 때 운동장을 가로질러 가지 않도록 주의해주세요.",
    "도서관":"도서관은 미래관 2층에 있습니다.",
    "미래관":"교문에서 들어왔을 때 오른쪽에 있는 가장 큰 건물로 예전에는 과학동이라고 불렸습니다. 과학 실험실, 도서관, 미술실 등 여러 교실이 존재합니다.",
    "화장실":"화장실은 후관 각 층마다 2개씩 존재하며 3층 교무실 바로 옆 화장실은 여자화장실입니다. 화장실에는 휴지가 존재하지 않으니 주의해주시기 바랍니다.",
    "컴퓨터실":"컴퓨터실은 명문관 옆 미령관(기숙사)과 미래관 사이에 있는 건물 1층에 있습니다.",
    "매점":"매점은 본관과 후관 사이 왼쪽에 위치해있습니다. 쉬는시간 외에 수업시간에는 매점을 운영하지 않으니 알아두시기 바랍니다.",
    "와이파이":"와이파이 비밀번호의 이름은 JBEDU_WIFI-S이고 비밀번호는 jbe_iri_20st 입니다",
    "주차장":"주차장은 교문 오른쪽과 급식실 아래에 위치해 있습니다.",
    "엘리베이터":"후관에는 엘리베이터가 존재하지 않고 본관에는 존재합니다.",
    "스마트폰":"스마트폰은 규정상 수거하지 않고 있습니다. 하지만 수업시간에 사용하다 걸리면 선생님께 제출해야 할 수도 있습니다.",
    "휴대폰":"스마트폰은 규정상 수거하지 않고 있습니다. 하지만 수업시간에 사용하다 걸리면 선생님께 제출해야 할 수도 있습니다.",
    "폰":"스마트폰은 규정상 수거하지 않고 있습니다. 하지만 수업시간에 사용하다 걸리면 선생님께 제출해야 할 수도 있습니다.",
    "핸드폰":"스마트폰은 규정상 수거하지 않고 있습니다. 하지만 수업시간에 사용하다 걸리면 선생님께 제출해야 할 수도 있습니다.",
}

# 한국어 전처리 함수
def preprocess(text):
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 소문자 변환
    text = text.lower()
    return text

# TF-IDF 벡터라이저 초기화
vectorizer = TfidfVectorizer(preprocessor=preprocess)

# 질문들 벡터화
questions = list(qa_dict.keys())
X = vectorizer.fit_transform(questions)

st.title("이리고 챗봇")

# 사용자 입력
user_input = st.text_input("이리고등학교에 대해 궁금한 것을 물어보세요(이리고등학교에 관한 질문만 답변합니다.)")

if user_input:
    # 전처리
    processed_input = preprocess(user_input)
    
    # 규칙 기반 키워드 매칭
    for key, value in qa_dict.items():
        if key in processed_input:
            st.write(f"답변: {value}")
            break
    else:
        # 키워드 매칭 실패 시 TF-IDF 유사도 계산
        user_vector = vectorizer.transform([processed_input])
        similarities = cosine_similarity(user_vector, X).flatten()
        best_match_index = similarities.argmax()
        
        if similarities[best_match_index] > 0.3:  # 유사도 임계값
            st.write(f"답변: {qa_dict[questions[best_match_index]]}")
        else:
            st.write("죄송합니다. 해당 질문에 대한 답변을 찾지 못했습니다. 다른 방식으로 질문해 주시거나 학생회에 직접 문의해 주세요.")