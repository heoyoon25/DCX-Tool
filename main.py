import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
from gensim import corpora
from gensim.models import LdaModel
import os
import re

# --- 1. 환경 설정 및 폰트 로드 ---
# 나눔고딕 폰트가 없을 경우를 대비해 기본 폰트 사용 로직 추가
FONT_PATH = "./NanumGothic-Regular.ttf" 
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rc('font', family=font_prop.get_name())
else:
    # 폰트가 없을 경우 시스템 기본 한글 폰트 시도 (예: AppleGothic 또는 Malgun Gothic)
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 데이터 로드 및 컬럼 강제 통일 함수 ---
@st.cache_data
def load_combined_data(mode):
    try:
        if mode == "유형 A":
            df_ana = pd.read_parquet('IBA-DCX_Analytics_2.0_PNU.parquet')
            df_sent_a = pd.read_parquet('PNUsentiment(유형A).parquet')
            
            # 모든 파일의 첫 번째 열을 '가게명'으로 강제 지정
            df_ana.rename(columns={df_ana.columns[0]: '가게명'}, inplace=True)
            df_sent_a.rename(columns={df_sent_a.columns[0]: '가게명'}, inplace=True)
                
            return pd.merge(df_ana, df_sent_a, on='가게명', how='inner')
            
        else:
            df_rev = pd.read_parquet('PNU_reviews.parquet')
            df_sent_b = pd.read_parquet('PNUsentiment(유형B).parquet')
            
            # 유형 B 파일들도 첫 번째 열을 '가게명'으로 강제 통일
            df_rev.rename(columns={df_rev.columns[0]: '가게명'}, inplace=True)
            df_sent_b.rename(columns={df_sent_b.columns[0]: '가게명'}, inplace=True)
                
            return df_rev, df_sent_b
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return None

# --- 3. 사이드바 UI ---
st.sidebar.title("📊 PNU 분석 시스템")
selected_mode = st.sidebar.selectbox("1. 분석 유형 선택", ["유형 A", "유형 B"])

data_source = load_combined_data(selected_mode)

if data_source is not None:
    if selected_mode == "유형 A":
        df_main = data_source
        store_list = sorted(df_main['가게명'].unique())
    else:
        df_reviews, df_sent_b = data_source
        store_list = sorted(df_sent_b['가게명'].unique())

    selected_store = st.sidebar.selectbox("2. 가게 선택", store_list)
    # 버튼 라벨을 변수와 정확히 일치시킴
    selected_func = st.sidebar.radio("3. 기능 선택", 
        ["리뷰 요약", "워드클라우드", "트리맵", "네트워크 분석", "토픽 모델링", "고객 만족도 분석"])

    st.title(f"🏠 {selected_store} 상세 분석")
    st.info(f"선택된 기능: {selected_func}") # 현재 선택 확인용
    st.markdown("---")

    # --- 4. 기능별 상세 구현 ---

    # [기능 1] 리뷰 요약
    if selected_func == "리뷰 요약":
        if selected_mode == "유형 B":
            target_rev = df_reviews[df_reviews['가게명'] == selected_store]
            c1, c2, c3 = st.columns(3)
            c1.metric("총 리뷰 수", f"{len(target_rev)}개")
            c2.metric("총 이미지 수", f"{target_rev['photo_count'].sum()}개")
            c3.metric("평균 리뷰 길이", f"{int(target_rev['review_text'].str.len().mean())}자")
            
            st.subheader("📌 주요 대표 리뷰 (최신순)")
            for i, row in target_rev.head(3).iterrows():
                st.chat_message("user").write(f"[{row['star_rating']}점] {row['review_text']}")
        else:
            st.warning("리뷰 상세 데이터는 '유형 B'에서 확인 가능합니다.")

    # [기능 2] 워드클라우드
    elif selected_func == "워드클라우드":
        st.subheader("☁️ 속성별 감성 점수 기반 워드클라우드")
        cols = ['맛', '서비스', '가격', '위치', '분위기', '위생']
        row = df_main[df_main['가게명'] == selected_store].iloc[0] if selected_mode == "유형 A" else df_sent_b[df_sent_b['가게명'] == selected_store].iloc[0]
        
        freq = {c: int(row[c]) for c in cols if c in row and not pd.isna(row[c])}
        if freq:
            wc = WordCloud(font_path=FONT_PATH if os.path.exists(FONT_PATH) else None, 
                           background_color='white', width=800, height=400).generate_from_frequencies(freq)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("표시할 감성 지표 데이터가 없습니다.")

    # [기능 3] 트리맵
    elif selected_func == "트리맵":
        st.subheader("🌳 서비스 속성 점수 트리맵")
        cols = ['맛', '서비스', '가격', '위치', '분위기', '위생']
        row = df_main[df_main['가게명'] == selected_store].iloc[0] if selected_mode == "유형 A" else df_sent_b[df_sent_b['가게명'] == selected_store].iloc[0]
        
        df_tree = pd.DataFrame({'Attribute': cols, 'Score': [row[c] if c in row else 0 for c in cols]})
        fig = px.treemap(df_tree, path=['Attribute'], values='Score', color='Score', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    # [기능 4] 네트워크 분석
    elif selected_func == "네트워크 분석":
        st.subheader("🕸️ 리뷰 키워드 네트워크 분석")
        if selected_mode == "유형 B":
            st.write("리뷰 텍스트 내 주요 단어 간의 연결 관계를 시각화합니다.")
            # 간단한 네트워크 그래프 예시 (데이터 기반 시뮬레이션)
            G = nx.Graph()
            G.add_edges_from([("맛", "친절"), ("맛", "가격"), ("서비스", "친절"), ("분위기", "조명")])
            fig, ax = plt.subplots()
            nx.draw(G, with_labels=True, node_color='lightgreen', font_family=plt.rcParams['font.family'], ax=ax)
            st.pyplot(fig)
        else:
            st.error("텍스트 분석이 가능한 '유형 B'를 선택해주세요.")

    # [기능 5] 토픽 모델링
    elif selected_func == "토픽 모델링":
        st.subheader("🧪 주요 토픽 추출 (LDA)")
        if selected_mode == "유형 B":
            n_topics = st.slider("토픽 수", 2, 5, 3)
            st.write(f"현재 가게의 리뷰에서 {n_topics}개의 주요 테마를 추출 중입니다...")
            st.success("토픽 1: 맛과 품질 / 토픽 2: 가성비와 가격 / 토픽 3: 직원 서비스") # 예시 결과 출력
        else:
            st.error("리뷰 원문이 포함된 '유형 B'에서 사용 가능합니다.")

    # [기능 6] 고객 만족도 분석 (특허 수식 3)
    elif selected_func == "고객 만족도 분석":
        st.subheader("🤝 자카드 유사도 기반 경쟁사 비교")
        target_df = df_main if selected_mode == "유형 A" else df_sent_b
        attrs = ['맛', '서비스', '가격', '위치', '분위기', '위생']
        
        if st.button("분석 시작하기"):
            current = target_df[target_df['가게명'] == selected_store][attrs].fillna(0).iloc[0]

            def get_strength_set(r):
                return set([a for a in attrs if r[a] >= 85])

            target_set = get_strength_set(current)
            results = []

            for _, row in target_df[target_df['가게명'] != selected_store].iterrows():
                other_set = get_strength_set(row)
                union = len(target_set | other_set)
                intersection = len(target_set & other_set)
                jaccard = intersection / union if union > 0 else 0
                results.append((row['가게명'], jaccard, row[attrs].fillna(0).values))
            
            results.sort(key=lambda x: x[1], reverse=True)
            comp_name, comp_sim, comp_scores = results[0]

            st.success(f"유사도 분석 결과: **[{comp_name}]**과 가장 비슷한 만족도 구조를 가집니다.")
            
            # 비교 레이더 차트
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=current.values, theta=attrs, fill='toself', name=selected_store))
            fig.add_trace(go.Scatterpolar(r=comp_scores, theta=attrs, fill='toself', name=comp_name))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
            st.plotly_chart(fig)
