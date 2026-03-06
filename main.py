import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import os

# --- 1. 환경 설정 및 폰트 로드 ---
FONT_PATH = "./NanumGothic-Regular.ttf" 
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rc('font', family=font_prop.get_name())
    plt.rcParams['axes.unicode_minus'] = False

# --- 2. 데이터 로드 및 컬럼 강제 통일 함수 ---
@st.cache_data
def load_combined_data(mode):
    try:
        if mode == "유형 A":
            df_ana = pd.read_parquet('IBA-DCX_Analytics_2.0_PNU.parquet')
            df_sent_a = pd.read_parquet('PNUsentiment(유형A).parquet')
            
            # 존재하는 모든 가능성 있는 컬럼명을 '가게명'으로 변경
            for df in [df_ana, df_sent_a]:
                if 'restaurant_name' in df.columns:
                    df.rename(columns={'restaurant_name': '가게명'}, inplace=True)
                elif '가게' in df.columns: # '가게'라고만 되어있을 경우 대비
                    df.rename(columns={'가게': '가게명'}, inplace=True)
                
            return pd.merge(df_ana, df_sent_a, on='가게명', how='inner')
            
        else:
            df_rev = pd.read_parquet('PNU_reviews.parquet')
            df_sent_b = pd.read_parquet('PNUsentiment(유형B).parquet')
            
            # 유형 B 파일들도 컬럼명 강제 통일
            for df in [df_rev, df_sent_b]:
                if 'restaurant_name' in df.columns:
                    df.rename(columns={'restaurant_name': '가게명'}, inplace=True)
                elif '가게' in df.columns:
                    df.rename(columns={'가게': '가게명'}, inplace=True)
                
            return df_rev, df_sent_b
    except Exception as e:
        # 에러 발생 시 어떤 컬럼들이 있는지 화면에 출력하여 디버깅 도와줌
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return None

# --- 3. 메인 화면 ---
st.sidebar.title("📊 PNU 분석 시스템 2.0")
selected_mode = st.sidebar.selectbox("1. 분석 유형 선택", ["유형 A", "유형 B"])

data_source = load_combined_data(selected_mode)

if data_source is not None:
    if selected_mode == "유형 A":
        df_main = data_source
        store_list = df_main['가게명'].unique()
    else:
        df_reviews, df_sent_b = data_source
        # 여기서도 '가게명' 컬럼이 확실히 존재하는지 확인 후 리스트 생성
        store_list = df_sent_b['가게명'].unique()

    selected_store = st.sidebar.selectbox("2. 가게 선택", store_list)
    selected_func = st.sidebar.radio("3. 기능 선택", 
        ["리뷰 요약", "워드클라우드", "트리맵", "네트워크 분석", "토픽 모델링", "고객 만족도 분석"])

    st.title(f"🏠 {selected_store} 상세 분석")
    st.markdown("---")

    # --- 4. 기능 구현 ---

    # [기능 1] 리뷰 요약
    if selected_func == "리뷰 요약":
        if selected_mode == "유형 B":
            target_rev = df_reviews[df_reviews['가게명'] == selected_store]
            c1, c2, c3 = st.columns(3)
            c1.metric("총 리뷰 수", f"{len(target_rev)}개")
            c2.metric("총 이미지 수", f"{target_rev['photo_count'].sum()}개")
            c3.metric("평균 리뷰 길이", f"{int(target_rev['review_text'].str.len().mean())}자")
            
            st.subheader("📌 주요 대표 리뷰")
            for i, row in target_rev.sort_values(by='star_rating', ascending=False).head(3).iterrows():
                with st.expander(f"⭐ {row['star_rating']}점 리뷰"):
                    st.write(row['review_text'])
        else:
            st.warning("리뷰 상세 데이터는 '유형 B'에서 확인 가능합니다.")

    # [기능 2] 워드클라우드
    elif selected_func == "워드클라우드":
        st.subheader("☁️ 감성 키워드 워드클라우드")
        cols = ['맛', '서비스', '가격', '위치', '분위기', '위생']
        row = df_main[df_main['가게명'] == selected_store].iloc[0] if selected_mode == "유형 A" else df_sent_b[df_sent_b['가게명'] == selected_store].iloc[0]
        
        # 값이 있는 속성만 추출하여 시각화
        freq = {c: int(row[c]) for c in cols if not pd.isna(row[c])}
        if freq:
            wc = WordCloud(font_path=FONT_PATH, background_color='white', width=800, height=400).generate_from_frequencies(freq)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("데이터가 부족하여 워드클라우드를 생성할 수 없습니다.")

    # [기능 6] 고객 만족도 분석 (특허 수식 3)
    elif selected_func == "고객 만족도 분석":
        st.subheader("🤝 자카드 유사도 기반 경쟁사 비교")
        if st.button("분석 시작하기"):
            target_df = df_main if selected_mode == "유형 A" else df_sent_b
            attrs = ['맛', '서비스', '가격', '위치', '분위기', '위생']
            
            # 현재 가게 점수 (NaN은 0으로 처리)
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

            st.success(f"유사도 분석 결과: **[{comp_name}]** (유사도: {comp_sim:.2f})")
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=current.values, theta=attrs, fill='toself', name=selected_store))
            fig.add_trace(go.Scatterpolar(r=comp_scores, theta=attrs, fill='toself', name=comp_name))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
            st.plotly_chart(fig)
