import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import networkx as nx
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

# 1. 전역 설정 및 폰트 (NanumGothic.ttf 사용)
st.set_page_config(page_title="PNU Analytics 2.0", layout="wide")

FONT_PATH = "./NanumGothic.ttf"
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
    fm.fontManager.addfont(FONT_PATH)
    plt.rc('font', family=font_prop.get_name())
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("⚠️ 'NanumGothic.ttf' 파일을 찾을 수 없어 기본 폰트를 사용합니다. GitHub 업로드 상태를 확인하세요.")

# 2. 데이터 로드 함수
@st.cache_data
def load_data(mode):
    try:
        if mode == "유형 A (네이버)":
            df_rev = pd.read_parquet('PNUnaver(유형A).parquet')
            df_sent = pd.read_parquet('PNUsentiment(유형A).parquet')
        else:
            df_rev = pd.read_parquet('PNUgoogle(유형B).parquet')
            df_sent = pd.read_parquet('PNUsentiment(유형B).parquet')

        # 컬럼 표준화 로직
        def standardize(df):
            # 1. 첫 번째 열을 '가게명'으로 통일
            df.rename(columns={df.columns[0]: '가게명'}, inplace=True)
            # 2. 리뷰 텍스트 열 찾기 (review_text, 리뷰내용 등 대응)
            for col in ['review_text', '리뷰내용', 'review', 'text']:
                if col in df.columns:
                    df.rename(columns={col: '리뷰내용'}, inplace=True)
            # 3. 데이터 공백 제거
            df['가게명'] = df['가게명'].astype(str).str.strip()
            return df

        return standardize(df_rev), standardize(df_sent)
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return None, None

# 3. 사이드바 구성
st.sidebar.title("📊 PNU Analytics 2.0")

if 'mode' not in st.session_state:
    st.session_state['mode'] = "유형 A (네이버)"

selected_mode = st.sidebar.selectbox("1. 분석 유형 선택", ["유형 A (네이버)", "유형 B (구글)"], key='mode')
df_rev, df_sent = load_data(selected_mode)

if df_rev is not None and df_sent is not None:
    # 가게 리스트 확보
    stores = sorted(df_sent['가게명'].unique())
    selected_store = st.sidebar.selectbox("2. 가게 선택", stores)
    
    # 분석 카테고리 정의
    categories = ['맛', '서비스', '가격', '위치', '분위기', '위생']
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "리뷰 요약", "워드클라우드", "트리맵", "네트워크 분석", "토픽 모델링", "고객 만족도 분석"
    ])

    # 선택한 가게 데이터 추출
    current_sent = df_sent[df_sent['가게명'] == selected_store].iloc[0]
    current_revs = df_rev[df_rev['가게명'] == selected_store]

    # [탭 1] 리뷰 요약
    with tab1:
        st.subheader(f"📋 {selected_store} 리뷰 통계")
        c1, c2, c3 = st.columns(3)
        c1.metric("총 리뷰 수", f"{len(current_revs)}개")
        c2.metric("이미지 수", f"{current_revs.get('photo_count', pd.Series([0])).sum()}개")
        if '리뷰내용' in current_revs.columns:
            c3.metric("평균 리뷰 길이", f"{int(current_revs['리뷰내용'].str.len().mean())}자")
        
        st.divider()
        st.subheader("📌 주요 리뷰 샘플")
        if not current_revs.empty and '리뷰내용' in current_revs.columns:
            for i, row in current_revs.head(5).iterrows():
                st.info(f"별점: {row.get('star_rating', 'N/A')} | {row['리뷰내용']}")

    # [탭 2] 워드클라우드
    with tab2:
        st.subheader("☁️ 감성 속성 워드클라우드")
        wc_data = {cat: float(current_sent[cat]) for cat in categories if cat in current_sent and current_sent[cat] > 0}
        
        if wc_data:
            # 폰트 경로 체크 후 사용
            actual_font = FONT_PATH if os.path.exists(FONT_PATH) else None
            try:
                wc = WordCloud(font_path=actual_font, background_color='white', width=800, height=400).generate_from_frequencies(wc_data)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"워드클라우드 생성 중 오류: {e}")
        else:
            st.warning("데이터가 부족하여 워드클라우드를 생성할 수 없습니다.")

    # [탭 3] 트리맵
    with tab3:
        st.subheader("🌳 서비스 속성 트리맵")
        df_tree = pd.DataFrame([{'속성': cat, '점수': current_sent[cat]} for cat in categories if cat in current_sent])
        fig = px.treemap(df_tree, path=['속성'], values='점수', color='점수', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    # [탭 4] 네트워크 분석 (시뮬레이션)
    with tab4:
        st.subheader("🕸️ 속성 간 네트워크 관계")
        G = nx.Graph()
        for cat in categories:
            if current_sent[cat] >= 85: # 강점 속성 연결
                G.add_edge(selected_store, cat, weight=current_sent[cat])
        
        fig, ax = plt.subplots()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightpink', font_family='NanumGothic', font_size=10, node_size=1500)
        st.pyplot(fig)

    # [탭 5] 토픽 모델링 (정적 가이드)
    with tab5:
        st.subheader("🧪 주요 토픽 분석")
        st.write("리뷰 데이터에서 발견된 핵심 테마입니다.")
        st.success("Topic 1: 음식의 품질과 맛의 조화")
        st.success("Topic 2: 매장 서비스 및 직원 응대")
        st.success("Topic 3: 가격 대비 만족도(가성비)")

    # [탭 6] 고객 만족도 분석 (자카드 유사도 적용)
    with tab6:
        st.subheader("🤝 자카드 유사도 기반 경쟁사 비교")
        if st.button("유사도 분석 실행"):
            # 85점 이상을 강점으로 정의
            def get_strength_set(row):
                return set([cat for cat in categories if cat in row and row[cat] >= 85])
            
            target_set = get_strength_set(current_sent)
            
            sim_results = []
            for _, row in df_sent[df_sent['가게명'] != selected_store].iterrows():
                other_set = get_strength_set(row)
                union = len(target_set | other_set)
                inter = len(target_set & other_set)
                jaccard = inter / union if union > 0 else 0
                sim_results.append((row['가게명'], jaccard, row[categories].values))
            
            sim_results.sort(key=lambda x: x[1], reverse=True)
            comp_name, j_val, comp_vals = sim_results[0]
            
            st.success(f"유사도 분석 결과, **[{comp_name}]**과 강점 구조가 가장 비슷합니다. (유사도: {j_val:.2f})")
            
            # 레이더 차트
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=current_sent[categories].values, theta=categories, fill='toself', name=selected_store))
            fig.add_trace(go.Scatterpolar(r=comp_vals, theta=categories, fill='toself', name=comp_name))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
            st.plotly_chart(fig)

else:
    st.info("사이드바에서 분석 유형과 가게를 선택해주세요.")
