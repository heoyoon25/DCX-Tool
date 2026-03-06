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

# --- 2. 초강력 데이터 전처리 및 로드 함수 ---
def robust_standardize(df):
    if df is None: return None
    # 컬럼명 공백 제거 및 이름 통일
    df.columns = [str(c).strip() for c in df.columns]
    
    # 가게 이름 관련 컬럼 찾기
    name_cols = ['가게명', 'restaurant_name', '가게 명', '가게', 'name', '업소명']
    for col in name_cols:
        if col in df.columns:
            df.rename(columns={col: '가게명'}, inplace=True)
            break
    else:
        # 못 찾으면 첫 번째 컬럼을 가게명으로 간주
        df.rename(columns={df.columns[0]: '가게명'}, inplace=True)
    
    # 가게명 데이터 자체의 공백 제거 (병합 성공률 극대화)
    df['가게명'] = df['가게명'].astype(str).str.strip()
    return df

@st.cache_data
def load_data_optimized(mode):
    try:
        if mode == "유형 A":
            df_ana = robust_standardize(pd.read_parquet('IBA-DCX_Analytics_2.0_PNU.parquet'))
            df_sent = robust_standardize(pd.read_parquet('PNUsentiment(유형A).parquet'))
            
            # 병합 (가게명 기준)
            merged = pd.merge(df_ana, df_sent, on='가게명', how='inner')
            
            # 만약 병합 후 데이터가 0개라면 가게명이 매칭 안 된 것 -> outer join으로 시도
            if len(merged) == 0:
                merged = pd.merge(df_ana, df_sent, on='가게명', how='outer')
            return merged
        else:
            df_rev = robust_standardize(pd.read_parquet('PNU_reviews.parquet'))
            df_sent_b = robust_standardize(pd.read_parquet('PNUsentiment(유형B).parquet'))
            return df_rev, df_sent_b
    except Exception as e:
        st.error(f"⚠️ 파일 로드 중 오류 발생: {e}")
        return None

# --- 3. 사이드바 메뉴 ---
st.sidebar.title("🔍 PNU 분석 대시보드")
selected_mode = st.sidebar.selectbox("1. 분석 유형 선택", ["유형 A", "유형 B"])

data_source = load_data_optimized(selected_mode)

if data_source is not None:
    # 데이터 할당 및 가게 목록 생성
    if selected_mode == "유형 A":
        df_main = data_source
        store_list = sorted(df_main['가게명'].unique())
    else:
        df_reviews, df_sent_b = data_source
        store_list = sorted(df_sent_b['가게명'].unique())

    selected_store = st.sidebar.selectbox("2. 가게 선택", store_list)
    selected_func = st.sidebar.radio("3. 기능 선택", 
        ["리뷰 요약", "워드클라우드", "트리맵", "네트워크 분석", "토픽 모델링", "고객 만족도 분석"])

    st.title(f"🏠 {selected_store} 상세 분석")
    
    # [디버깅 도구] 데이터가 안 나올 때를 대비한 정보창 (평소엔 닫혀있음)
    with st.expander("🛠️ 데이터 컬럼 상태 확인 (문제가 있을 때만 열어보세요)"):
        if selected_mode == "유형 A":
            st.write("사용 가능한 컬럼:", df_main.columns.tolist())
        else:
            st.write("감성 데이터 컬럼:", df_sent_b.columns.tolist())

    st.markdown("---")

    # --- 4. 기능별 상세 구현 ---
    
    # 분석에 필요한 핵심 속성 정의
    target_cols = ['맛', '서비스', '가격', '위치', '분위기', '위생']

    # 해당 가게 데이터 추출
    if selected_mode == "유형 A":
        store_data = df_main[df_main['가게명'] == selected_store]
    else:
        store_data = df_sent_b[df_sent_b['가게명'] == selected_store]

    # [기능 1] 리뷰 요약
    if selected_func == "리뷰 요약":
        if selected_mode == "유형 B":
            revs = df_reviews[df_reviews['가게명'] == selected_store]
            if not revs.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("총 리뷰 수", len(revs))
                c2.metric("총 이미지 수", revs.get('photo_count', pd.Series([0])).sum())
                # 리뷰 텍스트 컬럼 자동 찾기
                text_col = 'review_text' if 'review_text' in revs.columns else revs.columns[1] 
                c3.metric("평균 리뷰 길이", int(revs[text_col].str.len().mean()))
                
                st.subheader("📌 대표 리뷰 (최상위 3개)")
                for _, r in revs.head(3).iterrows():
                    st.info(r[text_col])
            else:
                st.warning("이 가게의 리뷰 텍스트 데이터가 없습니다.")
        else:
            st.warning("리뷰 상세 분석은 '유형 B'에서만 지원합니다.")

    # [기능 2] 워드클라우드
    elif selected_func == "워드클라우드":
        st.subheader("☁️ 속성별 감성 점수 워드클라우드")
        if not store_data.empty:
            # 존재하는 컬럼만 추출하여 딕셔너리 생성
            available_cols = [c for c in target_cols if c in store_data.columns]
            scores = store_data[available_cols].iloc[0].to_dict()
            # 0보다 큰 점수만 필터링
            wc_data = {k: float(v) for k, v in scores.items() if pd.notna(v) and v > 0}
            
            if wc_data:
                wc = WordCloud(font_path=FONT_PATH, background_color='white', width=800, height=400).generate_from_frequencies(wc_data)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.error("⚠️ 감성 점수 데이터가 모두 0이거나 비어있습니다. (데이터를 확인해주세요)")
        else:
            st.error("가게 데이터를 찾을 수 없습니다.")

    # [기능 6] 고객 만족도 분석 (자카드 유사도)
    elif selected_func == "고객 만족도 분석":
        st.subheader("🤝 자카드 유사도 기반 경쟁사 비교")
        if st.button("만족도 분석 시작하기"):
            df_comp = df_main if selected_mode == "유형 A" else df_sent_b
            available_cols = [c for c in target_cols if c in df_comp.columns]
            
            if not store_data.empty and available_cols:
                current_scores = store_data[available_cols].iloc[0].fillna(0)
                
                # 자카드 유사도 계산 (85점 이상을 강점으로 정의)
                def get_strength_set(row):
                    return set([c for c in available_cols if row[c] >= 85])
                
                target_set = get_strength_set(current_scores)
                
                sim_list = []
                for _, row in df_comp[df_comp['가게명'] != selected_store].iterrows():
                    other_set = get_strength_set(row)
                    union = len(target_set | other_set)
                    intersection = len(target_set & other_set)
                    jaccard = intersection / union if union > 0 else 0
                    sim_list.append((row['가게명'], jaccard, row[available_cols].values))
                
                sim_list.sort(key=lambda x: x[1], reverse=True)
                if sim_list:
                    comp_name, j_val, comp_vals = sim_list[0]
                    st.success(f"분석 완료! 가장 유사한 경쟁사는 **[{comp_name}]** 입니다. (유사도: {j_val:.2f})")
                    
                    # 레이더 차트 비교
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(r=current_scores.values, theta=available_cols, fill='toself', name=selected_store))
                    fig.add_trace(go.Scatterpolar(r=comp_vals, theta=available_cols, fill='toself', name=comp_name))
                    st.plotly_chart(fig)
                else:
                    st.warning("비교할 경쟁사 데이터가 부족합니다.")
            else:
                st.error("분석할 감성 점수 컬럼이 존재하지 않습니다.")

# 나머지 트리맵, 네트워크, 토픽 기능도 위와 동일한 store_data 체크 로직을 추가하여 구현하면 됩니다.
