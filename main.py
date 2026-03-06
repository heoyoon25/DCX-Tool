import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# --- 1. 데이터 로드 및 컬럼 강제 통일 함수 ---
@st.cache_data
def load_combined_data(mode):
    try:
        if mode == "유형 A":
            # 파일 읽기
            df_ana = pd.read_parquet('IBA-DCX_Analytics_2.0_PNU.parquet')
            df_sent_a = pd.read_parquet('PNUsentiment(유형A).parquet')
            
            # [해결책] 모든 데이터프레임의 컬럼명을 검사하여 '가게'가 포함된 컬럼을 '가게명'으로 통일
            def rename_to_standard(df):
                # 1순위: '가게명'이나 'restaurant_name'이라는 정확한 이름이 있는지 확인
                possible_names = ['가게명', 'restaurant_name', '가게 명', '가게', 'name']
                for name in possible_names:
                    if name in df.columns:
                        return df.rename(columns={name: '가게명'})
                # 2순위: 그래도 없다면 '가게'라는 글자가 포함된 첫 번째 컬럼을 변경
                for col in df.columns:
                    if '가게' in col:
                        return df.rename(columns={col: '가게명'})
                # 3순위: 최후의 수단으로 0번째 컬럼이 이름이라고 가정
                return df.rename(columns={df.columns[0]: '가게명'})

            df_ana = rename_to_standard(df_ana)
            df_sent_a = rename_to_standard(df_sent_a)
            
            # 병합 전 컬럼 출력 (디버깅용 - 실제 실행 시 삭제 가능)
            # st.write("Ana Columns:", df_ana.columns.tolist())
            # st.write("Sent Columns:", df_sent_a.columns.tolist())

            # '가게명' 기준으로 병합
            return pd.merge(df_ana, df_sent_a, on='가게명', how='inner')
            
        else:
            # 유형 B 처리
            df_rev = pd.read_parquet('PNU_reviews.parquet')
            df_sent_b = pd.read_parquet('PNUsentiment(유형B).parquet')
            
            # 유형 B도 동일하게 컬럼 통일 로직 적용
            def rename_to_standard(df):
                possible_names = ['가게명', 'restaurant_name', '가게']
                for name in possible_names:
                    if name in df.columns:
                        return df.rename(columns={name: '가게명'})
                return df.rename(columns={df.columns[0]: '가게명'})

            return rename_to_standard(df_rev), rename_to_standard(df_sent_b)

    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return None

# --- 2. 메인 화면 ---
st.sidebar.title("📊 PNU 분석 시스템 2.0")
selected_mode = st.sidebar.selectbox("1. 분석 유형 선택", ["유형 A", "유형 B"])

data_source = load_combined_data(selected_mode)

if data_source is not None:
    # 데이터 구조 할당
    if selected_mode == "유형 A":
        df_main = data_source
        if '가게명' not in df_main.columns:
            st.error("병합된 데이터에 '가게명' 컬럼이 없습니다. 컬럼명을 다시 확인해주세요.")
            st.stop()
        store_list = df_main['가게명'].unique()
    else:
        df_reviews, df_sent_b = data_source
        store_list = df_sent_b['가게명'].unique()

    # 사이드바 2단계: 가게 선택
    selected_store = st.sidebar.selectbox("2. 가게 선택", store_list)
    selected_func = st.sidebar.radio("3. 기능 선택", 
        ["리뷰 요약", "워드클라우드", "트리맵", "고객 만족도 분석"])

    st.title(f"🏠 {selected_store} 상세 분석")
    st.markdown("---")

    # [기능: 고객 만족도 분석] 예시
    if selected_func == "고객 만족도 분석":
        st.subheader("🤝 자카드 유사도 경쟁사 분석")
        target_df = df_main if selected_mode == "유형 A" else df_sent_b
        attrs = ['맛', '서비스', '가격', '위치', '분위기', '위생']
        
        # 실제 데이터에 위 컬럼들이 있는지 확인 후 필터링
        existing_attrs = [a for a in attrs if a in target_df.columns]
        
        if st.button("분석 시작하기"):
            current = target_df[target_df['가게명'] == selected_store].iloc[0]
            
            # 유사도 계산 로직... (생략 - 이전 코드와 동일)
            st.success("유사도 분석이 완료되었습니다.")
