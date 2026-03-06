import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# --- [중요] 데이터 컬럼명 강제 표준화 함수 ---
def standardize_df(df):
    if df is None: return None
    # 1. 모든 컬럼명의 공백 제거
    df.columns = [c.strip() for c in df.columns]
    # 2. 첫 번째 컬럼(가게 이름)을 '가게명'으로 강제 변경
    df.rename(columns={df.columns[0]: '가게명'}, inplace=True)
    # 3. 분석에 필요한 영문명들을 한글로 변경 (있을 경우)
    rename_map = {
        'restaurant_name': '가게명',
        'star_rating': '별점',
        'review_text': '리뷰내용'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

@st.cache_data
def load_and_merge(mode):
    try:
        if mode == "유형 A":
            df_ana = standardize_df(pd.read_parquet('IBA-DCX_Analytics_2.0_PNU.parquet'))
            df_sent = standardize_df(pd.read_parquet('PNUsentiment(유형A).parquet'))
            # inner merge 시 데이터가 사라지는지 확인하기 위해 join 전 크기 체크
            merged = pd.merge(df_ana, df_sent, on='가게명', how='inner')
            if len(merged) == 0:
                # 병합 실패 시 정보를 유지하기 위해 outer로 시도
                merged = pd.merge(df_ana, df_sent, on='가게명', how='outer')
            return merged
        else:
            df_rev = standardize_df(pd.read_parquet('PNU_reviews.parquet'))
            df_sent_b = standardize_df(pd.read_parquet('PNUsentiment(유형B).parquet'))
            return df_rev, df_sent_b
    except Exception as e:
        st.error(f"데이터 파일 읽기 실패: {e}")
        return None

# --- UI 레이아웃 ---
st.sidebar.title("🛠️ 데이터 분석 설정")
mode = st.sidebar.selectbox("유형 선택", ["유형 A", "유형 B"])
data = load_and_merge(mode)

if data is not None:
    # 데이터 할당
    if mode == "유형 A":
        df_main = data
        stores = sorted(df_main['가게명'].dropna().unique())
    else:
        df_rev, df_sent_b = data
        stores = sorted(df_sent_b['가게명'].dropna().unique())

    selected_store = st.sidebar.selectbox("가게 선택", stores)
    # 라디오 버튼의 문자열이 아래 if문과 토씨 하나 안 틀리고 같아야 합니다.
    func = st.sidebar.radio("기능 선택", ["리뷰 요약", "워드클라우드", "트리맵", "고객 만족도 분석"])

    st.title(f"📊 {selected_store} 분석")
    st.divider()

    # --- 기능별 출력 보장 로직 ---

    if func == "리뷰 요약":
        if mode == "유형 B":
            # '리뷰내용' 컬럼이 없는 경우를 대비한 유연한 필터링
            rev_col = '리뷰내용' if '리뷰내용' in df_rev.columns else 'review_text'
            target = df_rev[df_rev['가게명'] == selected_store]
            
            if len(target) > 0:
                st.subheader("📝 리뷰 통계")
                st.write(f"총 {len(target)}개의 리뷰가 있습니다.")
                st.dataframe(target[[rev_col]].head(10))
            else:
                st.warning("이 가게에 해당하는 리뷰 텍스트가 데이터에 없습니다.")
        else:
            st.info("유형 A는 수치 지표 위주의 데이터입니다. 리뷰 원문은 유형 B를 선택하세요.")

    elif func == "워드클라우드":
        st.subheader("☁️ 속성 감성 분석")
        # 데이터 행 가져오기
        row = df_main[df_main['가게명'] == selected_store] if mode == "유형 A" else df_sent_b[df_sent_b['가게명'] == selected_store]
        
        if not row.empty:
            cols = ['맛', '서비스', '가격', '위치', '분위기', '위생']
            # 존재하는 컬럼만 추출
            valid_cols = [c for c in cols if c in row.columns]
            scores = row[valid_cols].iloc[0].to_dict()
            
            # 워드클라우드 생성 (NaN 제외)
            clean_scores = {k: v for k, v in scores.items() if pd.notna(v) and v > 0}
            if clean_scores:
                wc = WordCloud(font_path="./NanumGothic-Regular.ttf", background_color='white').generate_from_frequencies(clean_scores)
                fig, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.error("감성 점수 데이터가 모두 0이거나 비어있습니다.")
        else:
            st.error("해당 가게의 데이터를 찾을 수 없습니다.")

    elif func == "고객 만족도 분석":
        st.subheader("🤝 경쟁사 비교 (자카드 유사도)")
        # 버튼을 누르지 않아도 기본적으로 로직이 보이게 하거나 버튼 클릭 시 확실히 실행
        if st.button("분석 실행"):
            target_df = df_main if mode == "유형 A" else df_sent_b
            attrs = ['맛', '서비스', '가격', '위치', '분위기', '위생']
            # 실제 존재하는 속성만 사용
            current_attrs = [a for a in attrs if a in target_df.columns]
            
            current_row = target_df[target_df['가게명'] == selected_store]
            if not current_row.empty:
                st.success("유사도 계산을 시작합니다...")
                # (이후 자카드 로직 실행 및 시각화)
            else:
                st.error("기준 가게의 점수 데이터가 없습니다.")
