import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import re
from collections import Counter
import itertools
import networkx as nx
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from gensim import corpora
from gensim.models import LdaModel

# 1. 전역 설정 및 폰트 (NanumGothic.ttf)
st.set_page_config(page_title="PNU Analytics 2.0", layout="wide")

FONT_PATH = "./NanumGothic.ttf"
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
    fm.fontManager.addfont(FONT_PATH)
    plt.rc('font', family=font_prop.get_name())
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("⚠️ 'NanumGothic.ttf' 파일이 없습니다. 한글이 깨질 수 있습니다.")

# 2. 데이터 로드 및 텍스트 전처리 도구
@st.cache_data
def load_data(mode):
    try:
        if mode == "유형 A":
            df_rev = pd.read_parquet('PNUnaver(유형A).parquet')
            df_sent = pd.read_parquet('PNUsentiment(유형A).parquet')
        else:
            df_rev = pd.read_parquet('PNUgoogle(유형B).parquet')
            df_sent = pd.read_parquet('PNUsentiment(유형B).parquet')

        def standardize(df):
            df.rename(columns={df.columns[0]: '가게명'}, inplace=True)
            for col in ['review_text', '리뷰내용', 'review', 'text']:
                if col in df.columns:
                    df.rename(columns={col: '리뷰내용'}, inplace=True)
            df['가게명'] = df['가게명'].astype(str).str.strip()
            return df

        return standardize(df_rev), standardize(df_sent)
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return None, None

def get_words(texts):
    """간단한 텍스트 명사/형용사 추출 (형태소 분석기 대체)"""
    words = []
    for t in texts:
        if pd.notna(t):
            clean = re.sub(r'[^가-힣\s]', '', str(t))
            words.extend([w for w in clean.split() if len(w) >= 2])
    return words

# 3. 사이드바 구성
st.sidebar.title("📊 PNU Analytics 2.0")

if 'mode' not in st.session_state:
    st.session_state['mode'] = "유형 A"

# "유형 A", "유형 B"로 텍스트 간소화
selected_mode = st.sidebar.selectbox("1. 분석 유형 선택", ["유형 A", "유형 B"], key='mode')
df_rev, df_sent = load_data(selected_mode)

if df_rev is not None and df_sent is not None:
    stores = sorted(df_sent['가게명'].unique())
    selected_store = st.sidebar.selectbox("2. 가게 선택", stores)
    
    categories = ['맛', '서비스', '가격', '위치', '분위기', '위생']
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "리뷰 요약", "워드클라우드", "트리맵", "네트워크 분석", "토픽 모델링", "고객 만족도 분석"
    ])

    current_sent = df_sent[df_sent['가게명'] == selected_store].iloc[0]
    current_revs = df_rev[df_rev['가게명'] == selected_store]
    
    # 해당 가게의 전체 단어 풀 확보
    store_texts = current_revs['리뷰내용'].tolist() if not current_revs.empty and '리뷰내용' in current_revs.columns else []

    # [탭 1] 리뷰 요약
    with tab1:
        st.subheader(f"📋 {selected_store} 리뷰 요약")
        c1, c2, c3 = st.columns(3)
        c1.metric("총 리뷰 수", f"{len(current_revs)}개")
        c2.metric("이미지 수", f"{current_revs.get('photo_count', pd.Series([0])).sum()}개")
        if store_texts:
            c3.metric("평균 리뷰 길이", f"{int(current_revs['리뷰내용'].str.len().mean())}자")
            st.divider()
            for i, row in current_revs.head(5).iterrows():
                st.info(f"⭐ {row.get('star_rating', '-')} | {row['리뷰내용']}")

    # [탭 2] 워드클라우드 (빈도수 기반)
    with tab2:
        st.subheader("☁️ 리뷰 키워드 워드클라우드")
        topic_filter = st.selectbox("분석 기준 선택", ["전체 컨텐츠"] + categories, key="wc_filter")
        
        if store_texts:
            # 토픽 필터링 로직: 특정 카테고리 선택 시 해당 단어가 포함된 리뷰만 분석
            filtered_texts = store_texts
            if topic_filter != "전체 컨텐츠":
                filtered_texts = [t for t in store_texts if topic_filter in str(t)]
            
            words = get_words(filtered_texts)
            word_counts = Counter(words)
            
            if word_counts:
                actual_font = FONT_PATH if os.path.exists(FONT_PATH) else None
                wc = WordCloud(font_path=actual_font, background_color='white', width=800, height=400, 
                               colormap='Blues_r', max_words=100).generate_from_frequencies(word_counts)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning(f"'{topic_filter}' 관련 키워드를 리뷰에서 찾을 수 없습니다.")
        else:
            st.warning("리뷰 텍스트 데이터가 없습니다.")

    # [탭 3] 트리맵 (텍스트 빈도수 기반 타일형)
    with tab3:
        st.subheader("🌳 핵심 키워드 트리맵")
        tm_filter = st.selectbox("분석 기준 선택", ["전체 컨텐츠"] + categories, key="tm_filter")
        
        if store_texts:
            filtered_texts = store_texts if tm_filter == "전체 컨텐츠" else [t for t in store_texts if tm_filter in str(t)]
            words = get_words(filtered_texts)
            top_words = Counter(words).most_common(20) # 상위 20개 단어
            
            if top_words:
                df_tree = pd.DataFrame(top_words, columns=['키워드', '빈도수'])
                # 빈도가 높을수록 크고 진하게
                fig = px.treemap(df_tree, path=['키워드'], values='빈도수', color='빈도수', 
                                 color_continuous_scale='Teal', title=f"{selected_store} {tm_filter} 키워드")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("트리맵을 구성할 단어가 부족합니다.")

    # [탭 4] 네트워크 분석
    with tab4:
        st.subheader("🕸️ 단어 동시 출현 네트워크 분석")
        min_freq = st.slider("단어 등장 횟수 (최소 빈도)", 2, 20, 5)
        
        if store_texts:
            # 동시 출현(Co-occurrence) 행렬 생성
            co_occurrence = Counter()
            for text in store_texts:
                words = set(get_words([text]))
                for w1, w2 in itertools.combinations(words, 2):
                    if w1 != w2:
                        co_occurrence[tuple(sorted([w1, w2]))] += 1
            
            # 네트워크 그래프 생성
            G = nx.Graph()
            for (w1, w2), freq in co_occurrence.items():
                if freq >= min_freq:
                    G.add_edge(w1, w2, weight=freq)
            
            if len(G.nodes) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                pos = nx.spring_layout(G, k=0.5)
                # 엣지 굵기를 등장 횟수에 비례하게 설정
                weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
                nx.draw(G, pos, with_labels=True, node_color='lightgreen', font_family='NanumGothic',
                        node_size=1000, font_size=10, width=weights, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("선택한 빈도수를 만족하는 단어 연결이 없습니다. 빈도수를 낮춰보세요.")

    # [탭 5] 토픽 모델링 (LDA)
    with tab5:
        st.subheader("🧪 LDA(Latent Dirichlet Allocation) 토픽 모델링")
        num_topics = st.slider("추출할 토픽 수", 2, 5, 3)
        
        if st.button("토픽 모델링 실행") and store_texts:
            with st.spinner("텍스트 내 숨겨진 주제를 분석 중입니다..."):
                docs = [get_words([t]) for t in store_texts]
                dictionary = corpora.Dictionary(docs)
                corpus = [dictionary.doc2bow(text) for text in docs]
                
                if len(dictionary) > 0:
                    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
                    
                    # 결과 출력
                    cols = st.columns(num_topics)
                    for i in range(num_topics):
                        words = lda_model.show_topic(i, topn=5)
                        with cols[i]:
                            st.markdown(f"**🔥 Topic {i+1}**")
                            for word, prob in words:
                                st.write(f"- {word} ({prob:.2f})")
                else:
                    st.warning("분석할 단어가 충분하지 않습니다.")

    # [탭 6] 고객 만족도 분석 (평균 비교 + 자카드 유사도)
    with tab6:
        st.subheader("📈 감성 점수 기반 고객 만족도 분석")
        
        # 1. 지역 전체 평균 vs 해당 가게 비교 (강점 및 약점 분석)
        st.markdown("##### 1. 지역 평균 대비 경쟁력 분석 (SWOT)")
        regional_avg = df_sent[categories].mean()
        store_scores = current_sent[categories]
        
        # 바 차트 비교
        df_comp = pd.DataFrame({
            '항목': categories,
            f'[{selected_store}] 점수': store_scores.values,
            '지역 평균': regional_avg.values
        })
        fig_bar = px.bar(df_comp, x='항목', y=[f'[{selected_store}] 점수', '지역 평균'], barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # 2. 자카드 유사도 분석 구조
        st.markdown("##### 2. 자카드 유사도 기반 경쟁사 매칭")
        st.write("감성 점수 85점 이상인 항목을 **'강점 키워드'**로 정의하여, 우리 가게와 가장 유사한 경영 전략을 가진 경쟁사를 도출합니다.")
        
        if st.button("자카드 유사도 분석 실행"):
            def get_strength_set(row):
                return set([cat for cat in categories if cat in row and row[cat] >= 85])
            
            target_set = get_strength_set(current_sent)
            
            sim_results = []
            for _, row in df_sent[df_sent['가게명'] != selected_store].iterrows():
                other_set = get_strength_set(row)
                union = len(target_set | other_set)
                inter = len(target_set & other_set)
                jaccard = inter / union if union > 0 else 0
                sim_results.append({
                    '가게명': row['가게명'], 
                    '자카드 유사도': round(jaccard, 2), 
                    '강점 키워드 교집합': list(target_set & other_set)
                })
            
            # 결과 정렬 및 표시 (마지막 사진 2장의 구조 반영)
            df_sim = pd.DataFrame(sim_results).sort_values(by='자카드 유사도', ascending=False)
            top_competitors = df_sim[df_sim['가게명'] != selected_store].head(3)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.success(f"🏆 최우수 유사 경쟁사\n\n**{top_competitors.iloc[0]['가게명']}**\n(유사도: {top_competitors.iloc[0]['자카드 유사도']})")
                st.write(f"🤝 겹치는 강점: {', '.join(top_competitors.iloc[0]['강점 키워드 교집합'])}")
            
            with c2:
                # 상위 3개 가게 레이더 차트 비교
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=current_sent[categories].values, theta=categories, fill='toself', name=f'기준: {selected_store}'))
                
                for idx, comp in top_competitors.iterrows():
                    comp_scores = df_sent[df_sent['가게명'] == comp['가게명']][categories].iloc[0]
                    fig_radar.add_trace(go.Scatterpolar(r=comp_scores.values, theta=categories, fill='toself', name=comp['가게명']))
                    
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
                st.plotly_chart(fig_radar, use_container_width=True)

else:
    st.info("사이드바에서 분석 유형과 가게를 선택해주세요.")
