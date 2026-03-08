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
import streamlit.components.v1 as components
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# 1. 전역 설정 및 폰트
st.set_page_config(page_title="PNU Analytics 2.0", layout="wide")

FONT_PATH = "./NanumGothic.ttf"
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
    fm.fontManager.addfont(FONT_PATH)
    plt.rc('font', family=font_prop.get_name())
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("⚠️ 'NanumGothic.ttf' 파일이 없습니다. 한글이 깨질 수 있습니다.")

# [새로 추가된 핵심 로직] 카테고리별 유의어 사전 (이 단어들이 포함되면 해당 토픽으로 분류)
CATEGORY_KEYWORDS = {
    '맛': ['맛', '존맛', '음식', '메뉴', '식사', '먹', '달콤', '매콤', '짜', '싱거', '꿀맛', 'JMT'],
    '서비스': ['서비스', '친절', '직원', '사장', '알바', '응대', '불친절', '태도', '서비스가'],
    '가격': ['가격', '가성비', '비싸', '저렴', '싸다', '돈', '금액', '비싼', '싼', '값'],
    '위치': ['위치', '주차', '역', '접근성', '가깝', '멀다', '거리', '주차장', '골목', '교통'],
    '분위기': ['분위기', '인테리어', '데이트', '조명', '음악', '조용', '시끄', '감성', '뷰', '경치'],
    '위생': ['위생', '깨끗', '청결', '더럽', '냄새', '화장실', '벌레', '머리카락', '지저분']
}

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
    store_texts = current_revs['리뷰내용'].tolist() if not current_revs.empty and '리뷰내용' in current_revs.columns else []

    # [탭 1] 리뷰 요약 (더보기 버튼 추가)
    with tab1:
        st.subheader(f"📋 {selected_store} 리뷰 요약")
        c1, c2, c3 = st.columns(3)
        c1.metric("총 리뷰 수", f"{len(current_revs)}개")
        c2.metric("이미지 수", f"{current_revs.get('photo_count', pd.Series([0])).sum()}개")
        if store_texts:
            c3.metric("평균 리뷰 길이", f"{int(current_revs['리뷰내용'].str.len().mean())}자")
            st.divider()
            
            # 상위 5개 먼저 보여주기
            st.markdown("#### 📌 최근 리뷰 샘플")
            for i, row in current_revs.head(5).iterrows():
                st.info(f"⭐ 별점: {row.get('star_rating', '-')} | {row['리뷰내용']}")
            
            # [추가된 기능] 전체 리뷰 더보기 (Expander)
            if len(current_revs) > 5:
                with st.expander("🔍 전체 리뷰 더 보기 (클릭하여 펼치기)"):
                    for i, row in current_revs.iloc[5:].iterrows():
                        st.markdown(f"**⭐ {row.get('star_rating', '-')}** | {row['리뷰내용']}")
                        st.divider()

    # [탭 2] 워드클라우드 (유의어 사전 필터링 적용)
    with tab2:
        st.subheader("☁️ 리뷰 키워드 워드클라우드")
        topic_filter = st.selectbox("분석 기준 선택", ["전체 컨텐츠"] + categories, key="wc_filter")
        
        if store_texts:
            filtered_texts = []
            if topic_filter == "전체 컨텐츠":
                filtered_texts = store_texts
            else:
                # 유의어 사전을 기반으로 필터링
                keywords = CATEGORY_KEYWORDS.get(topic_filter, [topic_filter])
                for t in store_texts:
                    if any(k in str(t) for k in keywords):
                        filtered_texts.append(t)
            
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
                st.warning(f"'{topic_filter}'(으)로 분류될 만한 리뷰 텍스트가 존재하지 않습니다.")
        else:
            st.warning("리뷰 텍스트 데이터가 없습니다.")

    # [탭 3] 트리맵 (유의어 사전 필터링 적용)
    with tab3:
        st.subheader("🌳 핵심 키워드 트리맵")
        tm_filter = st.selectbox("분석 기준 선택", ["전체 컨텐츠"] + categories, key="tm_filter")
        
        if store_texts:
            filtered_texts = []
            if tm_filter == "전체 컨텐츠":
                filtered_texts = store_texts
            else:
                keywords = CATEGORY_KEYWORDS.get(tm_filter, [tm_filter])
                for t in store_texts:
                    if any(k in str(t) for k in keywords):
                        filtered_texts.append(t)
                        
            words = get_words(filtered_texts)
            top_words = Counter(words).most_common(20)
            
            if top_words:
                df_tree = pd.DataFrame(top_words, columns=['키워드', '빈도수'])
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
            co_occurrence = Counter()
            for text in store_texts:
                words = set(get_words([text]))
                for w1, w2 in itertools.combinations(words, 2):
                    co_occurrence[tuple(sorted([w1, w2]))] += 1
            
            G = nx.Graph()
            for (w1, w2), freq in co_occurrence.items():
                if freq >= min_freq:
                    G.add_edge(w1, w2, weight=freq)
            
            if len(G.nodes) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                pos = nx.spring_layout(G, k=0.5)
                weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
                nx.draw(G, pos, with_labels=True, node_color='lightgreen', font_family='NanumGothic',
                        node_size=1000, font_size=10, width=weights, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("선택한 빈도수를 만족하는 단어 연결이 없습니다. 빈도수를 낮춰보세요.")

    # [탭 5] 토픽 모델링
    with tab5:
        st.subheader("🧪 LDA(Latent Dirichlet Allocation) 토픽 모델링")
        num_topics = st.slider("추출할 토픽 수", 2, 5, 3)
        
        if st.button("토픽 모델링 실행") and store_texts:
            with st.spinner("시각화 맵을 구성 중입니다..."):
                docs = [get_words([t]) for t in store_texts]
                docs = [d for d in docs if len(d) > 0]
                
                if len(docs) > 0:
                    dictionary = corpora.Dictionary(docs)
                    corpus = [dictionary.doc2bow(text) for text in docs]
                    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
                    
                    st.markdown("#### 📌 토픽별 핵심 단어 요약")
                    cols = st.columns(num_topics)
                    for i in range(num_topics):
                        words = lda_model.show_topic(i, topn=5)
                        with cols[i]:
                            st.markdown(f"**🔥 Topic {i+1}**")
                            for word, prob in words:
                                st.write(f"- {word} ({prob:.2f})")
                    st.divider()
                    
                    st.markdown("#### 🔍 인터랙티브 토픽 거리 지도")
                    try:
                        vis = gensimvis.prepare(lda_model, corpus, dictionary)
                        html_string = pyLDAvis.prepared_data_to_html(vis)
                        components.html(html_string, width=1300, height=800, scrolling=True)
                    except Exception as e:
                        st.error(f"시각화 오류: {e}")
                else:
                    st.warning("분석할 유효 단어가 부족합니다.")

    # [탭 6] 고객 만족도 분석
    with tab6:
        st.subheader("📈 감성 점수 기반 고객 만족도 분석")
        
        regional_avg = df_sent[categories].mean().fillna(0)
        store_scores = current_sent[categories].fillna(0)
        
        df_comp = pd.DataFrame({
            '항목': categories,
            f'[{selected_store}] 점수': store_scores.values.astype(float),
            '지역 평균': regional_avg.values.astype(float)
        })
        
        fig_bar = px.bar(df_comp, x='항목', y=[f'[{selected_store}] 점수', '지역 평균'], barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        st.markdown("##### 🤝 자카드 유사도 기반 경쟁사 매칭 (특허 수식 3 적용)")
        if st.button("자카드 유사도 분석 실행"):
            def get_strength_set(row):
                return set([cat for cat in categories if cat in row and float(row[cat]) >= 85])
            
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
                    '강점 교집합': list(target_set & other_set)
                })
            
            df_sim = pd.DataFrame(sim_results).sort_values(by='자카드 유사도', ascending=False)
            top_competitors = df_sim[df_sim['가게명'] != selected_store].head(3)
            
            if not top_competitors.empty:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.success(f"🏆 최우수 유사 경쟁사\n\n**{top_competitors.iloc[0]['가게명']}**\n(유사도: {top_competitors.iloc[0]['자카드 유사도']})")
                    st.write(f"🤝 겹치는 강점: {', '.join(top_competitors.iloc[0]['강점 교집합'])}")
                
                with c2:
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(r=current_sent[categories].values.astype(float), theta=categories, fill='toself', name=f'{selected_store}'))
                    
                    for _, comp in top_competitors.iterrows():
                        comp_scores = df_sent[df_sent['가게명'] == comp['가게명']][categories].iloc[0]
                        fig_radar.add_trace(go.Scatterpolar(r=comp_scores.values.astype(float), theta=categories, fill='toself', name=comp['가게명']))
                        
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
                    st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.warning("비교할 경쟁사가 없습니다.")

else:
    st.info("사이드바에서 설정해주세요.")
