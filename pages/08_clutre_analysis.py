from __future__ import annotations
import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
import random
import hashlib
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# ──────────────────────────────────────────────────────────────────────────────
# Config & Paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CACHE_DIR = Path(".cache")
for d in (DATA_DIR, CACHE_DIR):
    d.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 0 · Kaggle Downloads (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def fetch_kaggle_csvs() -> None:
    """
    Download the Letterboxd CSVs once and save into data/.
    """
    files = ["movies.csv", "genres.csv", "themes.csv", "releases.csv", "countries.csv"]
    for fname in files:
        # if file already exists, skip
        out = DATA_DIR / fname
        if out.exists():
            continue
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "gsimonx37/letterboxd",
            fname
        )
        df.to_csv(out, index=False)

# Trigger download on first run only
fetch_kaggle_csvs()

# ──────────────────────────────────────────────────────────────────────────────
# 1 · Load & Cache Data
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    def read_csv(name: str) -> pd.DataFrame:
        path = DATA_DIR / name
        if not path.exists():
            st.error(f"Missing file: {path}")
            st.stop()
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if "id" in df.columns and "movie_id" not in df.columns:
            df = df.rename(columns={"id": "movie_id"})
        return df

    files = ["countries.csv", "genres.csv", "themes.csv", "releases.csv", "movies.csv"]
    data = {f.split('.')[0]: read_csv(f) for f in files}
    data["movies"]   = data["movies"].rename(columns={"name": "title", "date": "year"})
    data["releases"] = data["releases"].rename(columns={"country": "country_release"})
    data["releases"]["release_date"] = pd.to_datetime(data["releases"]["date"], errors="coerce")
    data["releases"]["release_year"] = data["releases"]["release_date"].dt.year
    return data

data = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# 2 · Sidebar: Filters & HCI
# ──────────────────────────────────────────────────────────────────────────────
DEC_META: dict[str, tuple[int, int, list[tuple[str, str]]]] = {
    "1920–1929 – Silent Era":    (1920, 1929, [("German Expressionism", "https://en.wikipedia.org/wiki/German_Expressionism"),
                                                ("Rise of Hollywood",     "https://en.wikipedia.org/wiki/History_of_Hollywood")]),
    "1930–1949 – Golden Age":     (1930, 1949, [("Golden Age of Hollywood", "https://en.wikipedia.org/wiki/Golden_Age_of_Hollywood"),
                                                ("Pre-/Post-war cinema",    "https://en.wikipedia.org/wiki/History_of_film")]),
    "1950–1969 – New Waves":      (1950, 1969, [("French New Wave",       "https://en.wikipedia.org/wiki/French_New_Wave"),
                                                ("Japanese New Wave",     "https://en.wikipedia.org/wiki/Japanese_New_Wave")]),
    "1970–1989 – Blockbusters":   (1970, 1989, [("New Hollywood",         "https://en.wikipedia.org/wiki/New_Hollywood"),
                                                ("Blockbuster era",       "https://en.wikipedia.org/wiki/Blockbuster_(entertainment)")]),
    "1990–2009 – Global Boom":    (1990, 2009, [("Nollywood",             "https://en.wikipedia.org/wiki/Cinema_of_Nigeria"),
                                                ("Asian cinema boom",     "https://en.wikipedia.org/wiki/Cinema_of_South_Korea")]),
    "2010–2025 – Streaming Era":  (2010, 2025, [("Streaming revolution",  "https://en.wikipedia.org/wiki/Streaming_media"),
                                                ("Hallyu wave",           "https://en.wikipedia.org/wiki/Hallyu")]),
}

st.set_page_config(
    page_title="🎬 Film Cultural Impact Explorer",
    layout="wide"
)
with st.sidebar:
    st.title("🎥 Film Impact Explorer")
    view = st.radio("🔍 Select View", ["Geo Map","Social Scatter","Cultural Embed","Historical"])
    search_query = st.text_input("🖊️ Search Titles/Descriptions")

    if view == "Historical":
        decade = st.selectbox("📆 Decade", list(DEC_META.keys()))
        yr_min, yr_max, cites = DEC_META[decade]
        st.markdown(f"**Years:** {yr_min} – {yr_max}")
        year_range = (yr_min, yr_max)
    else:
        yrs = data["releases"]["release_year"].dropna()
        year_range = st.slider("📅 Release Year Range",
                               int(yrs.min()), int(yrs.max()),
                               (int(yrs.min()), int(yrs.max())))

    with st.expander("📦 Metadata Filters"):
        genres_sel    = st.multiselect("🎭 Genres",
                                       ["All"] + sorted(data["genres"]["genre"].unique()),
                                       ["All"])
        themes_sel    = st.multiselect("🏷️ Themes",
                                       ["All"] + sorted(data["themes"]["theme"].unique()),
                                       ["All"])
        countries_sel = st.multiselect("🌍 Production Countries",
                                       ["All"] + sorted(data["countries"]["country"].unique()),
                                       ["All"])

    if view in ("Cultural Embed", "Historical"):
        with st.expander("🧮 Embedding Settings"):
            sample_size = st.slider("📏 Sample Size",  900, 100000, 4000, step=500)
            marker_size = st.slider("🖋️ Marker Size", 3, 20, 7)

    cmap = st.selectbox("🌈 Color Palette", ["Blues","Greens","Oranges","Purples"], index=0)

    if view == "Historical":
        st.markdown("---")
        st.markdown("### 📚 Further Reading")
        for txt, url in cites:
            st.markdown(f"* [{txt}]({url})")

# ──────────────────────────────────────────────────────────────────────────────
# 3 · Data Filtering + Download Button
# ──────────────────────────────────────────────────────────────────────────────
def filter_by(df_subset: pd.DataFrame, key: str, selected: list[str], target: pd.DataFrame) -> pd.DataFrame:
    if "All" in selected:
        return target
    valid = df_subset[df_subset[key].isin(selected)]["movie_id"]
    return target[target["movie_id"].isin(valid)]

def filter_movies(year_range, genres_sel, themes_sel, countries_sel, search_query) -> pd.DataFrame:
    rel = data["releases"][['movie_id','country_release','release_year']]
    mv  = data["movies"][['movie_id','title','description']].merge(rel, on="movie_id", how="left")
    mv  = mv[mv["release_year"].between(*year_range)]
    mv = filter_by(data["genres"],    "genre",    genres_sel,    mv)
    mv = filter_by(data["themes"],    "theme",    themes_sel,    mv)
    mv = filter_by(data["countries"], "country",  countries_sel, mv)
    if search_query:
        mask = (
            mv["title"].str.contains(search_query, case=False, na=False) |
            mv["description"].str.contains(search_query, case=False, na=False)
        )
        mv = mv[mask]
    return mv

df = filter_movies(year_range, genres_sel, themes_sel, countries_sel, search_query)
st.download_button("⬇️ Download Filtered Data", df.to_csv(index=False), "filtered.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# 4 · Helpers & Insights
# ──────────────────────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
def _top_terms(descs: list[str], n_terms: int=5) -> list[tuple[str,int]]:
    if not descs:
        return []
    X = tfidf.fit_transform(descs)
    w = X.sum(axis=0).A1; vocab = tfidf.get_feature_names_out()
    idx = w.argsort()[::-1][:n_terms]
    return [(vocab[i], int(w[i])) for i in idx]

def render_metrics(df: pd.DataFrame, label1="Total Films", label2="Avg Release Year"):
    c1, c2 = st.columns(2)
    with c1: st.metric(label1, df.shape[0])
    with c2: st.metric(label2, round(df["release_year"].mean(), 0))

def render_insights(df: pd.DataFrame, title="🔍 Top Genres & Themes"):
    with st.expander(title, expanded=True):
        topg = data["genres"].loc[data["genres"].movie_id.isin(df.movie_id), "genre"].value_counts().head(5)
        if not topg.empty: st.markdown("**Genres:** " + ", ".join(topg.index))
        topt = data["themes"].loc[data["themes"].movie_id.isin(df.movie_id), "theme"].value_counts().head(5)
        if not topt.empty: st.markdown("**Themes:** " + ", ".join(topt.index))

# ──────────────────────────────────────────────────────────────────────────────
# 5 · Views
# ──────────────────────────────────────────────────────────────────────────────
def geo_map(df):
    if df.empty:
        st.info("No movies match filters."); return
    render_metrics(df)
    cnt = df.groupby('country_release').size().rename('film_count')
    avg = df.groupby('country_release')['release_year'].mean().rename('avg_release_year')
    gdf = pd.concat([cnt, avg], axis=1).reset_index()
    metric = st.selectbox("📊 Metric", ['film_count','avg_release_year'], format_func=lambda x: x.replace('_',' ').title(), key='geo')
    fig = px.choropleth(gdf, locations='country_release', locationmode='country names', color=metric,
                        projection='natural earth', color_continuous_scale=cmap, template='plotly_white',
                        title='Film Distribution by Country')
    fig.update_geos(showframe=True, showcoastlines=True, coastlinecolor='lightgray', showcountries=True, countrycolor='lightgray')
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=20,b=20), coloraxis_colorbar=dict(orientation='h',y=-0.12,x=0.5,len=0.6))
    c1,c2=st.columns([3,1]);
    with c1: st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True,"modeBarButtonsToAdd":["lasso2d","select2d"],"displaylogo":False})
    with c2: render_insights(df); st.markdown("**Top plot-keywords:** " + ", ".join(f"\"{w}\"" for w,_ in _top_terms(df['description'].dropna().astype(str).tolist(),5)))

def social_scatter(df_fil):
    if df_fil.empty: st.info("No data for selection."); return
    render_metrics(df_fil, label2="Film Count")
    dfv = df_fil[['movie_id','release_year','title']].merge(data['themes'], on='movie_id').rename(columns={'theme':'original_theme'})
    focus = st.multiselect("Focus themes", sorted(dfv['original_theme'].unique()), default=sorted(dfv['original_theme'].unique())[:5])
    dfv=dfv[dfv['original_theme'].isin(focus)]; top10=dfv['original_theme'].value_counts().nlargest(10).index; dfv['theme']=dfv['original_theme'].where(dfv['original_theme'].isin(top10),'Other')
    fig=px.violin(dfv,x='theme',y='release_year',color='theme',box=True,points='all',hover_data=['title'],title='Release Year by Theme',template='plotly_white')
    fig.update_traces(meanline_visible=True,opacity=0.6); fig.update_layout(height=600,xaxis_tickangle=-30,showlegend=False,margin=dict(l=40,r=20,t=60,b=120))
    c1,c2=st.columns([3,1]);
    with c1: st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"modeBarButtonsToAdd":["lasso2d","select2d"],"displaylogo":False})
    with c2: render_insights(df_fil); st.markdown("**Top plot-keywords:** " + ", ".join(f"\"{w}\"" for w,_ in _top_terms(df_fil['description'].dropna().astype(str).tolist(),5)))

def cultural_embed(df):
    if df.empty: st.info("No data."); return
    render_metrics(df)
    # lazy build sparse
    X, ORDERED_MIDS = build_sparse_features()
    mids = random.sample(df.movie_id.tolist(), min(len(df), sample_size))
    idx = [ORDERED_MIDS.index(m) for m in mids]
    Xs = X[idx]
    if Xs.shape[0]>6000: coords=UMAP(n_components=2,random_state=42).fit_transform(Xs); method="UMAP"
    else: coords=TSNE(n_components=2,perplexity=30,random_state=42).fit_transform(Xs.toarray()); method="t-SNE"
    clusters=DBSCAN(eps=0.5,min_samples=5).fit_predict(coords)
    emb=pd.DataFrame(coords,columns=['x','y']); emb['cluster']=clusters.astype(str); emb['movie_id']=mids
    emb=emb.merge(df.set_index('movie_id')[['title','release_year','description']],left_on='movie_id',right_index=True)
    sizes=df['movie_id'].map(data['releases'].groupby('movie_id').size()).fillna(1); emb['size']=np.log1p(sizes)*marker_size
    emb['genre']=emb['movie_id'].map(data['genres'].groupby('movie_id')['genre'].first()); emb['desc_snip']=emb['description'].fillna('').str[:100]+'...'
    fig=px.scatter(emb,x='x',y='y',color='genre',symbol='cluster',size='size',hover_data={'title':True,'release_year':True,'cluster':True,'desc_snip':True,'x':False,'y':False,'size':False},title=f"{method} Cultural Clusters by Genre",template='plotly_dark')
    fig.update_layout(height=700,legend=dict(itemsizing='constant'))
    c1,c2=st.columns([3,1]);
    with c1: st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"modeBarButtonsToAdd":["lasso2d","select2d"],"displaylogo":False})
    with c2: render_insights(df)

# ──────────────────────────────────────────────────────────────────────────────
# 6 · Main render
# ──────────────────────────────────────────────────────────────────────────────
st.title("🎬 Film Cultural Impact Explorer – v1.5")
if   view=="Geo Map": geo_map(df)
elif view=="Social Scatter": social_scatter(df)
elif view=="Cultural Embed": cultural_embed(df)
else:
    tabs=st.tabs(["Geo","Social","Embed"])
    with tabs[0]: geo_map(df)
    with tabs[1]: social_scatter(df)
    with tabs[2]: cultural_embed(df)
