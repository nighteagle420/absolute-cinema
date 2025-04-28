from __future__ import annotations
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
# 0 · Google Drive Streaming Loader (no local writes)
# ──────────────────────────────────────────────────────────────────────────────
# If you have full shareable Drive URLs, list them here:
GDRIVE_LINKS = {
    "movies.csv":    "https://drive.google.com/file/d/1-AxbDiUcLN8KGFb3gIXrSfyyHdPWV_iN/view?usp=sharing",
    "genres.csv":    "https://drive.google.com/file/d/10bKEjkyWOquTpCJZIwz2EE3AAW5HbEjm/view?usp=sharing",
    "themes.csv":    "https://drive.google.com/file/d/1KvIdV7lEhBXiPgmINYeGEwAC6mPnl41V/view?usp=sharing",
    "releases.csv":  "https://drive.google.com/file/d/1d5GnDcCfX04vgmPavf0vjP2uf4nliJzf/view?usp=sharing",
    "countries.csv": "https://drive.google.com/file/d/1oWCucWLLKlA9EShNoh2dVb6Tgyf4ohm7/view?usp=sharing",
}

def read_gdrive_csv_from_link(share_link: str) -> pd.DataFrame:
    """
    Convert a Google Drive shareable link into a direct-download URL
    and stream it into pandas without writing to disk.
    """
    # extract the file ID from the shareable URL
    # e.g. https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    file_id = share_link.split('/d/')[1].split('/')[0]
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    """
    Stream all CSVs directly from Google Drive into memory and preprocess.
    """
    data: dict[str, pd.DataFrame] = {}
    # load each CSV by streaming from Drive
    for fname, link in GDRIVE_LINKS.items():
        df = read_gdrive_csv_from_link(link)
        key = fname.replace('.csv', '')
        # rename id->movie_id if needed
        if 'id' in df.columns and 'movie_id' not in df.columns:
            df = df.rename(columns={'id':'movie_id'})
        data[key] = df

    # standard pipeline renames
    data['movies']   = data['movies'].rename(columns={'name':'title','date':'year'})
    data['releases'] = (
        data['releases']
        .rename(columns={'country':'country_release'})
        .assign(
            release_date=lambda d: pd.to_datetime(d['date'], errors='coerce'),
            release_year=lambda d: pd.to_datetime(d['date'], errors='coerce').dt.year
        )
    )
    return data

# load all tables in-memory without any disk writes
data = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# Replace each <FILE_ID> with your actual Google Drive file ID for the CSV
GDRIVE_IDS = {
    "movies.csv":    "1-AxbDiUcLN8KGFb3gIXrSfyyHdPWV_iN",
    "genres.csv":    "10bKEjkyWOquTpCJZIwz2EE3AAW5HbEjm",
    "themes.csv":    "1KvIdV7lEhBXiPgmINYeGEwAC6mPnl41V",
    "releases.csv":  "1d5GnDcCfX04vgmPavf0vjP2uf4nliJzf",
    "countries.csv": "1oWCucWLLKlA9EShNoh2dVb6Tgyf4ohm7",
}

def read_gdrive_csv(fname: str) -> pd.DataFrame:
    """
    Stream a CSV directly from Google Drive into pandas without writing to disk.
    """
    fid = GDRIVE_IDS[fname]
    # direct-download URL for Google Drive
    url = f"https://drive.google.com/uc?export=download&id={fid}"
    return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    # load each CSV into memory
    data: dict[str, pd.DataFrame] = {}
    for fname in GDRIVE_IDS:
        df = read_gdrive_csv(fname)
        key = fname.replace('.csv','')
        # rename id->movie_id if present
        if 'id' in df.columns and 'movie_id' not in df.columns:
            df = df.rename(columns={'id':'movie_id'})
        data[key] = df

    # post-process to mirror original pipeline
    # movies: rename name->title, date->year
    data['movies'] = data['movies'].rename(columns={'name':'title','date':'year'})
    # releases: rename country->country_release, parse dates
    rel = data['releases'].rename(columns={'country':'country_release'})
    rel['release_date'] = pd.to_datetime(rel['date'], errors='coerce')
    rel['release_year'] = rel['release_date'].dt.year
    data['releases'] = rel

    return data

# load all tables in-memory
data = load_data()

@st.cache_resource(show_spinner=False)
def build_sparse_features() -> tuple[sparse.csr_matrix, list[int]]:
    # generate hash signature to detect data changes
    h = hashlib.md5()
    # include each table's contents in signature
    for key, df in data.items():
        h.update(key.encode())
        h.update(pd.util.hash_pandas_object(df, index=True).values)
    sig = h.hexdigest()

    # use in-memory caching
    cache = {}  # simple dict cache
    if cache.get('sig') == sig:
        return cache['X'], cache['mids']

    # build features
    mids = data['movies']['movie_id'].tolist()
    def collect(df: pd.DataFrame, col: str) -> list[list[str]]:
        grp = df.groupby('movie_id')[col].apply(list)
        return [grp.get(mid, []) for mid in mids]

    mlb_g = MultiLabelBinarizer(sparse_output=True)
    mlb_t = MultiLabelBinarizer(sparse_output=True)
    Xg = mlb_g.fit_transform(collect(data['genres'], 'genre'))
    Xt = mlb_t.fit_transform(collect(data['themes'], 'theme'))
    X = sparse.hstack([Xg, Xt]).tocsr()

    # update cache
    cache['X'], cache['mids'], cache['sig'] = X, mids, sig
    return X, mids

# precompute embedding features
SPARSE_X, ORDERED_MIDS = build_sparse_features()

# ──────────────────────────────────────────────────────────────────────────────
# 1 · Streamlit UI: Sidebar, Filters & Config
# ──────────────────────────────────────────────────────────────────────────────
DEC_META: dict[str, tuple[int, int, list[tuple[str, str]]]] = {
    '1920–1929 – Silent Era':  (1920, 1929, [('German Expressionism','https://en.wikipedia.org/wiki/German_Expressionism'),('Rise of Hollywood','https://en.wikipedia.org/wiki/History_of_Hollywood')]),
    '1930–1949 – Golden Age':   (1930, 1949, [('Golden Age of Hollywood','https://en.wikipedia.org/wiki/Golden_Age_of_Hollywood'),('Pre-/Post-war cinema','https://en.wikipedia.org/wiki/History_of_film')]),
    '1950–1969 – New Waves':    (1950, 1969, [('French New Wave','https://en.wikipedia.org/wiki/French_New_Wave'),('Japanese New Wave','https://en.wikipedia.org/wiki/Japanese_New_Wave')]),
    '1970–1989 – Blockbusters': (1970, 1989, [('New Hollywood','https://en.wikipedia.org/wiki/New_Hollywood'),('Blockbuster era','https://en.wikipedia.org/wiki/Blockbuster_(entertainment)')]),
    '1990–2009 – Global Boom':  (1990, 2009, [('Nollywood','https://en.wikipedia.org/wiki/Cinema_of_Nigeria'),('Asian cinema boom','https://en.wikipedia.org/wiki/Cinema_of_South_Korea')]),
    '2010–2025 – Streaming Era':(2010,2025,[('Streaming revolution','https://en.wikipedia.org/wiki/Streaming_media'),('Hallyu wave','https://en.wikipedia.org/wiki/Hallyu')]),
}

st.set_page_config(page_title='🎬 Film Cultural Impact Explorer', layout='wide')
with st.sidebar:
    st.title('🎥 Film Impact Explorer')
    view = st.radio('🔍 Select View',['Geo Map','Social Scatter','Cultural Embed','Historical'])
    search_query = st.text_input('🖊️ Search Titles/Descriptions')
    if view=='Historical':
        decade = st.selectbox('📆 Decade', list(DEC_META.keys()))
        yr_min,yr_max,cites = DEC_META[decade]
        st.markdown(f'**Years:** {yr_min} – {yr_max}')
        year_range=(yr_min,yr_max)
    else:
        yrs = data['releases']['release_year'].dropna()
        year_range=st.slider('📅 Release Year Range',int(yrs.min()),int(yrs.max()),(int(yrs.min()),int(yrs.max())))
    with st.expander('📦 Metadata Filters'):
        genres_sel  = st.multiselect('🎭 Genres',['All']+sorted(data['genres']['genre'].unique()),['All'])
        themes_sel  = st.multiselect('🏷️ Themes',['All']+sorted(data['themes']['theme'].unique()),['All'])
        countries_sel=st.multiselect('🌍 Production Countries',['All']+sorted(data['countries']['country'].unique()),['All'])
    if view in ('Cultural Embed','Historical'):
        with st.expander('🧮 Embedding Settings'):
            sample_size=st.slider('📏 Sample Size',900,100000,4000,step=500)
            marker_size=st.slider('🖋️ Marker Size',3,20,7)
    cmap=st.selectbox('🌈 Color Palette',['Blues','Greens','Oranges','Purples'],index=0)
    if view=='Historical':
        st.markdown('---'); st.markdown('### 📚 Further Reading')
        for txt,url in cites: st.markdown(f'* [{txt}]({url})')

# ──────────────────────────────────────────────────────────────────────────────
# 2 · Data Filtering & Download
# ──────────────────────────────────────────────────────────────────────────────
def filter_by(df,col,sel,target):
    return target if 'All' in sel else target[target['movie_id'].isin(df[df[col].isin(sel)]['movie_id'])]

def filter_movies(year_range,genres_sel,themes_sel,countries_sel,search_query):
    mv=data['movies'][['movie_id','title','description']].merge(data['releases'][['movie_id','release_year','country_release']],on='movie_id',how='left')
    mv=mv[mv['release_year'].between(*year_range)]
    mv=filter_by(data['genres'],'genre',genres_sel,mv)
    mv=filter_by(data['themes'],'theme',themes_sel,mv)
    mv=filter_by(data['countries'],'country',countries_sel,mv)
    if search_query:
        mask=mv['title'].str.contains(search_query,case=False,na=False)|mv['description'].str.contains(search_query,case=False,na=False)
        mv=mv[mask]
    return mv

df=filter_movies(year_range,genres_sel,themes_sel,countries_sel,search_query)
st.download_button('⬇️ Download Filtered Data',df.to_csv(index=False),'filtered.csv','text/csv')

# ──────────────────────────────────────────────────────────────────────────────
# 3–6 · Helpers, Views & Main Render (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
# … include your _top_terms, render_metrics, render_insights, geo_map, social_scatter, cultural_embed, and final st.title/logic here …

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

def cultural_embed(df: pd.DataFrame):
    if df.empty:
        st.info("No data.")
        return
    render_metrics(df)

    # Use precomputed globals SPARSE_X and ORDERED_MIDS
    mids = random.sample(df.movie_id.tolist(), min(len(df), sample_size))
    idx = [ORDERED_MIDS.index(m) for m in mids]
    Xs = SPARSE_X[idx]

    if Xs.shape[0] > 6000:
        coords = UMAP(n_components=2, random_state=42).fit_transform(Xs)
        method = "UMAP"
    else:
        coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(Xs.toarray())
        method = "t-SNE"

    clusters = DBSCAN(eps=0.5, min_samples=5).fit_predict(coords)

    emb_df = pd.DataFrame(coords, columns=["x","y"])
    emb_df["cluster"]  = clusters.astype(str)
    emb_df["movie_id"] = mids
    emb_df = emb_df.merge(
        df.set_index("movie_id")[['title','release_year','description']],
        left_on="movie_id", right_index=True
    )

    cnt_map = data['releases'].groupby('movie_id').size().to_dict()
    emb_df['size'] = emb_df['movie_id'].map(lambda m: np.log1p(cnt_map.get(m,1)) * marker_size)
    emb_df['genre']     = emb_df['movie_id'].map(data['genres'].groupby('movie_id')['genre'].first())
    emb_df['desc_snip'] = emb_df['description'].fillna('').str.slice(0,100) + '...'

    fig = px.scatter(
        emb_df, x='x', y='y', color='genre', symbol='cluster', size='size',
        hover_data={'title':True,'release_year':True,'cluster':True,'desc_snip':True,'x':False,'y':False,'size':False},
        title=f"{method} Cultural Clusters by Genre", template='plotly_dark'
    )
    fig.update_layout(height=700, legend=dict(itemsizing='constant'))

    c1, c2 = st.columns([3,1])
    with c1:
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True,"modeBarButtonsToAdd":["lasso2d","select2d"],"displaylogo":False})
    with c2:
        render_insights(df)

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
