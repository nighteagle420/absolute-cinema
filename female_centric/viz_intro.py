import streamlit as st
import os
import pandas as pd
import requests
import base64

TMDB_API_KEY = "ed2e35fa097949e807f98615c4e9a79d"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w300"
CSV_DIR = os.path.join(os.getcwd(), 'female_centric/Puspesh_45K_MovieLens')
CSV_FILE = 'female_centric_movies.csv'

# Intro rendering
def render_intro(data):
    # 1. Definition & importance (5 lines)
    '''st.markdown("""
**What is a Female-Centric Movie?**  
A film that puts women’s stories, experiences, and leadership at its core.  
It centers female protagonists and creative voices, challenging traditional gender norms.  
These movies deliver richer perspectives and often drive cultural and commercial success.  
Understanding them reveals insights into representation, audience demand, and industry evolution.
""", unsafe_allow_html=True)

    # 2. Highlight 4 criteria in 1–2 sentences each
    st.markdown("""
- **Female Lead**: The top-billed cast member is female, ensuring the story follows a woman’s journey.  
- **Bechdel Test**: The movie passes the Bechdel Test—featuring at least two named women who talk to each other about something other than a man. 
- **Majority Female Top-3**: At least two of the first three billed roles are female, emphasizing ensemble representation.  
- **Community Keywords**: Movie metadata tags or keywords include at least one term from a curated set of women-centric themes (e.g., “female empowerment,” “sisterhood,” “feminist protagonist”).
""", unsafe_allow_html=True)'''
    # First, define your custom CSS styles
    st.markdown("""
    <style>
    .definition-title {
        font-size: 24px;
        font-weight: bold;
    }
    .definition-text {
        font-size: 18px;
    }
    .criteria-title {
        font-size: 24px;
        font-weight: bold;
    }
    .criteria-item {
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 1. Definition & importance with increased font size
    st.markdown("""
    <div class="definition-title">What is a Female-Centric Movie?</div>  
    <div class="definition-text">
    A film that puts women's stories, experiences, and leadership at its core.<br>  
    It centers female protagonists and creative voices, challenging traditional gender norms.<br>  
    These movies deliver richer perspectives and often drive cultural and commercial success.<br>  
    Understanding them reveals insights into representation, audience demand, and industry evolution.
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Highlight 4 criteria with increased font size
    st.markdown("""
    <div class="criteria-title">Key Criteria:</div>
    <ul>
    <div class="criteria-item"><b>1. Female Lead</b>: The top-billed cast member is female, ensuring the story follows a woman's journey.</div>
    <div class="criteria-item"><b>2. Female Director</b>: A woman director shapes the narrative through a female creative lens.</div>
    <div class="criteria-item"><b>3. Majority Female Top-3</b>: At least two of the first three billed roles are female, emphasizing ensemble representation.</div>
    <div class="criteria-item"><b>4. Community Keywords</b>: Movie metadata tags or keywords include at least one term from a curated set of women-centric themes (e.g., "female empowerment," "sisterhood," "feminist protagonist").</div>
    </ul>
    """, unsafe_allow_html=True)

    # 3. Load CSV and filter movies meeting all criteria
    csv_path = os.path.join(CSV_DIR, CSV_FILE)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"CSV not found at {csv_path}")
        return

    # require all four criteria true
    mask = (
        (df['bechdel'] == True) &
        (df['lead'] == True) &
        (df['ensemble'] == True) &
        (df['community'] == True)
    )
    df_fc = df[mask]
    # sort by popularity, vote_average, revenue
    df_top = df_fc.sort_values(
        ['popularity', 'vote_average', 'revenue'],
        ascending=[False, False, False]
    ).head(10)

    # 4. Display posters of top 10
    st.subheader("Top 10 Female-Centric Movies by Popularity, Rating & Revenue")
    cols = st.columns(5)

    for idx, (_, row) in enumerate(df_top.iterrows()):
        col = cols[idx % 5]
        imdb_id = row['imdb_id']
        # fetch poster via TMDb by IMDb ID first, then by title if needed
        poster_url = None
        
        # 1) Try find by IMDb ID
        if imdb_id:
            try:
                res = requests.get(
                    f"https://api.themoviedb.org/3/find/{imdb_id}",
                    params={"api_key": "ed2e35fa097949e807f98615c4e9a79d", "external_source": "imdb_id"}
                ).json()
                mr = res.get('movie_results', [])
                if mr and mr[0].get('poster_path'):
                    poster_url = "https://image.tmdb.org/t/p/w300" + mr[0]['poster_path']
            except:
                poster_url = None
        
        # 2) If still no poster, try search by title
        if not poster_url:
            title = best.get('title') or best.get('original_title') or ''
            if title:
                try:
                    sr = requests.get(
                        "https://api.themoviedb.org/3/search/movie",
                        params={"api_key": "ed2e35fa097949e807f98615c4e9a79d", "query": title}
                    ).json()
                    results = sr.get('results', [])
                    if results and results[0].get('poster_path'):
                        poster_url = "https://image.tmdb.org/t/p/w300" + results[0]['poster_path']
                except:
                    poster_url = None
        
        # 3) Final fallback to CSV poster_path
        if not poster_url and pd.notna(best.get('poster_path')):
            poster_url = "https://image.tmdb.org/t/p/w300" + best.get('poster_path')
        '''poster_url = None
        # fetch from TMDb
        try:
            res = requests.get(
                f"https://api.themoviedb.org/3/find/{imdb_id}",
                params={"api_key": TMDB_API_KEY, "external_source": "imdb_id"}
            ).json()
            mr = res.get('movie_results', [])
            if mr and mr[0].get('poster_path'):
                poster_url = POSTER_BASE_URL + mr[0]['poster_path']
        except:
            pass
        # fallback to CSV poster_path
        if not poster_url and pd.notna(row.get('poster_path')):
            poster_url = POSTER_BASE_URL + row['poster_path']
'''
        if poster_url:
            col.image(poster_url, use_container_width=True)
        else:
            col.write("No poster available")

        # caption title & year
        title = row.get('title') or row.get('original_title')
        rd = row.get('release_date', '')
        year = rd[:4] if pd.notna(rd) and len(rd) >= 4 else ''
        col.markdown(f"**{title}**{f' ({year})' if year else ''}")
