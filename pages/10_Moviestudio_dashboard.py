import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="Movie Studio Dashboard", layout="wide")

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigate to", [
    "Revenue vs Budget",
    "Movies by Studio",
    "Movie Release Window"
])

# --- Page Main Title ---
st.markdown("<h1 style='text-align: left; color: black;'>🎬 Movie Studio Dashboard</h1>", unsafe_allow_html=True)
st.write("")  # Small space

# --- Load Data ---
@st.cache_data
def load_data():
    movies = pd.read_csv(r"archive/movies2.csv")
    studios = pd.read_csv(r"archive/studio3.csv")
    revenues = pd.read_csv(r"archive/revenuebudget.csv")
    releases = pd.read_csv(r"archive/releases.csv")

    releases['release_date'] = pd.to_datetime(releases['release_date'], errors='coerce')

    movies = movies.drop_duplicates(subset=['id'])
    studios = studios.drop_duplicates(subset=['id'])
    releases = releases.drop_duplicates(subset=['id'])

    return movies, studios, revenues, releases

movies, studios, revenues, releases = load_data()

# --- Pages ---

# Page 1: Revenue vs Budget
if page == "Revenue vs Budget":
    st.header("📈 Revenue vs Budget")

    scatter = px.scatter(
        revenues.dropna(subset=['budget', 'revenue']),
        x='budget',
        y='revenue',
        hover_data=['title', 'budget', 'revenue'],
        log_x=True,
        log_y=True,
        title="Revenue vs Budget (Log Scale)"
    )
    st.plotly_chart(scatter, use_container_width=True)

# Page 2: Movies by Studio (Bubble Chart)
elif page == "Movies by Studio":
    st.header("🏢 Movies Produced by Studio")

    if 'country' not in studios.columns:
        st.error("The 'country' column is missing from the studios dataset.")
    else:
        selected_country = st.selectbox(
            "Select Country to Filter By",
            options=studios['country'].dropna().unique()
        )

        filtered_studios = studios[studios['country'] == selected_country]

        studio_movie_counts = filtered_studios.groupby('studio_name').agg(
            movie_count=('id', 'count')
        ).reset_index()

        bubble_chart = px.scatter(
            studio_movie_counts,
            x='studio_name',
            y='movie_count',
            size='movie_count',
            color='studio_name',
            hover_name='studio_name',
            title="Movies Produced by Each Studio (Bubble Plot)",
            size_max=50
        )
        st.plotly_chart(bubble_chart, use_container_width=True)

# Page 3: Movie Release Window
elif page == "Movie Release Window":
    st.header("🗓️ Movie Release Window by Studio and Year")

    # Merge selected columns
    movies_releases = pd.merge(
        movies[['id', 'title']],
        releases[['id', 'release_date']],
        on='id',
        how='left'
    )

    # Merge with studios
    movies_releases_studios = pd.merge(
        movies_releases,
        studios[['id', 'studio_name']],
        on='id',
        how='left'
    )

    selected_studio = st.selectbox(
        "Select Studio",
        options=movies_releases_studios['studio_name'].dropna().unique()
    )

    studio_movies = movies_releases_studios[movies_releases_studios['studio_name'] == selected_studio]

    if not studio_movies.empty:
        studio_movies['release_date'] = pd.to_datetime(studio_movies['release_date'], errors='coerce')
        studio_movies['year'] = studio_movies['release_date'].dt.year

        # Clean years
        studio_movies = studio_movies.dropna(subset=['year'])
        studio_movies['year'] = studio_movies['year'].astype(int)

        # Plot histogram for number of movies released per year
        fig = px.histogram(
            studio_movies,
            x='year',
            nbins=30,
            title=f"Number of Movies Released Per Year - {selected_studio}",
            labels={'year': 'Release Year', 'count': 'Number of Movies'},
            color_discrete_sequence=['#00CC96']
        )

        fig.update_layout(
            xaxis_title="Release Year",
            yaxis_title="Number of Movies",
            bargap=0.2
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No movies found for the selected studio.")
