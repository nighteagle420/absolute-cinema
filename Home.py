import streamlit as st
import base64
from pathlib import Path

# Page Config
st.set_page_config(
    page_title='Absolute Cinema',
    page_icon='🎥',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Utility: load binary as base64
def get_base64(path: Path) -> str:
    """Return base64-encoded string of file contents."""
    return base64.b64encode(path.read_bytes()).decode()

# Load background image
bg_path = BASE_DIR / 'archive' / 'img.jpg'
bg_img = ''
try:
    if bg_path.exists():
        bg_img = get_base64(bg_path)
except Exception as e:
    st.warning(f"Background load failed: {e}")

# Inject CSS
css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
html, body, [class*='css'] {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: linear-gradient(rgba(0,0,0,0.6),rgba(0,0,0,0.6)), url('data:image/jpeg;base64,{bg_img}') center/cover fixed; }}
.navbar {{ position:sticky; top:0; z-index:999; background:rgba(0,0,0,0.8); padding:1rem 2rem; text-align:center; border-bottom:1px solid rgba(255,255,255,0.2); }}
.navbar h1 {{ color:#00FFFF; font-size:2rem; margin:0; font-weight:800; }}
.container {{ padding:4rem 2rem; animation:fadeInUp 1s ease-out; }}
.section {{ background:rgba(0,0,0,0.7); padding:2rem; border-radius:1.25rem; margin-bottom:3rem; box-shadow:0 8px 30px rgba(0,0,0,0.6); }}
h2 {{ color:#FFD700; text-align:center; margin-bottom:1.25rem; }}
.intro-text {{ color:#ddd; font-size:1.125rem; line-height:1.7; text-align:justify; }}
.team-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:1.5rem; margin-top:2rem; }}
.team-card {{ background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.2); border-radius:0.75rem; padding:1.25rem; text-align:center; transition:0.3s; }}
.team-card:hover {{ background:rgba(255,255,255,0.15); transform:translateY(-8px); }}
.team-card h4 {{ color:#00e6e6; margin-bottom:0.5rem; }}
.team-card p {{ color:#FFD700; font-size:0.9375rem; }}
@keyframes fadeInUp {{ from {{ opacity:0; transform:translateY(30px); }} to {{ opacity:1; transform:translateY(0); }} }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Navbar
st.markdown("<div class='navbar'><h1>Absolute Cinema 🎥</h1></div>", unsafe_allow_html=True)

# Main container
st.markdown("<div class='container'>", unsafe_allow_html=True)

# Introduction
st.markdown("<div class='section'><h2>Behind the Scenes of Movie Industry</h2>", unsafe_allow_html=True)
intro = ("In the dynamic world of cinema, understanding what resonates with audiences is crucial.  "
         "Our data-driven journey uncovers patterns and insights that drive iconic achievements.  "
         "From genre trends to cultural shifts, we visualize and celebrate the art of storytelling.")
st.markdown(f"<div class='intro-text'>{intro}</div></div>", unsafe_allow_html=True)

# Team
st.markdown("<div class='section'><h2>Meet Our Team</h2><div class='team-grid'>", unsafe_allow_html=True)
credits = [
    ("Puspesh Kumar Srivastava","241110405","Female Centric Visualization Dashboard"),
    ("Sagar Kumar","24111060","Cultural Analysis"),
    ("Krishanu Ray","24111037","OscarLens: Oscar Awards Analysis"),
    ("Uddeshya Raj","241110406","Movie Recommender"),
    ("Suvradip Das","231110051","Genre Analysis"),
    ("Sayak Bera","241110608","Actor-Director Synergy"),
    ("Praveen Patel","241110404","Budget and Revenue Analysis"),
    ("Srinjoy Sarkar","231110050","Review Sentiment Analysis"),
    ("Sangharsh Nagdevte","231110064","Movie Studio Analysis"),
]
for name, roll, role in credits:
    card = f"""
    <div class='team-card'>
      <h4>{name}</h4><p>{roll}</p><p>{role}</p>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)

# Footer text
st.markdown("**Join us as we decode the secrets of cinematic success, shaping the future of storytelling.**")
