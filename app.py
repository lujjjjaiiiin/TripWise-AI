"""
TripWise AI — Premium Streamlit Travel Assistant
Splash screen (SVG heart-path flight animation) -> Main App (Home / nav pages)
Destination Explorer page runs the real Cosine Similarity recommendation engine.
"""

import time
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="TripWise AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FEATURE_COLS = [
    "latitude", "longitude", "culture", "adventure", "nature", "beaches",
    "nightlife", "cuisine", "wellness", "urban", "seclusion",
    "has_airport", "is_short_trip", "is_one_week", "temp_avg_yearly",
    "budget_level_encoded", "HotelRating_encoded", "rating_was_unknown",
    "region_africa", "region_asia", "region_europe", "region_middle_east",
    "region_north_america", "region_oceania", "region_south_america",
]

REGION_COLS = [
    "region_africa", "region_asia", "region_europe", "region_middle_east",
    "region_north_america", "region_oceania", "region_south_america",
]

# Heart-shaped flight path (SVG user units, viewBox 0 0 1200 750).
# The SAME string is drawn on screen AND fed to <animateMotion><mpath>, so the
# plane rides exactly on the drawn line at any screen size — no CSS offset-path
# drift between the guide and the aircraft.
HEART_PATH = (
    "M 120 660 C 230 600 300 520 360 460 C 300 430 250 380 255 310 "
    "C 260 220 340 155 430 165 C 490 172 535 210 555 260 "
    "C 575 210 620 172 680 165 C 770 155 850 220 855 310 "
    "C 860 380 810 430 750 460 C 820 500 900 500 940 430 "
    "C 985 355 985 270 940 200 C 1005 165 1060 130 1120 90"
)

# Top-view jet silhouette, nose pointing +x so rotate="auto" aligns it perfectly.
PLANE_SHAPE = (
    "M22,0 L8,-3.4 L-2,-16.5 L-8,-16.5 L-4,-3.4 L-14,-3.4 L-19,-8 L-22.5,-8 "
    "L-20,-2.6 L-20,2.6 L-22.5,8 L-19,8 L-14,3.4 L-4,3.4 L-8,16.5 L-2,16.5 "
    "L8,3.4 Z"
)


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads the cleaned dataset exported from the Colab notebook (tripwise_data.csv)."""
    return pd.read_csv("tripwise_data.csv")


@st.cache_resource
def build_scaler(df: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    return scaler, X_scaled


# ----------------------------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        :root{
            --sky:#0EA5E9;
            --blue:#2563EB;
            --white:#FFFFFF;
            --light-gray:#F3F6FA;
            --text-dark:#0F172A;
            --text-muted:#475569;
            --shadow: 0 20px 60px rgba(37,99,235,0.12);
        }

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; font-size: 17.5px; line-height:1.65; }
        h1,h2,h3, .tw-display { font-family: 'Outfit', sans-serif; }

        #MainMenu, footer, header {visibility:hidden;}
        .block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width: 1200px;}
        .stApp{ background: linear-gradient(180deg, #DCEFFC 0%, #C9E6FA 55%, #BFE0F8 100%); }
        body{ background: linear-gradient(180deg, #DCEFFC 0%, #C9E6FA 55%, #BFE0F8 100%); }

        /* ---------------- READABILITY / TYPOGRAPHY ---------------- */
        .stApp, .main, .block-container { direction: ltr; }
        label, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        .stSelectbox label, .stRadio label, .stCheckbox label, .stSlider label {
            color: var(--text-dark) !important;
            font-weight: 650 !important;
            font-size: 1.08rem !important;
            letter-spacing: .01em;
            margin-bottom: .35rem !important;
        }
        .tw-section-title{ font-weight:800; font-size:1.9rem; color:var(--text-dark); margin: 2.6rem 0 1.2rem 0; letter-spacing:-.01em;}
        .tw-muted{ color:var(--text-muted); font-size:1.05rem; line-height:1.6; }

        /* Consistent vertical rhythm between Streamlit widgets */
        div[data-testid="stVerticalBlock"] > div{ margin-bottom:.35rem; }
        div[data-testid="column"]{ padding: 0 .6rem; }

        /* ---------------- SLIDERS (fixed direction + bigger + clearer) ---------------- */
        [data-testid="stSlider"]{
            direction: ltr !important;
            padding-top:.3rem; padding-bottom:.6rem;
        }
        [data-testid="stSlider"] *{
            direction: ltr !important;
            unicode-bidi: plaintext !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"]{ margin-top:.5rem; }
        [data-testid="stTickBarMin"], [data-testid="stTickBarMax"]{
            color: var(--blue) !important; font-weight:700 !important; font-size:.95rem !important;
        }
        [data-testid="stThumbValue"]{
            color: white !important; font-weight:700 !important; font-size:.9rem !important;
            background: var(--blue) !important; padding:.15rem .5rem !important; border-radius:8px !important;
        }
        [data-baseweb="slider"] [role="slider"]{
            box-shadow: 0 0 0 4px rgba(37,99,235,.18) !important;
        }

        /* ---------------- FORM CONTROLS (bigger + more spacing) ---------------- */
        .stSelectbox, .stRadio, .stCheckbox, .stTextInput, .stNumberInput { margin-bottom: 1.1rem; }
        div[data-baseweb="select"] > div{
            border-radius:14px !important; min-height:3rem !important; font-size:1.05rem !important;
            border-color: rgba(30,58,138,0.25) !important;
        }
        .stTextInput input, .stNumberInput input{
            border-radius:14px !important; min-height:3rem !important; font-size:1.05rem !important;
        }
        div[role="radiogroup"] label{ font-size:1rem !important; margin-right:1.2rem !important; }
        .stCheckbox label p{ font-size:1.05rem !important; }
        form[data-testid="stForm"]{
            background: rgba(255,255,255,0.25);
            border-radius: 26px;
            padding: 2rem 1.8rem;
            border: 1px solid rgba(30,58,138,0.1);
        }

        section[data-testid="stSidebar"]{
            background: linear-gradient(180deg, #0B1220 0%, #111B33 100%);
            border-right: none;
        }
        section[data-testid="stSidebar"] * { color:#E2E8F0 !important; }
        .tw-logo{
            font-family:'Outfit',sans-serif; font-weight:800; font-size:1.4rem;
            background: linear-gradient(90deg,#38BDF8,#60A5FA);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            margin-bottom:1.6rem; display:block;
        }

        .stButton>button{
            background: linear-gradient(90deg, var(--sky), var(--blue));
            color:white; border:none; border-radius:999px;
            padding:0.95rem 2.2rem; font-weight:700; font-size:1.1rem;
            box-shadow: 0 10px 30px rgba(37,99,235,.35);
            transition: all .25s ease;
        }
        .stButton>button:hover{ transform: translateY(-3px) scale(1.03); box-shadow:0 16px 40px rgba(37,99,235,.5); }
        .stButton>button:active{ transform: translateY(-1px) scale(1.0); }

        .tw-back-btn button{
            background: rgba(15,23,42,0.08) !important;
            color: var(--text-dark) !important;
            font-size: 0.95rem !important;
            padding: 0.55rem 1.3rem !important;
            box-shadow: none !important;
        }
        .tw-back-btn button:hover{
            background: rgba(15,23,42,0.15) !important;
            transform: translateX(-3px) !important;
        }

        .tw-glass{
            background: rgba(219,234,254,0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(30,58,138,0.15);
            border-radius: 24px;
            padding: 2.2rem;
            box-shadow: var(--shadow);
            transition: transform .3s ease, box-shadow .3s ease;
        }
        .tw-glass:hover{ transform: translateY(-6px); box-shadow: 0 26px 70px rgba(37,99,235,.18); }

        .tw-hero{ text-align:center; padding: 3.4rem 1rem 1.2rem 1rem; }
        .tw-hero h1{
            font-size: 3.4rem; font-weight:800; color:var(--text-dark);
            line-height:1.18; margin-bottom:1.2rem;
            animation: fadeUp .9s ease both;
        }
        .tw-hero p{
            font-size:1.3rem; color:var(--text-muted); max-width:700px;
            margin: 0 auto 2.2rem auto; line-height:1.7;
            animation: fadeUp .9s ease .15s both;
        }
        .tw-badge{
            display:inline-block; padding:.5rem 1.2rem; border-radius:999px;
            background: rgba(14,165,233,.14); color:var(--blue); font-weight:700;
            font-size:.95rem; margin-bottom:1.4rem; letter-spacing:.02em;
            animation: fadeUp .9s ease both;
        }

        @keyframes fadeUp{ from{opacity:0; transform: translateY(16px);} to{opacity:1; transform: translateY(0);} }
        @keyframes floaty{ 0%,100%{ transform: translateY(0px) rotate(0deg);} 50%{ transform: translateY(-14px) rotate(2deg);} }
        .tw-float{ animation: floaty 5s ease-in-out infinite; display:inline-block; }
        .tw-float-slow{ animation: floaty 7s ease-in-out infinite; display:inline-block; }

        .tw-illustration-row{ display:flex; justify-content:center; gap:2.2rem; font-size:4.4rem; margin: 1.4rem 0 2.6rem 0; }

        .tw-icon-circle{
            width:84px; height:84px; border-radius:22px;
            background: linear-gradient(135deg, var(--sky), var(--blue));
            display:flex; align-items:center; justify-content:center;
            font-size:2.4rem; margin-bottom:.9rem; color:white;
            box-shadow: 0 12px 28px rgba(14,165,233,.32);
        }

        .tw-result-card{
            background: linear-gradient(135deg, #EEF4FC 0%, #E1EBFA 100%);
            border-radius:20px; padding:1.6rem 1.8rem;
            box-shadow: var(--shadow); margin-bottom:1.1rem;
            border-left: 6px solid var(--blue);
        }
        .tw-score-pill{
            display:inline-block; padding:.35rem .9rem; border-radius:999px;
            background: rgba(30,58,138,.14); color:var(--blue); font-weight:800; font-size:.9rem;
        }

        /* ---------------- RESULT DETAIL ROWS (hotel + airport) ---------------- */
        .tw-detail{
            display:flex; align-items:flex-start; gap:.6rem;
            margin-top:.6rem; color:#1E293B; font-size:1.04rem; line-height:1.5;
        }
        .tw-detail b{ color:#0F172A; }
        .tw-detail-ic{ font-size:1.15rem; line-height:1.4; flex:0 0 auto; }
        .tw-detail-muted{ color:#64748B; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SPLASH SCREEN
# ----------------------------------------------------------------------------
def render_splash():
    st.markdown(
        f"""
        <style>
        .tw-splash-wrap{{
            position:fixed; inset:0; z-index:9999;
            background: linear-gradient(180deg, #DFF6FB 0%, #B9E7F2 35%, #79CFE0 70%, #4CB8CE 100%);
            display:flex; align-items:center; justify-content:center;
            animation: splashFade 5s ease forwards;
            overflow:hidden;
        }}
        .tw-wave{{
            position:absolute; bottom:-10%; left:-10%; width:120%; height:40%;
            background: rgba(255,255,255,0.25);
            border-radius:45%;
            animation: waveSway 8s ease-in-out infinite;
        }}
        .tw-wave.tw-wave-2{{
            bottom:-16%; background: rgba(255,255,255,0.18);
            animation: waveSway 10s ease-in-out infinite reverse;
        }}
        @keyframes waveSway{{
            0%,100%{{ transform: translateX(0) translateY(0) rotate(0deg); }}
            50%{{ transform: translateX(-3%) translateY(1%) rotate(1deg); }}
        }}
        @keyframes splashFade{{ 0%{{opacity:0;}} 8%{{opacity:1;}} 88%{{opacity:1;}} 100%{{opacity:0;}} }}
        .tw-splash-canvas{{
            position:relative; width:min(90vw,1100px); aspect-ratio:1200/750;
            animation: mapZoom 5s ease forwards;
        }}
        @keyframes mapZoom{{ 0%{{transform:scale(1.12);}} 100%{{transform:scale(1);}} }}
        .tw-flight-scale{{
            position:absolute; inset:0;
            transform: scale(0.72);
            transform-origin: 50% 50%;
        }}
        .tw-cloud{{ position:absolute; opacity:.55; filter:blur(0.5px); animation: cloudDrift linear infinite; }}
        @keyframes cloudDrift{{ from{{transform: translateX(0);}} to{{transform: translateX(60px);}} }}

        /* glowing gradient trail that draws itself as the plane advances */
        .tw-trail, .tw-trail-glow{{
            stroke-dasharray:1; stroke-dashoffset:1;
            animation: trailDraw 4s cubic-bezier(.45,.05,.55,.95) .3s forwards;
        }}
        @keyframes trailDraw{{ to{{ stroke-dashoffset:0; }} }}

        /* the aircraft — hidden until motion begins, then a gentle bob */
        .tw-carrier{{ opacity:0; animation: carrierShow .01s .3s forwards; }}
        @keyframes carrierShow{{ to{{ opacity:1; }} }}
        .tw-bob{{ animation: bob 1.5s ease-in-out infinite; }}
        @keyframes bob{{ 0%,100%{{ transform: translateY(0); }} 50%{{ transform: translateY(-2.4px); }} }}

        /* arrival marker glowing in when the plane lands */
        .tw-arrive{{ opacity:0; animation: arriveIn .6s ease 4.05s forwards; transform-origin:center; }}
        .tw-arrive .halo{{ animation: arriveHalo 1.8s ease-out 4.1s infinite; transform-origin:center; }}
        @keyframes arriveIn{{ from{{ opacity:0; }} to{{ opacity:1; }} }}
        @keyframes arriveHalo{{ 0%{{ r:6; opacity:.6; }} 70%,100%{{ r:24; opacity:0; }} }}

        .tw-splash-logo{{
            position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
            flex-direction:column; opacity:0; animation: logoIn 1.1s ease 4.0s forwards;
        }}
        @keyframes logoIn{{ 0%{{opacity:0; transform:scale(.88);}} 100%{{opacity:1; transform:scale(1);}} }}
        .tw-splash-logo .tw-splash-title{{
            font-family:'Outfit', sans-serif; font-weight:800; font-size:2.8rem; color:white;
            letter-spacing:.03em; text-shadow: 0 8px 30px rgba(0,0,0,.18);
        }}
        .tw-splash-logo .tw-splash-sub{{
            font-family:'Inter', sans-serif; color:rgba(255,255,255,.92); margin-top:.5rem;
            font-size:1.05rem; letter-spacing:.06em; text-transform:uppercase; font-weight:500;
        }}

        @media (prefers-reduced-motion: reduce){{
            .tw-splash-wrap{{ animation: splashFade 5s linear forwards; }}
            .tw-trail, .tw-trail-glow, .tw-carrier{{ animation: none; opacity:1; stroke-dashoffset:0; }}
        }}
        </style>

        <div class="tw-splash-wrap">
          <div class="tw-wave"></div>
          <div class="tw-wave tw-wave-2"></div>
          <div class="tw-splash-canvas">
            <div class="tw-flight-scale">
              <svg viewBox="0 0 1200 750" width="100%" height="100%"
                   xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
                   style="position:absolute; inset:0; overflow:visible;">
                <defs>
                  <linearGradient id="twTrailGrad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%"  stop-color="#0EA5E9"/>
                    <stop offset="100%" stop-color="#2563EB"/>
                  </linearGradient>
                  <radialGradient id="twArrive">
                    <stop offset="0%"  stop-color="#38BDF8"/>
                    <stop offset="100%" stop-color="#2563EB"/>
                  </radialGradient>
                  <!-- soft neon bloom for the trail -->
                  <filter id="twTrailGlow" x="-40%" y="-40%" width="180%" height="180%">
                    <feGaussianBlur stdDeviation="6"/>
                  </filter>
                  <!-- glow that hugs the aircraft -->
                  <filter id="twPlaneGlow" x="-90%" y="-90%" width="280%" height="280%">
                    <feDropShadow dx="0" dy="0" stdDeviation="5" flood-color="#38BDF8" flood-opacity=".9"/>
                    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0B3B5C" flood-opacity=".45"/>
                  </filter>
                </defs>

                <!-- decorative cloud blobs -->
                <g fill="#ffffff" opacity="0.28">
                  <ellipse cx="180" cy="180" rx="150" ry="90"/>
                  <ellipse cx="420" cy="140" rx="120" ry="70"/>
                  <ellipse cx="230" cy="420" rx="140" ry="110"/>
                  <ellipse cx="650" cy="200" rx="180" ry="100"/>
                  <ellipse cx="780" cy="450" rx="160" ry="120"/>
                  <ellipse cx="1020" cy="250" rx="150" ry="90"/>
                  <ellipse cx="980" cy="560" rx="120" ry="80"/>
                </g>

                <!-- dotted guide path (the reference the plane rides on) -->
                <path id="twHeartPath" d="{HEART_PATH}"
                      fill="none" stroke="#0F172A" stroke-width="4"
                      stroke-dasharray="2 14" stroke-linecap="round" opacity="0.5"/>

                <!-- glowing trail drawn along the exact same curve -->
                <path class="tw-trail-glow" d="{HEART_PATH}" pathLength="1"
                      fill="none" stroke="url(#twTrailGrad)" stroke-width="10"
                      stroke-linecap="round" filter="url(#twTrailGlow)" opacity="0.55"/>
                <path class="tw-trail" d="{HEART_PATH}" pathLength="1"
                      fill="none" stroke="url(#twTrailGrad)" stroke-width="4.5"
                      stroke-linecap="round"/>

                <!-- departure pin -->
                <g transform="translate(105,635)">
                  <path d="M15 0 C24 0 30 7 30 15 C30 26 15 40 15 40 C15 40 0 26 0 15 C0 7 6 0 15 0 Z" fill="#EF4444"/>
                  <circle cx="15" cy="15" r="7" fill="white"/>
                </g>

                <!-- arrival marker (glows in on landing) -->
                <g class="tw-arrive" transform="translate(1120,90)">
                  <circle class="halo" r="6" fill="#38BDF8"/>
                  <circle r="8.5" fill="#ffffff"/>
                  <circle r="5" fill="url(#twArrive)"/>
                </g>

                <!-- the aircraft: rides #twHeartPath via animateMotion + mpath -->
                <g class="tw-carrier" filter="url(#twPlaneGlow)">
                  <g class="tw-bob">
                    <g transform="scale(1.05)">
                      <path d="{PLANE_SHAPE}" fill="#0B3B5C"/>
                      <path d="M22,0 L8,-3.4 L-4,-3.4 L-4,3.4 L8,3.4 Z" fill="#EAF7FE" opacity=".75"/>
                    </g>
                  </g>
                  <animateMotion dur="4s" begin=".3s" fill="freeze" rotate="auto"
                                 calcMode="spline" keyPoints="0;1" keyTimes="0;1"
                                 keySplines=".45 .05 .55 .95">
                    <mpath href="#twHeartPath" xlink:href="#twHeartPath"/>
                  </animateMotion>
                </g>
              </svg>
            </div>

            <div class="tw-cloud" style="top:12%; left:8%; font-size:2.2rem; animation-duration:9s;">☁️</div>
            <div class="tw-cloud" style="top:28%; left:60%; font-size:1.6rem; animation-duration:12s;">☁️</div>
            <div class="tw-cloud" style="top:65%; left:20%; font-size:1.8rem; animation-duration:10s;">☁️</div>

            <div class="tw-splash-logo">
              <div class="tw-splash-title">✈️ TripWise AI</div>
              <div class="tw-splash-sub">Plan smarter. Travel further.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
NAV_ITEMS = [
    ("Home", "🏠"),
    ("Plan Trip", "🧳"),
    ("AI Travel Assistant", "🤖"),
    ("Destination Explorer", "📍"),
    ("Budget Planner", "💰"),
    ("Weather", "🌦️"),
    ("Saved Trips", "❤️"),
    ("Settings", "⚙️"),
]


def go_to(page: str):
    """Navigate to a page, pushing the current one onto history for Back support."""
    if st.session_state.page != page:
        st.session_state.history.append(st.session_state.page)
    st.session_state.page = page
    st.rerun()


def go_back():
    if st.session_state.history:
        st.session_state.page = st.session_state.history.pop()
    else:
        st.session_state.page = "Home"
    st.rerun()


def render_back_button():
    if st.session_state.page != "Home":
        st.markdown('<div class="tw-back-btn">', unsafe_allow_html=True)
        if st.button("← Back", key="back_btn"):
            go_back()
        st.markdown('</div>', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown('<span class="tw-logo">✈️ TripWise AI</span>', unsafe_allow_html=True)
        for label, icon in NAV_ITEMS:
            if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
                go_to(label)


# ----------------------------------------------------------------------------
# PAGE: HOME
# ----------------------------------------------------------------------------
def page_home(df: pd.DataFrame):
    st.markdown(
        """
        <div class="tw-hero">
            <span class="tw-badge">✨ AI-Powered Travel Planning</span>
            <h1>✈️ Plan Smarter with<br/>TripWise AI</h1>
            <p>Your intelligent travel companion that creates personalized itineraries,
            predicts travel costs, recommends destinations, and helps you explore the
            world effortlessly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col = st.columns([1, 1, 1])[1]
    with col:
        if st.button("Start Planning →", use_container_width=True):
            go_to("Destination Explorer")

    st.markdown(
        """
        <div class="tw-illustration-row">
            <span class="tw-float">🗺️</span>
            <span class="tw-float-slow">🧳</span>
            <span class="tw-float">🏝️</span>
            <span class="tw-float-slow">🛫</span>
            <span class="tw-float">🏨</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="tw-section-title">Everything you need to travel smarter</div>', unsafe_allow_html=True)
    features = [
        ("🎯", "Personalized Recommendations", "Destinations matched to your taste, budget and travel style."),
        ("💸", "Smart Budget Planning", "See realistic cost estimates before you book anything."),
        ("🌍", "Global Coverage", "Thousands of cities, hotels and airports at your fingertips."),
    ]
    cols = st.columns(3)
    for c, (icon, title, desc) in zip(cols, features):
        with c:
            st.markdown(
                f"""
                <div class="tw-glass">
                    <div class="tw-icon-circle">{icon}</div>
                    <div style="font-weight:800; font-size:1.25rem; margin-bottom:.6rem; color:#0F172A;">{title}</div>
                    <div class="tw-muted" style="font-size:1.02rem; line-height:1.6;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_insights(df)


def render_insights(df: pd.DataFrame):
    st.markdown('<div class="tw-section-title">📊 Key Insights</div>', unsafe_allow_html=True)

    plot_bg = "rgba(0,0,0,0)"
    accent_scale = ["#0EA5E9", "#2563EB", "#38BDF8", "#60A5FA", "#93C5FD", "#0284C7", "#1D4ED8"]

    i1, i2 = st.columns(2)

    with i1:
        st.markdown('<div class="tw-glass">', unsafe_allow_html=True)
        region_counts = (
            df[REGION_COLS].sum().rename(lambda c: c.replace("region_", "").replace("_", " ").title())
        )
        fig1 = px.pie(
            names=region_counts.index, values=region_counts.values, hole=0.55,
            color_discrete_sequence=accent_scale,
        )
        fig1.update_layout(
            template="plotly_white",
            title="Destinations by Region", paper_bgcolor=plot_bg, plot_bgcolor=plot_bg,
            font=dict(color="#0F172A", size=13),
            title_font=dict(color="#0F172A", size=15),
            legend=dict(font=dict(color="#0F172A")),
            margin=dict(t=50, b=10, l=10, r=10), height=340,
        )
        fig1.update_traces(textfont_color="#0F172A")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with i2:
        st.markdown('<div class="tw-glass">', unsafe_allow_html=True)
        budget_labels = {1: "Budget", 2: "Mid-range", 3: "Luxury"}
        budget_counts = df["budget_level_encoded"].map(budget_labels).value_counts()
        fig2 = px.bar(
            x=budget_counts.index, y=budget_counts.values,
            color=budget_counts.index, color_discrete_sequence=accent_scale,
            labels={"x": "", "y": "Destinations"},
        )
        fig2.update_layout(
            template="plotly_white",
            title="Destinations by Budget Level", paper_bgcolor=plot_bg, plot_bgcolor=plot_bg,
            font=dict(color="#0F172A", size=13),
            title_font=dict(color="#0F172A", size=15),
            xaxis=dict(color="#0F172A", tickfont=dict(color="#0F172A")),
            yaxis=dict(color="#0F172A", tickfont=dict(color="#0F172A")),
            showlegend=False, margin=dict(t=50, b=10, l=10, r=10), height=340,
        )
        fig2.update_traces(textfont_color="#0F172A")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="tw-glass" style="margin-top:1.2rem;">', unsafe_allow_html=True)
    region_temp = df.assign(
        region=df[REGION_COLS].idxmax(axis=1).str.replace("region_", "").str.replace("_", " ").str.title()
    ).groupby("region")["temp_avg_yearly"].mean().sort_values()
    fig3 = px.bar(
        x=region_temp.values, y=region_temp.index, orientation="h",
        color=region_temp.values, color_continuous_scale=["#93C5FD", "#0EA5E9", "#2563EB"],
        labels={"x": "Avg. Temperature (°C)", "y": ""},
    )
    fig3.update_layout(
        template="plotly_white",
        title="Average Yearly Temperature by Region", paper_bgcolor=plot_bg, plot_bgcolor=plot_bg,
        font=dict(color="#0F172A", size=13),
        title_font=dict(color="#0F172A", size=15),
        xaxis=dict(color="#0F172A", tickfont=dict(color="#0F172A")),
        yaxis=dict(color="#0F172A", tickfont=dict(color="#0F172A")),
        coloraxis_showscale=False, margin=dict(t=50, b=10, l=10, r=10), height=380,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE: DESTINATION EXPLORER (real recommendation engine)
# ----------------------------------------------------------------------------
def _clean(value):
    """Return a stripped string, or None for missing / placeholder values."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"unknown", "not specified", "nan", "none"}:
        return None
    return text


def page_destination_explorer(df: pd.DataFrame):
    st.markdown('<div class="tw-section-title">📍 Destination Explorer</div>', unsafe_allow_html=True)
    st.markdown('<p class="tw-muted">Tell us what you love, and we\'ll match you with the best destinations.</p>', unsafe_allow_html=True)

    scaler, X_scaled = build_scaler(df)

    with st.form("preferences_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            culture = st.slider("🏛️ Culture", 1, 5, 3)
            adventure = st.slider("🧗 Adventure", 1, 5, 3)
            nature = st.slider("🌿 Nature", 1, 5, 3)
        with c2:
            beaches = st.slider("🏖️ Beaches", 1, 5, 3)
            nightlife = st.slider("🌃 Nightlife", 1, 5, 3)
            cuisine = st.slider("🍽️ Cuisine", 1, 5, 3)
        with c3:
            wellness = st.slider("🧘 Wellness", 1, 5, 3)
            urban = st.slider("🏙️ Urban", 1, 5, 3)
            seclusion = st.slider("🏝️ Seclusion", 1, 5, 3)

        c4, c5, c6 = st.columns(3)
        with c4:
            budget_label = st.selectbox("💰 Budget level", ["Budget", "Mid-range", "Luxury"], index=1)
        with c5:
            region_label = st.selectbox(
                "🌍 Preferred region",
                ["Any", "Africa", "Asia", "Europe", "Middle East", "North America", "Oceania", "South America"],
            )
        with c6:
            temp_pref = st.slider("🌡️ Preferred avg. temperature (°C)", -5, 40, 22)

        c7, c8 = st.columns(2)
        with c7:
            trip_length = st.radio("🗓️ Trip length", ["Short trip", "One week"], horizontal=True)
        with c8:
            wants_airport = st.checkbox("✈️ Must have a nearby airport", value=True)

        submitted = st.form_submit_button("Find my destinations →", use_container_width=True)

    if submitted:
        budget_map = {"Budget": 1, "Mid-range": 2, "Luxury": 3}
        region_map = {
            "Africa": "region_africa", "Asia": "region_asia", "Europe": "region_europe",
            "Middle East": "region_middle_east", "North America": "region_north_america",
            "Oceania": "region_oceania", "South America": "region_south_america",
        }

        user_preferences = {col: 0 for col in FEATURE_COLS}
        user_preferences.update({
            "latitude": df["latitude"].mean(),
            "longitude": df["longitude"].mean(),
            "culture": culture, "adventure": adventure, "nature": nature,
            "beaches": beaches, "nightlife": nightlife, "cuisine": cuisine,
            "wellness": wellness, "urban": urban, "seclusion": seclusion,
            "has_airport": int(wants_airport),
            "is_short_trip": int(trip_length == "Short trip"),
            "is_one_week": int(trip_length == "One week"),
            "temp_avg_yearly": temp_pref,
            "budget_level_encoded": budget_map[budget_label],
            "HotelRating_encoded": 4,
            "rating_was_unknown": 0,
        })
        if region_label != "Any":
            user_preferences[region_map[region_label]] = 1

        user_df = pd.DataFrame([user_preferences])[FEATURE_COLS]
        user_scaled = scaler.transform(user_df)

        similarities = cosine_similarity(user_scaled, X_scaled)[0]
        results = df.copy()
        results["similarity_score"] = similarities
        top = (
            results.sort_values("similarity_score", ascending=False)
            .drop_duplicates(subset=["city", "country"])
            .head(8)
        )

        st.markdown('<div class="tw-section-title">Top matches for you</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="tw-muted" style="margin-top:-.6rem;">'
            'Each card is a suggested hotel in that city, with its nearest airport.</p>',
            unsafe_allow_html=True,
        )

        for _, row in top.iterrows():
            match_pct = round(row["similarity_score"] * 100, 1)

            hotel_name = _clean(row.get("HotelName"))
            airport_name = _clean(row.get("name"))       # airport name from the airports merge
            airport_iata = _clean(row.get("iata"))
            has_air = int(row.get("has_airport", 0)) == 1 and airport_name is not None

            # Line 1 — make it explicit this is a hotel pick for this city.
            if hotel_name:
                hotel_html = (
                    f'<div class="tw-detail"><span class="tw-detail-ic">🏨</span>'
                    f'<span><b>Suggested hotel</b> in {row["city"]} — {hotel_name}</span></div>'
                )
            else:
                hotel_html = (
                    f'<div class="tw-detail"><span class="tw-detail-ic">🏨</span>'
                    f'<span><b>Suggested stay</b> in {row["city"]}</span></div>'
                )

            # Line 2 — nearest airport, name + IATA code, or a clear fallback.
            if has_air:
                code = f' <span class="tw-score-pill" style="font-size:.78rem;">{airport_iata}</span>' if airport_iata else ""
                airport_html = (
                    f'<div class="tw-detail"><span class="tw-detail-ic">🛫</span>'
                    f'<span><b>Nearest airport</b> — {airport_name}{code}</span></div>'
                )
            else:
                airport_html = (
                    '<div class="tw-detail tw-detail-muted"><span class="tw-detail-ic">🛫</span>'
                    '<span>No major airport on record for this city</span></div>'
                )

            st.markdown(
                f"""
                <div class="tw-result-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:.5rem;">
                        <div style="font-weight:800; font-size:1.35rem; color:#0F172A;">📍 {row['city']}, {row['country']}</div>
                        <span class="tw-score-pill">{match_pct}% match</span>
                    </div>
                    {hotel_html}
                    {airport_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# PAGE: GENERIC PLACEHOLDER
# ----------------------------------------------------------------------------
def page_placeholder(title, icon, description):
    st.markdown(f'<div class="tw-section-title">{icon} {title}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="tw-glass" style="text-align:center; padding:3.6rem 2rem;">
            <div style="font-size:3rem; margin-bottom:1.1rem;">{icon}</div>
            <div style="font-weight:800; font-size:1.6rem; margin-bottom:.7rem; color:#0F172A;">{title} is coming soon</div>
            <div class="tw-muted" style="font-size:1.08rem;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


PAGE_CONTENT = {
    "Plan Trip": ("🧳", "Build a day-by-day itinerary tailored to your preferences and travel dates."),
    "AI Travel Assistant": ("🤖", "Chat with an AI assistant that answers your travel questions in real time."),
    "Budget Planner": ("💰", "Estimate flights, hotels, and daily costs for your next trip."),
    "Weather": ("🌦️", "Check destination forecasts before you pack."),
    "Saved Trips": ("❤️", "All the destinations and plans you've bookmarked, in one place."),
    "Settings": ("⚙️", "Manage your profile, preferences, and app settings."),
}


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    inject_css()

    if "splash_done" not in st.session_state:
        st.session_state.splash_done = False
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "history" not in st.session_state:
        st.session_state.history = []

    if not st.session_state.splash_done:
        placeholder = st.empty()
        with placeholder.container():
            render_splash()
        time.sleep(4.8)
        st.session_state.splash_done = True
        placeholder.empty()
        st.rerun()
        return

    render_sidebar()
    render_back_button()

    if st.session_state.page == "Home":
        df = load_data()
        page_home(df)
    elif st.session_state.page == "Destination Explorer":
        df = load_data()
        page_destination_explorer(df)
    else:
        icon, desc = PAGE_CONTENT[st.session_state.page]
        page_placeholder(st.session_state.page, icon, desc)


if __name__ == "__main__":
    main()
