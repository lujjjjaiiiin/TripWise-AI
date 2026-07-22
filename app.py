
TripWise AI — Premium Streamlit Travel Assistant
Splash screen (SVG heart-path flight animation) -> Main App (Home / nav pages)
Destination Explorer page runs the real Cosine Similarity recommendation engine.
"""

import time
import pandas as pd
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
            --text-muted:#64748B;
            --shadow: 0 20px 60px rgba(37,99,235,0.12);
        }

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        h1,h2,h3, .tw-display { font-family: 'Outfit', sans-serif; }

        #MainMenu, footer, header {visibility:hidden;}
        .block-container{ padding-top:1rem; padding-bottom:2rem; max-width: 1200px;}
        body{ background: linear-gradient(180deg, #F8FBFF 0%, #EFF5FC 100%); }

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
            padding:0.7rem 1.8rem; font-weight:600; font-size:1rem;
            box-shadow: 0 10px 30px rgba(37,99,235,.35);
            transition: all .25s ease;
        }
        .stButton>button:hover{ transform: translateY(-2px) scale(1.02); box-shadow:0 14px 36px rgba(37,99,235,.45); }

        .tw-glass{
            background: rgba(255,255,255,0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.6);
            border-radius: 24px;
            padding: 1.8rem;
            box-shadow: var(--shadow);
            transition: transform .3s ease, box-shadow .3s ease;
        }
        .tw-glass:hover{ transform: translateY(-6px); box-shadow: 0 26px 70px rgba(37,99,235,.18); }

        .tw-hero{ text-align:center; padding: 3.2rem 1rem 1rem 1rem; }
        .tw-hero h1{
            font-size: 3.1rem; font-weight:800; color:var(--text-dark);
            line-height:1.15; margin-bottom:1rem;
            animation: fadeUp .9s ease both;
        }
        .tw-hero p{
            font-size:1.15rem; color:var(--text-muted); max-width:680px;
            margin: 0 auto 2rem auto; line-height:1.6;
            animation: fadeUp .9s ease .15s both;
        }
        .tw-badge{
            display:inline-block; padding:.4rem 1rem; border-radius:999px;
            background: rgba(14,165,233,.12); color:var(--blue); font-weight:600;
            font-size:.85rem; margin-bottom:1.2rem; letter-spacing:.02em;
            animation: fadeUp .9s ease both;
        }

        @keyframes fadeUp{ from{opacity:0; transform: translateY(16px);} to{opacity:1; transform: translateY(0);} }
        @keyframes floaty{ 0%,100%{ transform: translateY(0px) rotate(0deg);} 50%{ transform: translateY(-14px) rotate(2deg);} }
        .tw-float{ animation: floaty 5s ease-in-out infinite; display:inline-block; }
        .tw-float-slow{ animation: floaty 7s ease-in-out infinite; display:inline-block; }

        .tw-illustration-row{ display:flex; justify-content:center; gap:2.2rem; font-size:2.6rem; margin: 1.2rem 0 2.4rem 0; }
        .tw-section-title{ font-weight:700; font-size:1.6rem; color:var(--text-dark); margin: 2.2rem 0 1rem 0;}
        .tw-muted{ color:var(--text-muted); }

        .tw-icon-circle{
            width:52px; height:52px; border-radius:16px;
            background: linear-gradient(135deg, var(--sky), var(--blue));
            display:flex; align-items:center; justify-content:center;
            font-size:1.4rem; margin-bottom:.8rem; color:white;
            box-shadow: 0 10px 24px rgba(14,165,233,.3);
        }

        .tw-result-card{
            background:white; border-radius:20px; padding:1.4rem 1.6rem;
            box-shadow: var(--shadow); margin-bottom:1rem;
            border-left: 6px solid var(--sky);
        }
        .tw-score-pill{
            display:inline-block; padding:.25rem .7rem; border-radius:999px;
            background: rgba(37,99,235,.1); color:var(--blue); font-weight:700; font-size:.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SPLASH SCREEN
# ----------------------------------------------------------------------------
def render_splash():
    st.markdown(
        """
        <style>
        .tw-splash-wrap{
            position:fixed; inset:0; z-index:9999;
            background: radial-gradient(circle at 30% 20%, #BFEFFF 0%, #7FD9F0 45%, #4FC3E0 100%);
            display:flex; align-items:center; justify-content:center;
            animation: splashFade 5s ease forwards;
            overflow:hidden;
        }
        @keyframes splashFade{ 0%{opacity:0;} 8%{opacity:1;} 88%{opacity:1;} 100%{opacity:0;} }
        .tw-splash-canvas{
            position:relative; width:min(90vw,1100px); aspect-ratio:1200/750;
            animation: mapZoom 5s ease forwards;
        }
        @keyframes mapZoom{ 0%{transform:scale(1.12);} 100%{transform:scale(1);} }
        .tw-cloud{ position:absolute; opacity:.55; filter:blur(0.5px); animation: cloudDrift linear infinite; }
        @keyframes cloudDrift{ from{transform: translateX(0);} to{transform: translateX(60px);} }
        .tw-plane{
            position:absolute; top:0; left:0; font-size:2.6rem;
            offset-path: path("M 120 660 C 230 600 300 520 360 460 C 300 430 250 380 255 310 C 260 220 340 155 430 165 C 490 172 535 210 555 260 C 575 210 620 172 680 165 C 770 155 850 220 855 310 C 860 380 810 430 750 460 C 820 500 900 500 940 430 C 985 355 985 270 940 200 C 1005 165 1060 130 1120 90");
            offset-rotate: auto;
            animation: flyMotion 4.6s cubic-bezier(.45,.05,.55,.95) forwards, planeFloat 1.4s ease-in-out infinite;
        }
        @keyframes flyMotion{ 0%{offset-distance:0%; opacity:0;} 5%{opacity:1;} 96%{offset-distance:100%; opacity:1;} 100%{offset-distance:100%; opacity:0;} }
        @keyframes planeFloat{ 0%,100%{margin-top:0px;} 50%{margin-top:-6px;} }
        .tw-splash-logo{
            position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
            flex-direction:column; opacity:0; animation: logoIn 1.1s ease 4.0s forwards;
        }
        @keyframes logoIn{ 0%{opacity:0; transform:scale(.88);} 100%{opacity:1; transform:scale(1);} }
        .tw-splash-logo .tw-splash-title{ font-family:'Outfit', sans-serif; font-weight:800; font-size:2.6rem; color:white; text-shadow: 0 8px 30px rgba(0,0,0,.15); }
        .tw-splash-logo .tw-splash-sub{ font-family:'Inter', sans-serif; color:rgba(255,255,255,.9); margin-top:.4rem; font-size:1rem; }
        </style>

        <div class="tw-splash-wrap">
          <div class="tw-splash-canvas">
            <svg viewBox="0 0 1200 750" width="100%" height="100%" style="position:absolute; inset:0;">
              <g fill="#ffffff" opacity="0.28">
                <ellipse cx="180" cy="180" rx="150" ry="90"/>
                <ellipse cx="420" cy="140" rx="120" ry="70"/>
                <ellipse cx="230" cy="420" rx="140" ry="110"/>
                <ellipse cx="650" cy="200" rx="180" ry="100"/>
                <ellipse cx="780" cy="450" rx="160" ry="120"/>
                <ellipse cx="1020" cy="250" rx="150" ry="90"/>
                <ellipse cx="980" cy="560" rx="120" ry="80"/>
              </g>
              <path d="M 120 660 C 230 600 300 520 360 460 C 300 430 250 380 255 310 C 260 220 340 155 430 165 C 490 172 535 210 555 260 C 575 210 620 172 680 165 C 770 155 850 220 855 310 C 860 380 810 430 750 460 C 820 500 900 500 940 430 C 985 355 985 270 940 200 C 1005 165 1060 130 1120 90"
                    fill="none" stroke="#0F172A" stroke-width="4" stroke-dasharray="2 14" stroke-linecap="round" opacity="0.75"/>
              <g transform="translate(105,635)">
                <path d="M15 0 C24 0 30 7 30 15 C30 26 15 40 15 40 C15 40 0 26 0 15 C0 7 6 0 15 0 Z" fill="#EF4444"/>
                <circle cx="15" cy="15" r="7" fill="white"/>
              </g>
            </svg>
            <div class="tw-cloud" style="top:12%; left:8%; font-size:2.2rem; animation-duration:9s;">☁️</div>
            <div class="tw-cloud" style="top:28%; left:60%; font-size:1.6rem; animation-duration:12s;">☁️</div>
            <div class="tw-cloud" style="top:65%; left:20%; font-size:1.8rem; animation-duration:10s;">☁️</div>
            <div class="tw-plane">✈️</div>
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


def render_sidebar():
    with st.sidebar:
        st.markdown('<span class="tw-logo">✈️ TripWise AI</span>', unsafe_allow_html=True)
        for label, icon in NAV_ITEMS:
            if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label
                st.rerun()


# ----------------------------------------------------------------------------
# PAGE: HOME
# ----------------------------------------------------------------------------
def page_home():
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
            st.session_state.page = "Destination Explorer"
            st.rerun()

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
                    <div style="font-weight:700; font-size:1.05rem; margin-bottom:.4rem;">{title}</div>
                    <div class="tw-muted" style="font-size:.92rem; line-height:1.5;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# PAGE: DESTINATION EXPLORER (real recommendation engine)
# ----------------------------------------------------------------------------
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
        for _, row in top.iterrows():
            match_pct = round(row["similarity_score"] * 100, 1)
            hotel_name = row["HotelName"] if "HotelName" in row and pd.notna(row["HotelName"]) else ""
            st.markdown(
                f"""
                <div class="tw-result-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-weight:700; font-size:1.15rem;">📍 {row['city']}, {row['country']}</div>
                        <span class="tw-score-pill">{match_pct}% match</span>
                    </div>
                    <div class="tw-muted" style="margin-top:.4rem; font-size:.92rem;">{hotel_name}</div>
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
        <div class="tw-glass" style="text-align:center; padding:3.2rem 2rem;">
            <div style="font-size:2.6rem; margin-bottom:1rem;">{icon}</div>
            <div style="font-weight:700; font-size:1.3rem; margin-bottom:.6rem;">{title} is coming soon</div>
            <div class="tw-muted">{description}</div>
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

    if st.session_state.page == "Home":
        page_home()
    elif st.session_state.page == "Destination Explorer":
        df = load_data()
        page_destination_explorer(df)
    else:
        icon, desc = PAGE_CONTENT[st.session_state.page]
        page_placeholder(st.session_state.page, icon, desc)


if __name__ == "__main__":
    main()
