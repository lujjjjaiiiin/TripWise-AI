"""
TripWise AI — an AI travel platform (single-file build).

    streamlit run app.py

This is the packaged form of the modular project: the same code with every
module inlined, so it deploys anywhere without a package folder alongside it.
Section banners below mark what was theme.py, art.py, engine.py, data.py,
ui.py, splash.py and app.py.

Python does the thinking — loading the catalogue, ranking destinations, deriving
seasons, costs and insights. Everything visible is custom markup styled by the
design system; Streamlit's own interface is hidden.
"""

from __future__ import annotations

import hashlib
import math
import os
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="TripWise AI — Intelligent travel planning",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================================================
# DESIGN SYSTEM — The whole visual language in one place.
# ==========================================================================

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Plus+Jakarta+Sans:wght@400;500;600;700;800&"
    "family=Inter:wght@400;450;500;600&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)

THEME_CSS = """
<style>
@import url('__FONTS__');

/* ========================================================================
   TOKENS
   ===================================================================== */
:root{
  --ink:#080D18;
  --ink-2:#1B2536;
  --slate:#55637A;
  --slate-2:#8494A8;
  --sky:#0EA5E9;
  --blue:#2563EB;
  --deep:#1E3A8A;
  --line:rgba(9,16,32,.08);
  --line-2:rgba(9,16,32,.14);
  --glass:rgba(255,255,255,.72);
  --glass-2:rgba(255,255,255,.55);

  --r-xl:34px; --r-lg:24px; --r-md:16px; --r-sm:10px;

  --sh-1:0 1px 2px rgba(9,16,32,.04), 0 4px 14px rgba(9,16,32,.05);
  --sh-2:0 2px 6px rgba(9,16,32,.05), 0 18px 42px rgba(37,99,235,.10);
  --sh-3:0 30px 70px rgba(37,99,235,.17);
  --sh-btn:0 10px 26px rgba(37,99,235,.32);

  --ease:cubic-bezier(.22,.75,.28,1);
  --nav-h:70px;
}

/* ========================================================================
   STREAMLIT CHROME — removed so the page reads as a website
   ===================================================================== */
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"]{ display:none !important; }
[data-testid="stHeader"]{ background:transparent !important; height:0 !important; }

.stApp{
  background:
    radial-gradient(1100px 620px at 8% -6%,  rgba(14,165,233,.16), transparent 62%),
    radial-gradient(900px 560px at 96% 2%,   rgba(37,99,235,.14),  transparent 60%),
    radial-gradient(760px 620px at 50% 108%, rgba(14,165,233,.10), transparent 62%),
    linear-gradient(180deg,#FBFDFF 0%,#F1F5FA 52%,#EAF0F7 100%);
  background-attachment:fixed;
}
.block-container{
  max-width:1180px !important;
  padding:calc(var(--nav-h) + 26px) 22px 90px 22px !important;
}
[data-testid="stVerticalBlock"]{ gap:0 !important; }
html{ scroll-behavior:smooth; }
section.main{ scroll-padding-top:calc(var(--nav-h) + 20px); }

/* ========================================================================
   TYPE
   ===================================================================== */
html, body, [class*="css"], .stApp{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  color:var(--ink);
  -webkit-font-smoothing:antialiased;
}
.tw-display{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  letter-spacing:-.035em; line-height:1.04; color:var(--ink);
}
.tw-eyebrow{
  font-family:'JetBrains Mono',ui-monospace,monospace;
  font-size:.68rem; letter-spacing:.24em; text-transform:uppercase;
  color:var(--blue); display:inline-flex; align-items:center; gap:.6rem;
}
.tw-eyebrow::before{
  content:""; width:24px; height:1px;
  background:linear-gradient(90deg,var(--sky),transparent);
}
.tw-lede{ color:var(--slate); font-size:1.06rem; line-height:1.72; }
.tw-mono{ font-family:'JetBrains Mono',ui-monospace,monospace; }
.tw-grad{
  background:linear-gradient(96deg,var(--sky) 0%,var(--blue) 55%,var(--deep) 100%);
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; color:var(--blue);
  filter:drop-shadow(0 0 26px rgba(14,165,233,.34));
}
::selection{ background:rgba(14,165,233,.22); }
:focus-visible{ outline:2px solid var(--blue); outline-offset:3px; border-radius:8px; }

/* ========================================================================
   NAVBAR
   ===================================================================== */
.tw-nav{
  position:fixed; top:0; left:0; right:0; height:var(--nav-h); z-index:9990;
  display:flex; align-items:center; justify-content:space-between;
  padding:0 clamp(18px,4vw,44px);
  background:rgba(255,255,255,.78);
  backdrop-filter:blur(20px) saturate(1.6);
  -webkit-backdrop-filter:blur(20px) saturate(1.6);
  border-bottom:1px solid var(--line);
}
.tw-nav__brand{ display:flex; align-items:center; gap:.6rem; text-decoration:none; }
.tw-nav__brand svg{ width:26px; height:26px; }
.tw-nav__brand b{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.08rem;
  letter-spacing:-.02em; color:var(--ink);
}
.tw-nav__links{ display:flex; align-items:center; gap:.35rem; }
.tw-nav__links a{
  position:relative; text-decoration:none; color:var(--slate);
  font-size:.9rem; font-weight:500; padding:.5rem .85rem; border-radius:99px;
  transition:color .25s var(--ease), background .25s var(--ease);
}
.tw-nav__links a:hover{ color:var(--ink); background:rgba(9,16,32,.045); }
.tw-nav__links a.is-active{ color:var(--blue); }
.tw-nav__links a::after{
  content:""; position:absolute; left:50%; bottom:5px; height:2px; width:0;
  background:linear-gradient(90deg,var(--sky),var(--blue));
  border-radius:2px; transform:translateX(-50%);
  transition:width .3s var(--ease);
}
.tw-nav__links a:hover::after, .tw-nav__links a.is-active::after{ width:20px; }

/* ========================================================================
   BUTTONS — one definition, used by links and Streamlit alike
   ===================================================================== */
.tw-btn, .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button{
  display:inline-flex; align-items:center; justify-content:center; gap:.55rem;
  font-family:'Inter',sans-serif; font-weight:600; font-size:.95rem;
  padding:.82rem 1.6rem; border-radius:99px; border:1px solid transparent;
  text-decoration:none; cursor:pointer; white-space:nowrap;
  background:linear-gradient(96deg,var(--sky),var(--blue));
  color:#fff !important; box-shadow:var(--sh-btn);
  transition:transform .26s var(--ease), box-shadow .26s var(--ease), filter .26s var(--ease);
}
.tw-btn:hover, .stButton>button:hover, .stFormSubmitButton>button:hover{
  transform:translateY(-2px); box-shadow:0 18px 40px rgba(37,99,235,.42);
  filter:saturate(1.08);
}
.tw-btn:active, .stButton>button:active, .stFormSubmitButton>button:active{
  transform:translateY(0);
}
.tw-btn--ghost{
  background:rgba(255,255,255,.8); color:var(--ink) !important;
  border-color:var(--line-2); box-shadow:var(--sh-1);
}
.tw-btn--ghost:hover{ border-color:var(--sky); color:var(--blue) !important; }
.tw-btn--sm{ padding:.55rem 1.1rem; font-size:.85rem; }

/* ========================================================================
   SURFACES
   ===================================================================== */
.tw-card{
  position:relative; background:var(--glass);
  backdrop-filter:blur(18px) saturate(1.4);
  -webkit-backdrop-filter:blur(18px) saturate(1.4);
  border:1px solid rgba(255,255,255,.86);
  border-radius:var(--r-lg); box-shadow:var(--sh-2);
  transition:transform .38s var(--ease), box-shadow .38s var(--ease);
}
.tw-card::before{
  content:""; position:absolute; inset:0 0 auto; height:1px;
  border-radius:var(--r-lg) var(--r-lg) 0 0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.96),transparent);
}
.tw-card:hover{ transform:translateY(-6px); box-shadow:var(--sh-3); }

.tw-section{ margin:clamp(64px,9vw,116px) 0 0; scroll-margin-top:calc(var(--nav-h) + 24px); }
.tw-section__head{ max-width:680px; margin-bottom:2.4rem; }
.tw-section__head h2{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  font-size:clamp(1.9rem,3.6vw,2.7rem); letter-spacing:-.03em;
  margin:.7rem 0 .7rem; color:var(--ink); line-height:1.1;
}
.tw-section__head p{ color:var(--slate); font-size:1.02rem; line-height:1.7; margin:0; }

/* equal-height responsive grids */
.tw-grid{ display:grid; gap:20px; }
.tw-grid--3{ grid-template-columns:repeat(auto-fit,minmax(268px,1fr)); }
.tw-grid--2{ grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }
.tw-grid > *{ height:100%; }

/* ========================================================================
   FEATURE CARD
   ===================================================================== */
.tw-feat{ padding:2rem 1.7rem; display:flex; flex-direction:column; }
.tw-feat__icon{
  width:76px; height:76px; border-radius:22px; display:grid; place-items:center;
  background:linear-gradient(140deg,var(--sky),var(--blue));
  box-shadow:0 16px 32px rgba(14,165,233,.32); margin-bottom:1.3rem; flex:0 0 auto;
}
.tw-feat__icon svg{ width:36px; height:36px; fill:none; stroke:#fff; stroke-width:1.7;
  stroke-linecap:round; stroke-linejoin:round; }
.tw-feat h3{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.14rem;
  letter-spacing:-.015em; margin:0 0 .55rem; color:var(--ink);
}
.tw-feat p{ color:var(--slate); font-size:.94rem; line-height:1.68; margin:0; }

/* ========================================================================
   DESTINATION CARD
   ===================================================================== */
.tw-dest{ overflow:hidden; display:flex; flex-direction:column; }
.tw-dest__art{ position:relative; height:186px; overflow:hidden; flex:0 0 auto; }
.tw-dest__art svg{ width:100%; height:100%; display:block;
  transition:transform .7s var(--ease); }
.tw-dest:hover .tw-dest__art svg{ transform:scale(1.07); }
.tw-dest__scrim{
  position:absolute; inset:auto 0 0 0; height:64%;
  background:linear-gradient(180deg,transparent,rgba(6,12,24,.66));
}
.tw-dest__place{ position:absolute; left:18px; bottom:14px; right:96px; }
.tw-dest__place h3{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:#fff;
  font-size:1.32rem; letter-spacing:-.02em; margin:0; line-height:1.15;
  text-shadow:0 2px 14px rgba(0,0,0,.4);
}
.tw-dest__place span{ color:rgba(255,255,255,.86); font-size:.82rem; }
.tw-dest__match{
  position:absolute; right:14px; top:14px; padding:.42rem .7rem; border-radius:99px;
  background:rgba(255,255,255,.92); backdrop-filter:blur(8px);
  font-family:'JetBrains Mono',monospace; font-size:.78rem; font-weight:500;
  color:var(--blue); box-shadow:0 6px 18px rgba(0,0,0,.18);
}
.tw-dest__body{ padding:1.25rem 1.35rem 1.4rem; display:flex; flex-direction:column;
  gap:1rem; flex:1 1 auto; }

.tw-facts{ display:grid; grid-template-columns:repeat(3,1fr); gap:.5rem; }
.tw-facts div span{
  display:block; font-family:'JetBrains Mono',monospace; font-size:.6rem;
  letter-spacing:.14em; text-transform:uppercase; color:var(--slate-2); margin-bottom:.2rem;
}
.tw-facts div b{ font-size:.94rem; font-weight:600; color:var(--ink); }

.tw-why{
  border-left:2px solid var(--sky); padding:.15rem 0 .15rem .85rem;
  color:var(--slate); font-size:.9rem; line-height:1.62;
}
.tw-why b{ color:var(--ink-2); font-weight:600; }

.tw-chips{ display:flex; flex-wrap:wrap; gap:.4rem; }
.tw-chip{
  padding:.3rem .68rem; border-radius:99px; font-size:.74rem; font-weight:500;
  background:rgba(14,165,233,.1); color:#0369A1;
}
.tw-chip--muted{ background:rgba(9,16,32,.05); color:var(--slate); }

.tw-meta{ border-top:1px solid var(--line); padding-top:.85rem; display:grid; gap:.5rem; }
.tw-meta__row{ display:flex; gap:.6rem; align-items:flex-start;
  font-size:.86rem; color:var(--slate); line-height:1.55; }
.tw-meta__row svg{ width:15px; height:15px; flex:0 0 auto; margin-top:2px;
  stroke:var(--sky); fill:none; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.tw-meta__row b{ color:var(--ink-2); font-weight:600; }

/* ========================================================================
   INSIGHT CARD
   ===================================================================== */
.tw-insight{ padding:1.5rem 1.6rem; display:flex; gap:1.1rem; align-items:flex-start; }
.tw-insight__dot{
  width:42px; height:42px; border-radius:14px; flex:0 0 auto; display:grid; place-items:center;
  background:linear-gradient(140deg,rgba(14,165,233,.16),rgba(37,99,235,.16));
  border:1px solid rgba(37,99,235,.16);
}
.tw-insight__dot svg{ width:20px; height:20px; stroke:var(--blue); fill:none;
  stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.tw-insight h4{
  font-family:'Plus Jakarta Sans',sans-serif; font-size:.98rem; font-weight:700;
  margin:0 0 .35rem; color:var(--ink); letter-spacing:-.01em;
}
.tw-insight p{ margin:0; font-size:.92rem; line-height:1.68; color:var(--slate); }

/* ========================================================================
   HERO
   ===================================================================== */
.tw-hero{ position:relative; padding:clamp(28px,5vw,54px) 0 0; }
.tw-hero__inner{ max-width:820px; }
.tw-hero h1{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  font-size:clamp(2.6rem,6.6vw,5rem); letter-spacing:-.042em; line-height:1.02;
  margin:1.2rem 0 1.4rem; color:var(--ink);
}
.tw-hero p.tw-lede{ font-size:clamp(1rem,1.6vw,1.18rem); max-width:620px; }
.tw-hero__cta{ display:flex; gap:.8rem; flex-wrap:wrap; margin-top:2.2rem; }
.tw-hero__stats{
  display:flex; gap:clamp(1.6rem,4vw,3.4rem); flex-wrap:wrap; margin-top:3.4rem;
  padding-top:2rem; border-top:1px solid var(--line);
}
.tw-stat b{
  display:block; font-family:'JetBrains Mono',monospace; font-weight:500;
  font-size:clamp(1.5rem,3vw,2rem); letter-spacing:-.03em; color:var(--ink);
}
.tw-stat span{ font-size:.78rem; color:var(--slate); letter-spacing:.03em; }

/* the plane drifting across the hero */
.tw-hero__plane{
  position:absolute; right:-2%; top:6%; width:min(46%,470px); pointer-events:none;
  opacity:.95;
}
.tw-hero__plane svg{ width:100%; height:auto; overflow:visible; }
@media (max-width:900px){ .tw-hero__plane{ display:none; } }

/* ========================================================================
   FORM — Streamlit widgets restyled into bespoke controls
   ===================================================================== */
.tw-panel{
  background:var(--glass); backdrop-filter:blur(18px) saturate(1.4);
  -webkit-backdrop-filter:blur(18px) saturate(1.4);
  border:1px solid rgba(255,255,255,.86); border-radius:var(--r-xl);
  box-shadow:var(--sh-2); padding:clamp(1.4rem,3vw,2.4rem);
}
.tw-fieldset{ margin:0 0 .4rem; }
.tw-fieldset__title{
  font-family:'JetBrains Mono',monospace; font-size:.66rem; letter-spacing:.2em;
  text-transform:uppercase; color:var(--slate-2); margin:1.8rem 0 1rem;
  display:flex; align-items:center; gap:.7rem;
}
.tw-fieldset__title::after{ content:""; flex:1; height:1px; background:var(--line); }

[data-testid="stSlider"] label, .stSelectbox label, .stRadio label,
.stCheckbox label, .stNumberInput label, .stMultiSelect label{
  font-family:'Inter',sans-serif !important; font-size:.86rem !important;
  font-weight:600 !important; color:var(--ink-2) !important;
  letter-spacing:-.005em !important; margin-bottom:.15rem !important;
}
[data-testid="stSlider"]{ direction:ltr !important; padding:.1rem 0 .4rem; }
[data-testid="stSlider"] *{ direction:ltr !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background:#fff !important; border:2px solid var(--blue) !important;
  box-shadow:0 2px 10px rgba(37,99,235,.34) !important;
}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"]{
  font-family:'JetBrains Mono',monospace !important; font-size:.66rem !important;
  color:var(--slate-2) !important;
}
[data-testid="stThumbValue"]{
  font-family:'JetBrains Mono',monospace !important; font-size:.72rem !important;
  color:#fff !important; background:var(--blue) !important;
  padding:.1rem .45rem !important; border-radius:7px !important;
}
div[data-baseweb="select"] > div{
  background:rgba(255,255,255,.9) !important; border-radius:var(--r-sm) !important;
  border:1px solid var(--line-2) !important; min-height:2.9rem !important;
  font-size:.94rem !important; box-shadow:var(--sh-1) !important;
  transition:border-color .22s var(--ease), box-shadow .22s var(--ease) !important;
}
div[data-baseweb="select"] > div:hover{ border-color:var(--sky) !important; }
div[data-baseweb="select"] > div:focus-within{
  border-color:var(--blue) !important; box-shadow:0 0 0 3px rgba(37,99,235,.14) !important;
}
.stNumberInput input, .stTextInput input{
  background:rgba(255,255,255,.9) !important; border-radius:var(--r-sm) !important;
  border:1px solid var(--line-2) !important; min-height:2.9rem !important;
}
div[role="radiogroup"]{ gap:.5rem !important; }
div[role="radiogroup"] label{
  background:rgba(255,255,255,.85); border:1px solid var(--line-2);
  border-radius:99px; padding:.42rem .95rem !important; margin:0 !important;
  transition:all .22s var(--ease);
}
div[role="radiogroup"] label:hover{ border-color:var(--sky); }

/* ========================================================================
   CHARTS
   ===================================================================== */
.tw-chart{ padding:1.4rem 1.5rem 1rem; }
.tw-chart h4{
  font-family:'Plus Jakarta Sans',sans-serif; font-size:1rem; font-weight:700;
  margin:0 0 .2rem; color:var(--ink); letter-spacing:-.015em;
}
.tw-chart p{ margin:0 0 .6rem; font-size:.84rem; color:var(--slate); }
.js-plotly-plot .plotly text{ font-family:'Inter',sans-serif !important; }

/* ========================================================================
   FOOTER / CONTACT
   ===================================================================== */
.tw-foot{
  margin-top:clamp(64px,9vw,110px); padding:2.6rem 0 .5rem;
  border-top:1px solid var(--line); display:flex; flex-wrap:wrap; gap:1.6rem;
  justify-content:space-between; align-items:flex-start;
}
.tw-foot__note{ color:var(--slate-2); font-size:.82rem; line-height:1.7; max-width:420px; }
.tw-foot a{ color:var(--blue); text-decoration:none; }
.tw-foot a:hover{ text-decoration:underline; }

/* ========================================================================
   ENTRANCE ANIMATION
   ===================================================================== */
.tw-rise{ opacity:0; transform:translateY(20px); animation:twRise .8s var(--ease) forwards; }
@keyframes twRise{ to{ opacity:1; transform:none; } }
.d1{animation-delay:.05s}.d2{animation-delay:.12s}.d3{animation-delay:.19s}
.d4{animation-delay:.26s}.d5{animation-delay:.33s}.d6{animation-delay:.4s}
.tw-float{ animation:twFloat 7s ease-in-out infinite; }
@keyframes twFloat{ 0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)} }

/* ========================================================================
   RESPONSIVE
   ===================================================================== */
@media (max-width:820px){
  :root{ --nav-h:62px; }
  .tw-nav__links a{ padding:.45rem .58rem; font-size:.82rem; }
  .tw-nav__links .tw-btn{ display:none; }
  .tw-facts{ grid-template-columns:repeat(2,1fr); }
}
@media (max-width:560px){
  .tw-nav__brand b{ display:none; }
  .tw-nav__links a{ padding:.4rem .44rem; font-size:.78rem; }
  .tw-hero__cta .tw-btn{ width:100%; }
  .tw-dest__place{ right:88px; }
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{
    animation-duration:.01ms !important; animation-iteration-count:1 !important;
    transition-duration:.01ms !important; scroll-behavior:auto !important;
  }
  .tw-card:hover, .tw-dest:hover .tw-dest__art svg{ transform:none; }
}
</style>
""".replace("__FONTS__", FONTS)

# ==========================================================================
# DESTINATION ARTWORK — Generative destination artwork.
# ==========================================================================

import hashlib
import math

# --------------------------------------------------------------------------- #
# palettes: sky stops, land tones, sun colour
# --------------------------------------------------------------------------- #

PALETTES = {
    "tropic": dict(
        sky=["#F9D18B", "#F5A25D", "#E9738D", "#8E63A8"],
        land=["#1F3A5F", "#16293F"], sun="#FFF1C9", water="#2E7FA8", accent="#FFD9A0",
    ),
    "warm": dict(
        sky=["#BFE5F5", "#8FD0EC", "#6FB6DE", "#4A93C4"],
        land=["#2C5C7A", "#1D3F55"], sun="#FFF6DC", water="#3E8FB8", accent="#FFE3B0",
    ),
    "temperate": dict(
        sky=["#DCEEF9", "#AFD8F0", "#7FBEE2", "#5A9ECB"],
        land=["#33607A", "#22455C"], sun="#FFFBEF", water="#4C97BE", accent="#CFE9F6",
    ),
    "cool": dict(
        sky=["#EAF2F8", "#CBE0EF", "#A6C7E0", "#7FA8C9"],
        land=["#46647C", "#2F4759"], sun="#FFFFFF", water="#6FA0C0", accent="#E4F0F8",
    ),
    "cold": dict(
        sky=["#F2F6FA", "#DCE8F2", "#BED2E4", "#9BB6CE"],
        land=["#5A7288", "#405467"], sun="#FFFFFF", water="#89AAC6", accent="#F0F6FB",
    ),
}


def _palette(temp_c: float) -> dict:
    if temp_c >= 27:
        return PALETTES["tropic"]
    if temp_c >= 21:
        return PALETTES["warm"]
    if temp_c >= 13:
        return PALETTES["temperate"]
    if temp_c >= 6:
        return PALETTES["cool"]
    return PALETTES["cold"]


def _rng(seed_text: str):
    """Small deterministic generator so a city always draws identically."""
    h = hashlib.sha256(seed_text.encode("utf-8")).digest()
    state = int.from_bytes(h[:8], "big")

    def nxt(lo: float = 0.0, hi: float = 1.0) -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (2 ** 64)
        return lo + (state >> 11) / float(1 << 53) * (hi - lo)

    return nxt


def _scene_for(tastes: dict) -> str:
    """Pick the landscape that matches what a place is actually known for."""
    ranked = sorted(tastes.items(), key=lambda kv: -kv[1])
    for key, score in ranked:
        if score < 4:
            break
        if key == "beaches":
            return "coast"
        if key in ("nature", "adventure"):
            return "peaks"
        if key == "urban":
            return "skyline"
        if key == "culture":
            return "heritage"
        if key == "seclusion":
            return "dunes"
    return "coast" if tastes.get("beaches", 0) >= 3 else "peaks"


# --------------------------------------------------------------------------- #
# scene builders — each returns SVG that fills a 400 x 240 viewBox
# --------------------------------------------------------------------------- #

ART_W, ART_H = 400, 240


def _sun(p: dict, rnd, low: bool) -> str:
    cx = rnd(90, 310)
    cy = rnd(ART_H * 0.42, ART_H * 0.6) if low else rnd(38, 70)
    r = 24 if low else 17
    return (
        f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r * 3.4:.0f}" fill="url(#glow)"/>'
        f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r}" fill="{p["sun"]}" opacity=".95"/>'
    )


def _coast(p: dict, rnd) -> str:
    horizon = ART_H * 0.60
    waves = ""
    for i in range(7):
        y = horizon + 8 + i * 11
        off = rnd(-30, 30)
        waves += (
            f'<path d="M{-20 + off:.0f} {y:.0f} q 40 -5 80 0 t 80 0 t 80 0 t 80 0 t 80 0" '
            f'fill="none" stroke="{p["accent"]}" stroke-width="{2.2 - i * 0.2:.1f}" '
            f'opacity="{0.5 - i * 0.05:.2f}" stroke-linecap="round"/>'
        )
    palms = ""
    for i in range(int(rnd(2, 4))):
        x = rnd(24, 90) if i % 2 == 0 else rnd(310, 376)
        base = horizon + rnd(26, 54)
        hgt = rnd(38, 60)
        palms += f'<path d="M{x:.0f} {base:.0f} q {rnd(-6, 6):.0f} -{hgt * .6:.0f} {rnd(-3, 5):.0f} -{hgt:.0f}" stroke="{p["land"][1]}" stroke-width="3" fill="none" stroke-linecap="round"/>'
        for a in (-52, -22, 18, 48):
            palms += (
                f'<path d="M{x:.0f} {base - hgt:.0f} q {math.cos(math.radians(a)) * 20:.0f} '
                f'{math.sin(math.radians(a)) * 12 - 6:.0f} {math.cos(math.radians(a)) * 34:.0f} '
                f'{math.sin(math.radians(a)) * 20 + 2:.0f}" stroke="{p["land"][1]}" '
                f'stroke-width="2.6" fill="none" stroke-linecap="round"/>'
            )
    return (
        f'<rect x="0" y="{horizon:.0f}" width="{ART_W}" height="{ART_H - horizon:.0f}" fill="url(#sea)"/>'
        f"{waves}{palms}"
    )


def _peaks(p: dict, rnd) -> str:
    out = ""
    for layer, (tone, base, height, op) in enumerate(
        ((p["land"][0], ART_H * 0.78, 96, 0.55), (p["land"][0], ART_H * 0.86, 130, 0.8), (p["land"][1], ART_H * 0.95, 82, 1.0))
    ):
        pts, x = [], -20
        while x < ART_W + 20:
            step = rnd(48, 92)
            peak = base - rnd(height * 0.55, height)
            pts.append((x + step / 2, peak))
            x += step
        d = f"M-20 {ART_H} L-20 {base:.0f} "
        for px, py in pts:
            d += f"L{px:.0f} {py:.0f} "
        d += f"L{ART_W + 20} {base:.0f} L{ART_W + 20} {ART_H} Z"
        out += f'<path d="{d}" fill="{tone}" opacity="{op}"/>'
        if layer == 2:
            for px, py in pts[:4]:
                out += (
                    f'<path d="M{px - 13:.0f} {py + 18:.0f} L{px:.0f} {py:.0f} '
                    f'L{px + 13:.0f} {py + 18:.0f} Z" fill="#FFFFFF" opacity=".85"/>'
                )
    return out


def _skyline(p: dict, rnd) -> str:
    ground = ART_H * 0.88
    out = ""
    for layer, (tone, op, scale) in enumerate(((p["land"][0], 0.5, 0.7), (p["land"][1], 1.0, 1.0))):
        x = -10
        while x < ART_W + 10:
            bw = rnd(20, 40)
            bh = rnd(46, 128) * scale
            top = ground - bh
            out += f'<rect x="{x:.0f}" y="{top:.0f}" width="{bw:.0f}" height="{bh:.0f}" fill="{tone}" opacity="{op}"/>'
            if layer == 1 and rnd() > 0.55:  # a spire on some towers
                out += f'<rect x="{x + bw / 2 - 1.5:.0f}" y="{top - 18:.0f}" width="3" height="18" fill="{tone}"/>'
            if layer == 1:
                for wy in range(int(top) + 10, int(ground) - 6, 14):
                    for wx in range(int(x) + 5, int(x + bw) - 5, 10):
                        if rnd() > 0.42:
                            out += f'<rect x="{wx}" y="{wy}" width="3.4" height="5" fill="{p["accent"]}" opacity=".75"/>'
            x += bw + rnd(3, 9)
    out += f'<rect x="0" y="{ground:.0f}" width="{ART_W}" height="{ART_H - ground:.0f}" fill="{p["land"][1]}"/>'
    return out


def _heritage(p: dict, rnd) -> str:
    ground = ART_H * 0.88
    out = f'<rect x="0" y="{ground:.0f}" width="{ART_W}" height="{ART_H - ground:.0f}" fill="{p["land"][1]}"/>'
    x = 10
    while x < ART_W - 30:
        bw = rnd(46, 82)
        bh = rnd(48, 84)
        top = ground - bh
        out += f'<rect x="{x:.0f}" y="{top:.0f}" width="{bw:.0f}" height="{bh:.0f}" fill="{p["land"][1]}"/>'
        out += (
            f'<path d="M{x:.0f} {top:.0f} q {bw / 2:.0f} -{bw * 0.62:.0f} {bw:.0f} 0 Z" '
            f'fill="{p["land"][0]}"/>'
        )  # dome
        out += (
            f'<path d="M{x + bw / 2:.0f} {top - bw * 0.46:.0f} l0 -{rnd(10, 20):.0f}" '
            f'stroke="{p["land"][0]}" stroke-width="2.6" stroke-linecap="round"/>'
        )  # finial
        for ax in range(int(x) + 8, int(x + bw) - 12, 18):
            out += (
                f'<path d="M{ax} {ground:.0f} l0 -20 q 6 -12 12 0 l0 20 Z" '
                f'fill="{p["land"][0]}" opacity=".55"/>'
            )  # arches
        x += bw + rnd(8, 20)
    return out


def _dunes(p: dict, rnd) -> str:
    out = ""
    for i, (tone, base, op) in enumerate(
        ((p["land"][0], ART_H * 0.72, 0.45), (p["land"][0], ART_H * 0.82, 0.75), (p["land"][1], ART_H * 0.92, 1.0))
    ):
        d = f"M-20 {ART_H} L-20 {base:.0f} "
        x = -20
        while x < ART_W + 20:
            step = rnd(90, 150)
            d += f"q {step / 2:.0f} {rnd(-34, -12):.0f} {step:.0f} {rnd(-6, 10):.0f} "
            x += step
        d += f"L{ART_W + 20} {ART_H} Z"
        out += f'<path d="{d}" fill="{tone}" opacity="{op}"/>'
    return out


SCENES = {
    "coast": _coast,
    "peaks": _peaks,
    "skyline": _skyline,
    "heritage": _heritage,
    "dunes": _dunes,
}


def destination_art(city: str, tastes: dict, temp_c: float) -> str:
    """Return a standalone SVG landscape for one destination."""
    rnd = _rng(city)
    pal = _palette(temp_c)
    scene = _scene_for(tastes)
    low_sun = scene in ("coast", "dunes") or temp_c >= 27

    stops = "".join(
        f'<stop offset="{i / (len(pal["sky"]) - 1) * 100:.0f}%" stop-color="{c}"/>'
        for i, c in enumerate(pal["sky"])
    )
    clouds = ""
    for _ in range(int(rnd(2, 5))):
        cx, cy, s = rnd(30, 370), rnd(30, 110), rnd(0.5, 1.1)
        clouds += (
            f'<g opacity="{rnd(.16, .34):.2f}" transform="translate({cx:.0f},{cy:.0f}) scale({s:.2f})">'
            f'<ellipse cx="0" cy="0" rx="34" ry="11"/><ellipse cx="-18" cy="4" rx="20" ry="8"/>'
            f'<ellipse cx="16" cy="4" rx="24" ry="9"/><ellipse cx="-2" cy="-7" rx="17" ry="10"/></g>'
        )

    return (
        f'<svg viewBox="0 0 {ART_W} {ART_H}" xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="xMidYMid slice" class="tw-art">'
        f"<defs>"
        f'<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">{stops}</linearGradient>'
        f'<linearGradient id="sea" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{pal["water"]}"/>'
        f'<stop offset="100%" stop-color="{pal["land"][1]}"/></linearGradient>'
        f'<radialGradient id="glow">'
        f'<stop offset="0%" stop-color="{pal["sun"]}" stop-opacity=".55"/>'
        f'<stop offset="100%" stop-color="{pal["sun"]}" stop-opacity="0"/></radialGradient>'
        f"</defs>"
        f'<rect width="{ART_W}" height="{ART_H}" fill="url(#sky)"/>'
        f'<g fill="#FFFFFF">{clouds}</g>'
        f"{_sun(pal, rnd, low_sun)}"
        f"{SCENES[scene](pal, rnd)}"
        f"</svg>"
    )

# ==========================================================================
# RECOMMENDATION ENGINE — Recommendation engine and the travel intelligence layered on top of it.
# ==========================================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

TASTES = [
    ("culture", "Culture", "Museums, old towns, landmarks"),
    ("adventure", "Adventure", "Hiking, diving, adrenaline"),
    ("nature", "Nature", "Parks, mountains, wildlife"),
    ("beaches", "Beaches", "Coastline and swimming"),
    ("nightlife", "Nightlife", "Bars, music, late nights"),
    ("cuisine", "Food", "Restaurants and street food"),
    ("wellness", "Wellness", "Spas, retreats, slow days"),
    ("urban", "City life", "Design, shopping, skylines"),
    ("seclusion", "Quiet", "Few crowds, room to breathe"),
]
TASTE_KEYS = [k for k, _, _ in TASTES]
TASTE_LABEL = {k: lbl for k, lbl, _ in TASTES}

REGIONS = [
    ("region_africa", "Africa"),
    ("region_asia", "Asia"),
    ("region_europe", "Europe"),
    ("region_middle_east", "Middle East"),
    ("region_north_america", "North America"),
    ("region_oceania", "Oceania"),
    ("region_south_america", "South America"),
]
REGION_COLS = [c for c, _ in REGIONS]
REGION_LABEL = dict(REGIONS)

FEATURE_COLS = (
    ["latitude", "longitude"]
    + TASTE_KEYS
    + ["has_airport", "is_short_trip", "is_one_week", "temp_avg_yearly",
       "budget_level_encoded", "HotelRating_encoded", "rating_was_unknown"]
    + REGION_COLS
)

BUDGET_LABEL = {1: "Budget", 2: "Mid-range", 3: "Luxury"}
BUDGET_DAILY = {1: 55, 2: 130, 3: 290}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def clean(value) -> str | None:
    """Trim a cell, treating the notebook's placeholder fills as missing."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "not specified", "nan", "none", "n/a", "-"}:
        return None
    return text


def airport_of(row) -> tuple[str | None, str | None]:
    """Nearest airport as (name, code).

    The notebook fills unmatched cities with "Unknown" *after* computing
    has_airport, so a row can hold a perfectly good airport name while the flag
    reads 0. The flag is therefore ignored and the actual values decide.
    """
    name = clean(row.get("name"))
    code = clean(row.get("iata")) or clean(row.get("icao"))
    return (name, code) if (name or code) else (None, None)


def region_of(row) -> str | None:
    for col, label in REGIONS:
        try:
            if float(row.get(col, 0)) == 1:
                return label
        except (TypeError, ValueError):
            continue
    return None


def tastes_of(row) -> dict:
    out = {}
    for key in TASTE_KEYS:
        try:
            out[key] = float(row.get(key, 3))
        except (TypeError, ValueError):
            out[key] = 3.0
    return out


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #

class Recommender:
    """Holds the fitted scaler so it is computed once, not per request."""

    def __init__(self, df: pd.DataFrame):
        self.columns = [c for c in FEATURE_COLS if c in df.columns]
        self.scaler = StandardScaler()
        self.matrix = self.scaler.fit_transform(df[self.columns])
        self.df = df

    def rank(self, prefs: dict, top_n: int = 6) -> pd.DataFrame:
        user = pd.DataFrame([{c: prefs.get(c, 0) for c in self.columns}])[self.columns]
        scores = cosine_similarity(self.scaler.transform(user), self.matrix)[0]

        out = self.df.copy()
        out["similarity"] = scores
        out["match"] = ((scores + 1) / 2 * 100).round(1)
        out = out.sort_values("similarity", ascending=False)
        subset = [c for c in ("city", "country") if c in out.columns]
        if subset:
            out = out.drop_duplicates(subset=subset)
        return out.head(top_n).reset_index(drop=True)


def build_preferences(df: pd.DataFrame, answers: dict) -> dict:
    """Translate the form's answers into a point in feature space."""
    prefs = {c: 0 for c in FEATURE_COLS}
    prefs.update({
        "latitude": float(df["latitude"].mean()) if "latitude" in df else 0.0,
        "longitude": float(df["longitude"].mean()) if "longitude" in df else 0.0,
        "has_airport": 1 if answers.get("needs_airport", True) else 0,
        "is_short_trip": 1 if answers.get("trip_length") == "short" else 0,
        "is_one_week": 1 if answers.get("trip_length") == "week" else 0,
        "temp_avg_yearly": answers.get("temp", 22),
        "budget_level_encoded": answers.get("budget", 2),
        "HotelRating_encoded": answers.get("stars", 4),
        "rating_was_unknown": 0,
    })
    for key in TASTE_KEYS:
        prefs[key] = answers.get(key, 3)
    region = answers.get("region")
    if region and region in REGION_COLS:
        prefs[region] = 1
    return prefs


# --------------------------------------------------------------------------- #
# travel intelligence
# --------------------------------------------------------------------------- #

def daily_cost(row) -> int:
    try:
        tier = int(float(row.get("budget_level_encoded", 2)))
    except (TypeError, ValueError):
        tier = 2
    tier = min(3, max(1, tier))
    try:
        stars = float(row.get("HotelRating_encoded", 3))
    except (TypeError, ValueError):
        stars = 3.0
    return int(round(BUDGET_DAILY[tier] * (0.85 + 0.075 * stars), -1))


def trip_cost(row, nights: int, travellers: int) -> int:
    return daily_cost(row) * max(1, nights) * max(1, travellers)


def climate_summary(row) -> str:
    t = float(row.get("temp_avg_yearly", 20) or 20)
    if t >= 28:
        return f"Hot year-round, averaging {t:.0f}°C"
    if t >= 22:
        return f"Warm and settled, around {t:.0f}°C"
    if t >= 15:
        return f"Mild, averaging {t:.0f}°C"
    if t >= 8:
        return f"Cool, around {t:.0f}°C — pack layers"
    return f"Cold, averaging {t:.0f}°C"


def best_season(row) -> str:
    """Infer a sensible window from latitude and average temperature."""
    lat = float(row.get("latitude", 0) or 0)
    t = float(row.get("temp_avg_yearly", 20) or 20)
    north = lat >= 0
    if abs(lat) < 15:                       # tropics: the dry months matter, not heat
        return "December to March, outside the rains"
    if t >= 26:                             # hot climates are best off-peak
        return "November to March" if north else "May to September"
    if t <= 8:                              # cold climates: either peak summer or snow season
        return "June to August, or December for snow"
    return "April to June and September to October" if north else \
           "October to December and March to May"


def trip_style(row) -> list[str]:
    tastes = tastes_of(row)
    ranked = sorted(tastes.items(), key=lambda kv: -kv[1])
    return [TASTE_LABEL[k] for k, v in ranked[:3] if v >= 3.5] or ["Balanced"]


def explain(row, prefs: dict) -> str:
    """Say plainly why this place surfaced, using the traveller's own answers."""
    tastes = tastes_of(row)
    strong = []
    for key in TASTE_KEYS:
        want = float(prefs.get(key, 3))
        has = tastes.get(key, 3)
        if want >= 4 and has >= 4:
            strong.append(TASTE_LABEL[key].lower())

    city = clean(row.get("city")) or "This destination"
    tier = BUDGET_LABEL.get(int(float(row.get("budget_level_encoded", 2) or 2)), "Mid-range")

    if strong:
        wanted = ", ".join(strong[:2]) if len(strong) <= 2 else \
            f"{', '.join(strong[:2])} and {strong[2]}"
        lead = f"You asked for {wanted}, and {city} scores highly on all of it"
    else:
        lead = f"{city} sits closest to the overall balance you described"

    temp_gap = abs(float(row.get("temp_avg_yearly", 20) or 20) - float(prefs.get("temp_avg_yearly", 22)))
    climate = "with a climate close to your target" if temp_gap <= 4 else \
              "though the climate runs a little off your target"
    return f"{lead}, {climate}. It fits a {tier.lower()} budget."


def attractions_of(row, limit: int = 3) -> list[str]:
    """Pull attractions from the dataset if the export carried them."""
    raw = clean(row.get("Attractions"))
    if not raw:
        return []
    parts = [p.strip(" .;") for chunk in raw.split("|") for p in chunk.split(",")]
    seen, out = set(), []
    for p in parts:
        if 3 < len(p) < 60 and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p if len(p) < 46 else p[:43] + "…")
        if len(out) >= limit:
            break
    return out


def travel_tip(row) -> str:
    """A short, situation-specific pointer built from the row's own numbers."""
    tastes = tastes_of(row)
    t = float(row.get("temp_avg_yearly", 20) or 20)
    tier = int(float(row.get("budget_level_encoded", 2) or 2))
    _, code = airport_of(row)

    if t >= 28:
        return "Plan outdoor time early morning or after sunset — midday heat is punishing."
    if t <= 8:
        return "Daylight is short in winter; book the outdoor activities first."
    if tastes.get("seclusion", 3) >= 4:
        return "Getting around is easier with your own transport — arrange it before arrival."
    if tastes.get("nightlife", 3) >= 4 and tier <= 2:
        return "Stay central: the saving on taxis usually beats the cheaper outer hotels."
    if tastes.get("culture", 3) >= 4:
        return "Buy museum passes online — the queues are the real cost, not the ticket."
    if code:
        return f"Flights into {code} are cheapest midweek; avoid Friday and Sunday departures."
    return "Book accommodation before flights — availability moves faster than airfare."


# --------------------------------------------------------------------------- #
# insights across the whole result set
# --------------------------------------------------------------------------- #

def insights(results: pd.DataFrame, prefs: dict, answers: dict) -> list[tuple[str, str]]:
    """Plain-language readings of the result set, as (heading, body) pairs."""
    out: list[tuple[str, str]] = []
    if results.empty:
        return out

    wanted = [TASTE_LABEL[k].lower() for k in TASTE_KEYS if float(prefs.get(k, 3)) >= 4]
    tier = BUDGET_LABEL.get(int(answers.get("budget", 2)), "Mid-range").lower()
    if wanted:
        phrase = wanted[0] if len(wanted) == 1 else ", ".join(wanted[:-1]) + f" and {wanted[-1]}"
        out.append((
            "Your travel profile",
            f"You lean towards {phrase} at a {tier} budget. That combination is what "
            f"ranked these destinations, not their general popularity.",
        ))
    else:
        out.append((
            "Your travel profile",
            f"You kept most preferences balanced at a {tier} budget, so the ranking "
            "favours destinations that do many things well rather than specialising.",
        ))

    temps = results["temp_avg_yearly"].astype(float)
    target = float(answers.get("temp", 22))
    close = int((temps - target).abs().le(4).sum())
    out.append((
        "Climate fit",
        f"{close} of your {len(results)} matches sit within 4°C of the {target:.0f}°C "
        f"you asked for. The spread runs {temps.min():.0f}°C to {temps.max():.0f}°C.",
    ))

    costs = [daily_cost(r) for _, r in results.iterrows()]
    nights = int(answers.get("nights", 7))
    out.append((
        "What it costs",
        f"Daily spend across these matches runs ${min(costs)}–${max(costs)} per person. "
        f"Over {nights} nights that is roughly ${min(costs) * nights:,}–${max(costs) * nights:,} "
        "each, before flights.",
    ))

    regions = [region_of(r) for _, r in results.iterrows()]
    regions = [r for r in regions if r]
    if regions:
        top = max(set(regions), key=regions.count)
        share = regions.count(top)
        out.append((
            "Where they cluster",
            f"{share} of {len(results)} matches are in {top}. Booking two of them into one "
            "trip is usually cheaper than two separate journeys."
            if share > 1 else
            f"Your matches spread across {len(set(regions))} regions, so there is no obvious "
            "way to combine them — pick on preference, not logistics.",
        ))

    with_air = sum(1 for _, r in results.iterrows() if any(airport_of(r)))
    if with_air < len(results):
        out.append((
            "Getting there",
            f"{len(results) - with_air} of these have no major airport recorded in the "
            "dataset, which usually means a connecting flight plus ground transfer.",
        ))
    return out

# ==========================================================================
# CATALOGUE — Loading and validating the destination catalogue.
# ==========================================================================

import os

import numpy as np
import pandas as pd


CSV_NAME = "tripwise_data.csv"

REQUIRED = ["city", "country", "latitude", "longitude", "temp_avg_yearly",
            "budget_level_encoded"] + TASTE_KEYS

# city, country, region col, iata, airport, lat, lon,
# cul adv nat bea nig cui wel urb sec, budget, temp, stars
_DEMO = [
    ("Paris", "France", "region_europe", "CDG", "Charles de Gaulle", 48.86, 2.35, 5,2,2,1,4,5,3,5,1, 3, 12.3, 4.2),
    ("Barcelona", "Spain", "region_europe", "BCN", "Barcelona El Prat", 41.39, 2.17, 4,3,2,4,5,5,3,4,1, 2, 17.1, 4.0),
    ("Reykjavik", "Iceland", "region_europe", "KEF", "Keflavik International", 64.15, -21.94, 3,5,5,1,3,3,4,2,5, 3, 5.2, 4.0),
    ("Santorini", "Greece", "region_europe", "JTR", "Santorini National", 36.39, 25.46, 3,2,3,5,3,4,5,2,4, 3, 18.4, 4.4),
    ("Tokyo", "Japan", "region_asia", "HND", "Haneda", 35.68, 139.69, 5,2,2,1,5,5,3,5,1, 3, 16.0, 4.3),
    ("Bali", "Indonesia", "region_asia", "DPS", "Ngurah Rai International", -8.41, 115.19, 4,4,5,5,4,4,5,2,4, 1, 26.6, 4.1),
    ("Bangkok", "Thailand", "region_asia", "BKK", "Suvarnabhumi", 13.76, 100.50, 4,3,2,2,5,5,4,5,1, 1, 28.4, 4.0),
    ("Maldives", "Maldives", "region_asia", "MLE", "Velana International", 4.18, 73.51, 1,3,4,5,1,3,5,1,5, 3, 28.1, 4.6),
    ("Dubai", "UAE", "region_middle_east", "DXB", "Dubai International", 25.20, 55.27, 2,4,1,4,4,4,4,5,1, 3, 28.0, 4.5),
    ("AlUla", "Saudi Arabia", "region_middle_east", "ULH", "Prince Abdul Majeed", 26.61, 37.92, 5,5,5,1,1,3,4,1,5, 3, 26.0, 4.4),
    ("Istanbul", "Turkey", "region_middle_east", "IST", "Istanbul Airport", 41.01, 28.98, 5,2,2,2,4,5,3,5,1, 2, 15.0, 4.1),
    ("Marrakesh", "Morocco", "region_africa", "RAK", "Marrakesh Menara", 31.63, -7.99, 5,4,3,1,3,5,4,3,2, 2, 20.3, 4.0),
    ("Cape Town", "South Africa", "region_africa", "CPT", "Cape Town International", -33.92, 18.42, 4,5,5,5,4,5,3,4,3, 2, 17.0, 4.2),
    ("Zanzibar", "Tanzania", "region_africa", "ZNZ", "Abeid Amani Karume", -6.16, 39.20, 3,4,4,5,2,3,4,1,4, 2, 26.4, 4.0),
    ("New York", "United States", "region_north_america", "JFK", "John F. Kennedy", 40.71, -74.01, 5,2,2,2,5,5,3,5,1, 3, 12.9, 4.1),
    ("Banff", "Canada", "region_north_america", "YYC", "Calgary International", 51.18, -115.57, 2,5,5,1,2,3,4,1,5, 3, 3.0, 4.3),
    ("Mexico City", "Mexico", "region_north_america", "MEX", "Benito Juarez", 19.43, -99.13, 5,3,3,1,5,5,3,5,1, 2, 17.0, 4.0),
    ("Rio de Janeiro", "Brazil", "region_south_america", "GIG", "Galeao", -22.91, -43.17, 3,4,4,5,5,4,3,4,2, 2, 23.8, 4.0),
    ("Cusco", "Peru", "region_south_america", "CUZ", "Alejandro Velasco Astete", -13.53, -71.97, 5,5,5,1,3,4,3,2,4, 1, 12.3, 3.9),
    ("Patagonia", "Chile", "region_south_america", "PNT", "Teniente Julio Gallardo", -51.73, -72.51, 2,5,5,1,1,3,3,1,5, 3, 6.5, 4.0),
    ("Sydney", "Australia", "region_oceania", "SYD", "Kingsford Smith", -33.87, 151.21, 3,4,4,5,4,5,4,5,2, 3, 18.3, 4.2),
    ("Queenstown", "New Zealand", "region_oceania", "ZQN", "Queenstown Airport", -45.03, 168.66, 2,5,5,2,3,4,4,2,4, 3, 10.6, 4.3),
]


def _demo_frame() -> pd.DataFrame:
    rows = []
    for (city, country, region, iata, airport, lat, lon,
         cul, adv, nat, bea, nig, cui, wel, urb, sec, budget, temp, stars) in _DEMO:
        row = {
            "city": city, "country": country, "iata": iata, "name": airport,
            "latitude": lat, "longitude": lon,
            "culture": cul, "adventure": adv, "nature": nat, "beaches": bea,
            "nightlife": nig, "cuisine": cui, "wellness": wel, "urban": urb,
            "seclusion": sec,
            "budget_level_encoded": budget, "temp_avg_yearly": temp,
            "HotelRating_encoded": stars, "rating_was_unknown": 0,
            "has_airport": 1, "is_short_trip": 1, "is_one_week": 1,
            "HotelName": f"The {city} House",
        }
        for col in REGION_COLS:
            row[col] = 1 if col == region else 0
        rows.append(row)
    return pd.DataFrame(rows)


def load_catalogue_file() -> tuple[pd.DataFrame, bool, list[str]]:
    """Return (catalogue, is_real_data, missing_columns)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CSV_NAME)
    if not os.path.exists(path):
        path = CSV_NAME

    if os.path.exists(path):
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if not missing:
            for col in REGION_COLS + ["has_airport", "is_short_trip", "is_one_week",
                                      "HotelRating_encoded", "rating_was_unknown"]:
                if col not in df.columns:
                    df[col] = 0
            optional = [c for c in ("name", "iata", "HotelName", "Attractions")
                        if c not in df.columns]
            return df, True, optional
        return _demo_frame(), False, missing

    return _demo_frame(), False, []


def catalogue_stats(df: pd.DataFrame) -> list[tuple[str, str]]:
    cities = df["city"].nunique() if "city" in df else len(df)
    countries = df["country"].nunique() if "country" in df else 0
    regions = sum(1 for c in REGION_COLS if c in df.columns and df[c].sum() > 0)
    return [
        (f"{cities:,}", "Destinations ranked"),
        (f"{countries:,}", "Countries covered"),
        (f"{regions}", "World regions"),
        ("9", "Travel dimensions"),
    ]

# ==========================================================================
# HTML COMPONENTS — HTML component builders.
# ==========================================================================

from html import escape

# --------------------------------------------------------------------------- #
# icons — single source, stroked to match the design system
# --------------------------------------------------------------------------- #

ICONS = {
    "compass": '<circle cx="12" cy="12" r="9"/><path d="m15.6 8.4-2 5.2-5.2 2 2-5.2z"/>',
    "sparkle": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8"/><circle cx="12" cy="12" r="3"/>',
    "wallet": '<path d="M3 7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M16 11h5v4h-5a2 2 0 0 1 0-4z"/>',
    "sun": '<circle cx="12" cy="12" r="4.2"/><path d="M12 2v2.4M12 19.6V22M2 12h2.4M19.6 12H22M4.9 4.9l1.7 1.7M17.4 17.4l1.7 1.7M19.1 4.9l-1.7 1.7M6.6 17.4l-1.7 1.7"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3.5 9h17M3.5 15h17M12 3a15 15 0 0 1 0 18A15 15 0 0 1 12 3z"/>',
    "shield": '<path d="M12 3l7 3v6c0 4.2-2.9 7.6-7 9-4.1-1.4-7-4.8-7-9V6z"/><path d="m9.2 12 2 2 3.6-4"/>',
    "plane": '<path d="M2 13.4 21 4l-4.6 16.2-4.1-6.1z"/><path d="m12.3 14.1-3.4 6.9-1-5"/>',
    "pin": '<path d="M12 21s-7-5.6-7-11a7 7 0 1 1 14 0c0 5.4-7 11-7 11z"/><circle cx="12" cy="10" r="2.6"/>',
    "calendar": '<rect x="3.5" y="5" width="17" height="16" rx="2.5"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
    "tag": '<path d="M3 12.5V5a2 2 0 0 1 2-2h7.5L21 11.5 12.5 20z"/><circle cx="7.8" cy="7.8" r="1.4"/>',
    "bulb": '<path d="M9.2 17h5.6M10 21h4"/><path d="M12 3a6 6 0 0 1 3.6 10.8c-.5.4-.8 1-.8 1.6H9.2c0-.6-.3-1.2-.8-1.6A6 6 0 0 1 12 3z"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "layers": '<path d="M12 3 3 8l9 5 9-5z"/><path d="m3 13 9 5 9-5M3 18l9 5 9-5"/>',
}


def icon(name: str, cls: str = "") -> str:
    body = ICONS.get(name, ICONS["sparkle"])
    c = f' class="{cls}"' if cls else ""
    return f'<svg viewBox="0 0 24 24"{c}>{body}</svg>'


def _e(v) -> str:
    return escape(str(v), quote=True)


# --------------------------------------------------------------------------- #
# chrome
# --------------------------------------------------------------------------- #

NAV_ITEMS = [
    ("Home", "?view=home", False),
    ("Features", "?view=home#features", False),
    ("AI Planner", "?view=planner", False),
    ("About", "?view=home#about", False),
    ("Contact", "?view=home#contact", False),
]


def navbar(active: str = "home") -> str:
    links = ""
    for label, href, _ in NAV_ITEMS:
        key = "planner" if "planner" in href else "home"
        cls = " is-active" if (key == active and "#" not in href) else ""
        links += f'<a href="{href}" target="_top" class="{cls.strip()}">{label}</a>'
    return f"""
<div class="tw-nav">
  <a class="tw-nav__brand" href="?view=home" target="_top">
    <svg viewBox="0 0 24 24" fill="none" stroke="url(#navg)" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round">
      <defs><linearGradient id="navg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/>
      </linearGradient></defs>
      <path d="M2 13.4 21 4l-4.6 16.2-4.1-6.1z"/><path d="m12.3 14.1-3.4 6.9-1-5"/>
    </svg>
    <b>TripWise <span class="tw-grad">AI</span></b>
  </a>
  <nav class="tw-nav__links">
    {links}
    <a class="tw-btn tw-btn--sm" href="?view=planner" target="_top">Start planning</a>
  </nav>
</div>"""


def section_head(eyebrow: str, title: str, blurb: str = "", anchor: str = "") -> str:
    a = f' id="{anchor}"' if anchor else ""
    p = f"<p>{blurb}</p>" if blurb else ""
    return f"""
<div class="tw-section"{a}>
  <div class="tw-section__head tw-rise">
    <span class="tw-eyebrow">{eyebrow}</span>
    <h2>{title}</h2>{p}
  </div>"""


def close_section() -> str:
    return "</div>"


# --------------------------------------------------------------------------- #
# hero
# --------------------------------------------------------------------------- #

def hero(stats: list[tuple[str, str]]) -> str:
    stat_html = "".join(
        f'<div class="tw-stat"><b>{_e(v)}</b><span>{_e(lbl)}</span></div>'
        for v, lbl in stats
    )
    return f"""
<div class="tw-hero">
  <div class="tw-hero__plane tw-float">
    <svg viewBox="0 0 420 260" fill="none">
      <defs>
        <linearGradient id="pg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/>
        </linearGradient>
        <linearGradient id="trailg" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#0EA5E9" stop-opacity="0"/>
          <stop offset="100%" stop-color="#2563EB" stop-opacity=".55"/>
        </linearGradient>
      </defs>
      <path d="M10 210 C120 200 210 150 290 70" stroke="url(#trailg)" stroke-width="3"
            stroke-dasharray="2 12" stroke-linecap="round"/>
      <g transform="translate(300,62) rotate(-38)">
        <path d="M46 0 L16 -7 L-6 -34 L-18 -34 L-9 -7 L-30 -7 L-40 -17 L-48 -17
                 L-42 -5 L-42 5 L-48 17 L-40 17 L-30 7 L-9 7 L-18 34 L-6 34 L16 7 Z"
              fill="url(#pg)"/>
        <path d="M46 0 L16 -7 L-9 -7 L-9 7 L16 7 Z" fill="#fff" opacity=".28"/>
      </g>
      <circle cx="300" cy="62" r="46" fill="url(#pg)" opacity=".1"/>
    </svg>
  </div>
  <div class="tw-hero__inner">
    <span class="tw-eyebrow tw-rise d1">Intelligent travel planning</span>
    <h1 class="tw-rise d2">Plan Smarter with<br/><span class="tw-grad">TripWise AI</span></h1>
    <p class="tw-lede tw-rise d3">
      Tell us how you like to travel and TripWise ranks real destinations against
      your answers — with the cost, the climate, the right season and the nearest
      airport worked out before you commit to anything.
    </p>
    <div class="tw-hero__cta tw-rise d4">
      <a class="tw-btn" href="?view=planner" target="_top">Start planning &nbsp;&rarr;</a>
      <a class="tw-btn tw-btn--ghost" href="#features" target="_top">See how it works</a>
    </div>
    <div class="tw-hero__stats tw-rise d5">{stat_html}</div>
  </div>
</div>"""


# --------------------------------------------------------------------------- #
# cards
# --------------------------------------------------------------------------- #

def feature_card(icon_name: str, title: str, body: str, delay: int = 1) -> str:
    return f"""
<div class="tw-card tw-feat tw-rise d{delay}">
  <div class="tw-feat__icon">{icon(icon_name)}</div>
  <h3>{_e(title)}</h3>
  <p>{_e(body)}</p>
</div>"""


def insight_card(title: str, body: str, delay: int = 1) -> str:
    return f"""
<div class="tw-card tw-insight tw-rise d{delay}">
  <div class="tw-insight__dot">{icon('bulb')}</div>
  <div><h4>{_e(title)}</h4><p>{_e(body)}</p></div>
</div>"""


def destination_card(d: dict, delay: int = 1) -> str:
    """`d` is the plain dict assembled by app.py, never a raw dataframe row."""
    facts = f"""
    <div class="tw-facts">
      <div><span>Budget</span><b>{_e(d['tier'])}</b></div>
      <div><span>Per day</span><b>${_e(d['daily'])}</b></div>
      <div><span>Climate</span><b>{_e(d['temp'])}&deg;C</b></div>
    </div>"""

    chips = "".join(f'<span class="tw-chip">{_e(s)}</span>' for s in d["style"])
    if d.get("region"):
        chips += f'<span class="tw-chip tw-chip--muted">{_e(d["region"])}</span>'

    rows = []
    if d.get("airport"):
        rows.append((("plane"), "Nearest airport", d["airport"]))
    else:
        rows.append((("plane"), "Nearest airport", "None recorded for this city"))
    rows.append((("calendar"), "Best season", d["season"]))
    rows.append((("sun"), "Weather", d["climate"]))
    if d.get("hotel"):
        rows.append((("tag"), "Suggested stay", d["hotel"]))
    if d.get("attractions"):
        rows.append((("pin"), "Nearby", ", ".join(d["attractions"])))
    rows.append((("bulb"), "Tip", d["tip"]))

    meta = "".join(
        f'<div class="tw-meta__row">{icon(ic)}<div><b>{_e(lbl)}</b> — {_e(val)}</div></div>'
        for ic, lbl, val in rows
    )

    return f"""
<div class="tw-card tw-dest tw-rise d{delay}">
  <div class="tw-dest__art">
    {d['art']}
    <div class="tw-dest__scrim"></div>
    <div class="tw-dest__match">{_e(d['match'])}% match</div>
    <div class="tw-dest__place">
      <h3>{_e(d['city'])}</h3><span>{_e(d['country'])}</span>
    </div>
  </div>
  <div class="tw-dest__body">
    {facts}
    <div class="tw-chips">{chips}</div>
    <div class="tw-why">{_e(d['why'])}</div>
    <div class="tw-meta">{meta}</div>
  </div>
</div>"""


# --------------------------------------------------------------------------- #
# static content blocks
# --------------------------------------------------------------------------- #

def about_block() -> str:
    return """
<div class="tw-grid tw-grid--2">
  <div class="tw-card tw-feat tw-rise d1">
    <div class="tw-feat__icon">%s</div>
    <h3>Recommendations you can trace</h3>
    <p>TripWise compares your answers against every destination in its catalogue
       across nine travel dimensions, climate, budget tier and accommodation
       standard, then ranks by how closely each one sits to what you described.
       Every card explains its own reasoning in plain language.</p>
  </div>
  <div class="tw-card tw-feat tw-rise d2">
    <div class="tw-feat__icon">%s</div>
    <h3>Grounded in real data</h3>
    <p>Destinations, airports, accommodation standards and climate averages come
       from a curated catalogue rather than generated text, so a recommendation
       always points at somewhere that exists — and tells you when a detail is
       missing instead of inventing one.</p>
  </div>
</div>""" % (icon("sparkle"), icon("shield"))


def contact_block() -> str:
    return """
<div class="tw-card tw-feat tw-rise d1" style="padding:2.4rem 2rem;">
  <div class="tw-grid tw-grid--2" style="align-items:center;">
    <div>
      <h3 style="font-size:1.5rem;margin-bottom:.7rem;">Bring TripWise to your travellers</h3>
      <p>Agencies, airlines and tourism boards can run TripWise against their own
         catalogue — your destinations, your inventory, your branding.</p>
    </div>
    <div style="display:flex;gap:.7rem;flex-wrap:wrap;justify-content:flex-end;">
      <a class="tw-btn" href="mailto:hello@tripwise.ai" target="_top">Talk to us</a>
      <a class="tw-btn tw-btn--ghost" href="?view=planner" target="_top">Try the planner</a>
    </div>
  </div>
</div>"""


def footer() -> str:
    return """
<div class="tw-foot">
  <div class="tw-foot__note">
    <b style="color:var(--ink);">TripWise AI</b><br/>
    Destination matching, cost estimates and climate guidance from a curated
    travel catalogue. Figures are planning estimates, not quotes — confirm
    prices and seasons before booking.
  </div>
  <div class="tw-foot__note" style="text-align:right;">
    <a href="?view=home" target="_top">Home</a> &nbsp;·&nbsp;
    <a href="?view=planner" target="_top">AI Planner</a> &nbsp;·&nbsp;
    <a href="mailto:hello@tripwise.ai">Contact</a>
  </div>
</div>"""

# ==========================================================================
# SPLASH — Generate the TripWise boarding splash as one self-contained HTML string.
# ==========================================================================

SPL_W, SPL_H = 1200, 620
WIN_W, WIN_H = 300, 430
WIN_Y = 78
WIN_X = [72, 450, 828]

# glass opening, relative to each window's top-left
GX, GY, GW, GH = 40, 46, 220, 326
GRX, GRY = 62, 50

CLOUD_LAYERS = [
    # id, baseFrequency, octaves, seed, matrix slope, matrix offset, opacity,
    # mask id, drift seconds, drift px
    ("cA", 0.009, 5, 11, -1.9, 1.18, 0.85, "mDeep", 26, 16),
    ("cB", 0.017, 6, 5, -2.5, 1.45, 1.00, "mDeck", 19, 26),
    ("cC", 0.034, 5, 29, -3.0, 1.72, 0.90, "mLow", 13, 38),
    ("cD", 0.013, 4, 41, -2.6, 1.30, 0.28, "mHigh", 34, 12),
]


def cloud_filters() -> str:
    out = []
    for cid, freq, octv, seed, slope, off, *_ in CLOUD_LAYERS:
        out.append(
            f'<filter id="{cid}" x="0" y="0" width="100%" height="100%">'
            f'<feTurbulence type="fractalNoise" baseFrequency="{freq}" '
            f'numOctaves="{octv}" seed="{seed}" stitchTiles="stitch"/>'
            f'<feColorMatrix type="matrix" values="'
            f'0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  {slope} 0 0 0 {off}"/>'
            f"</filter>"
        )
    return "".join(out)


def deck_masks() -> str:
    """Vertical ramps that place each cloud layer relative to the horizon."""
    ramps = {
        "mDeep": [(0.40, 0), (0.66, 1), (1.0, 1)],
        "mDeck": [(0.44, 0), (0.70, 1), (1.0, 1)],
        "mLow": [(0.58, 0), (0.84, 1), (1.0, 1)],
        "mHigh": [(0.0, 1), (0.30, 0), (1.0, 0)],
    }
    out = []
    for name, stops in ramps.items():
        s = "".join(
            f'<stop offset="{p * 100:.0f}%" stop-color="#fff" stop-opacity="{v}"/>'
            for p, v in stops
        )
        out.append(
            f'<linearGradient id="{name}g" x1="0" y1="0" x2="0" y2="1">{s}</linearGradient>'
            f'<mask id="{name}" maskUnits="userSpaceOnUse" x="-40" y="0" '
            f'width="{GW + 80}" height="{GH}">'
            f'<rect x="-40" y="0" width="{GW + 80}" height="{GH}" fill="url(#{name}g)"/>'
            f"</mask>"
        )
    return "".join(out)


def window_svg(i: int, x: int) -> str:
    """One window: moulded surround, recessed well, glass, sky, shade."""
    gid = f"glass{i}"
    seed_shift = i * 60
    delay = 1.0 + i * 0.12

    clouds = []
    for n, (cid, freq, octv, seed, slope, off, op, mask, dur, dist) in enumerate(CLOUD_LAYERS):
        # a per-window filter so each window shows a different piece of sky
        wid = f"{cid}_{i}"
        clouds.append(
            f'<g class="tw-drift" style="animation-duration:{dur + i * 2}s;'
            f"animation-delay:-{n * 3 + i}s;--dx:{dist}px\">"
            f'<rect x="{GX - 40}" y="{GY}" width="{GW + 80}" height="{GH}" '
            f'filter="url(#{wid})" mask="url(#{mask})" opacity="{op}" '
            f'transform="translate(0,0)"/>'
            f"</g>"
        )

    return f"""
  <g transform="translate({x},{WIN_Y})">
    <rect x="0" y="0" width="{WIN_W}" height="{WIN_H}" rx="96" ry="78" fill="url(#bezel)"
          filter="url(#winShadow)"/>
    <rect x="0" y="0" width="{WIN_W}" height="{WIN_H}" rx="96" ry="78" fill="none"
          stroke="#aab3bd" stroke-width="1.1" opacity=".5"/>
    <rect x="24" y="24" width="{WIN_W - 48}" height="{WIN_H - 48}" rx="76" ry="60" fill="url(#well)"/>
    <rect x="32" y="34" width="{WIN_W - 64}" height="{WIN_H - 68}" rx="68" ry="54"
          fill="#5c6672" opacity=".5"/>

    <clipPath id="{gid}">
      <rect x="{GX}" y="{GY}" width="{GW}" height="{GH}" rx="{GRX}" ry="{GRY}"/>
    </clipPath>

    <g clip-path="url(#{gid})">
      <rect x="{GX}" y="{GY}" width="{GW}" height="{GH}" fill="url(#sky)"/>
      <g transform="translate({GX},{GY})">{''.join(clouds)}</g>
      <ellipse cx="{GX + GW * 0.74}" cy="{GY + GH * 0.34}" rx="{GW * 0.7}" ry="{GH * 0.42}"
               fill="url(#sun)" opacity=".55"/>

      <!-- shade: starts closed, glides up out of the opening -->
      <g class="tw-shade" style="animation-delay:{delay}s">
        <rect x="{GX - 4}" y="{GY - 2}" width="{GW + 8}" height="{GH + 4}" fill="url(#shade)"/>
        <rect x="{GX - 4}" y="{GY + GH - 40}" width="{GW + 8}" height="42" fill="url(#lipG)"/>
        <rect x="{GX + GW / 2 - 30}" y="{GY + GH - 26}" width="60" height="9" rx="4.5"
              fill="url(#gripG)"/>
      </g>

      <!-- glass depth + reflection, always above the shade -->
      <rect x="{GX}" y="{GY}" width="{GW}" height="26" fill="#33404f" opacity=".34"/>
      <rect x="{GX}" y="{GY}" width="13" height="{GH}" fill="#33404f" opacity=".2"/>
      <rect x="{GX + GW - 13}" y="{GY}" width="13" height="{GH}" fill="#33404f" opacity=".2"/>
      <path d="M{GX} {GY + 250} L{GX + 150} {GY} L{GX + 210} {GY} L{GX} {GY + GH} Z"
            fill="#fff" opacity=".08"/>
    </g>

    <path d="M{GX + 18} {GY + 6} C{GX + 60} {GY - 26} {GX + GW - 60} {GY - 26} {GX + GW - 18} {GY + 6}
             C{GX + GW - 60} {GY - 12} {GX + 60} {GY - 12} {GX + 18} {GY + 6} Z" fill="url(#lipHi)"/>
  </g>"""


def build_splash() -> str:
    windows = "".join(window_svg(i, x) for i, x in enumerate(WIN_X))

    # per-window copies of each cloud filter, re-seeded so no two windows match
    per_window = []
    for i in range(3):
        for cid, freq, octv, seed, slope, off, *_ in CLOUD_LAYERS:
            per_window.append(
                f'<filter id="{cid}_{i}" x="0" y="0" width="100%" height="100%">'
                f'<feTurbulence type="fractalNoise" baseFrequency="{freq}" '
                f'numOctaves="{octv}" seed="{seed + i * 60}" stitchTiles="stitch"/>'
                f'<feColorMatrix type="matrix" values="'
                f'0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  {slope} 0 0 0 {off}"/>'
                f"</filter>"
            )

    return _flatten(f"""
<style>
.tw-splash{{
  position:fixed; inset:0; z-index:99999; display:grid; place-items:center;
  background:linear-gradient(180deg,#f1f4f7 0%,#e3e9ef 42%,#ced6df 100%);
  animation:twSplashOut .9s ease 4.4s forwards;
}}
@keyframes twSplashOut{{ to{{ opacity:0; visibility:hidden; pointer-events:none; }} }}
.tw-splash .tw-scene{{
  width:min(96vw,1240px); aspect-ratio:{SPL_W}/{SPL_H};
  animation:twSceneIn 1.2s cubic-bezier(.2,.7,.3,1) both;
}}
@keyframes twSceneIn{{ from{{ opacity:0; transform:scale(1.05); }} to{{ opacity:1; transform:none; }} }}
.tw-splash .tw-scene svg{{
  width:100%; height:100%; display:block;
  filter:drop-shadow(0 30px 64px rgba(66,80,96,.22));
}}
.tw-splash .tw-shade{{
  transform:translateY(0);
  animation:twShadeUp 2.2s cubic-bezier(.72,0,.18,1) forwards;
}}
@keyframes twShadeUp{{ to{{ transform:translateY(-{GH + 60}px); }} }}
.tw-splash .tw-drift{{
  animation-name:twDrift; animation-timing-function:ease-in-out;
  animation-iteration-count:infinite; animation-direction:alternate;
}}
@keyframes twDrift{{
  from{{ transform:translateX(calc(var(--dx) * -1)); }}
  to{{ transform:translateX(var(--dx)); }}
}}
.tw-splash .tw-mark{{
  position:absolute; left:0; right:0; bottom:7%; text-align:center; opacity:0;
  animation:twMarkIn 1s cubic-bezier(.2,.7,.3,1) 2.8s both;
}}
.tw-splash .tw-mark .t{{
  font-family:'Plus Jakarta Sans','Outfit',sans-serif; font-weight:800;
  font-size:clamp(1.5rem,3.6vw,2.45rem); letter-spacing:-.02em;
  background:linear-gradient(96deg,#0EA5E9,#2563EB);
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; color:#2563EB;
  filter:drop-shadow(0 0 22px rgba(14,165,233,.45));
}}
.tw-splash .tw-mark .s{{
  font-family:'Inter',sans-serif; margin-top:.5rem; color:#5A6B7E; font-weight:500;
  font-size:clamp(.62rem,1.35vw,.8rem); letter-spacing:.34em; text-transform:uppercase;
}}
@keyframes twMarkIn{{ from{{ opacity:0; transform:translateY(12px); }} to{{ opacity:1; transform:none; }} }}
@media (prefers-reduced-motion:reduce){{
  .tw-splash .tw-shade{{ animation-duration:.01s; }}
  .tw-splash .tw-drift{{ animation:none; }}
  .tw-splash{{ animation:twSplashOut .5s linear 1.8s forwards; }}
}}
</style>
<div class="tw-splash" aria-hidden="true">
<div class="tw-scene">
<svg viewBox="0 0 {SPL_W} {SPL_H}" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="bezel" x1=".25" y1="0" x2=".75" y2="1">
<stop offset="0" stop-color="#ffffff"/><stop offset=".5" stop-color="#f4f6f8"/>
<stop offset="1" stop-color="#dbe1e7"/>
</linearGradient>
<linearGradient id="cabin" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#f1f4f7"/><stop offset=".42" stop-color="#e3e9ef"/>
<stop offset="1" stop-color="#ced6df"/>
</linearGradient>
<radialGradient id="vign" cx="50%" cy="42%" r="72%">
<stop offset="55%" stop-color="#000" stop-opacity="0"/>
<stop offset="100%" stop-color="#48535f" stop-opacity=".16"/>
</radialGradient>
<filter id="winShadow" x="-25%" y="-25%" width="150%" height="150%">
<feDropShadow dx="0" dy="12" stdDeviation="15" flood-color="#54626f" flood-opacity=".33"/>
</filter>
<linearGradient id="well" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#6f7883"/><stop offset=".28" stop-color="#98a1ab"/>
<stop offset=".75" stop-color="#c9d0d8"/><stop offset="1" stop-color="#eef1f4"/>
</linearGradient>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#1a4a8c"/><stop offset=".34" stop-color="#4886c4"/>
<stop offset=".56" stop-color="#9ec8e8"/><stop offset=".72" stop-color="#d9eaf6"/>
<stop offset="1" stop-color="#c2d9ec"/>
</linearGradient>
<radialGradient id="sun">
<stop offset="0" stop-color="#fff6dc" stop-opacity=".85"/>
<stop offset="55%" stop-color="#ffe9b8" stop-opacity=".22"/>
<stop offset="100%" stop-color="#ffe9b8" stop-opacity="0"/>
</radialGradient>
<linearGradient id="shade" x1="0" y1="0" x2="1" y2="0">
<stop offset="0" stop-color="#d9dee4"/><stop offset=".08" stop-color="#f2f4f7"/>
<stop offset=".5" stop-color="#fafbfc"/><stop offset=".92" stop-color="#eceff3"/>
<stop offset="1" stop-color="#d3d9e0"/>
</linearGradient>
<linearGradient id="lipG" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#d3d9e0"/><stop offset="1" stop-color="#b3bbc5"/>
</linearGradient>
<linearGradient id="gripG" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#98a0aa"/><stop offset="1" stop-color="#727a85"/>
</linearGradient>
<linearGradient id="lipHi" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#ffffff" stop-opacity=".92"/>
<stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
</linearGradient>
{deck_masks()}
{''.join(per_window)}
</defs>
<rect width="{SPL_W}" height="{SPL_H}" fill="url(#cabin)"/>
<rect x="0" y="{SPL_H*0.055:.0f}" width="{SPL_W}" height="2" fill="#c3cbd4" opacity=".55"/>
<rect x="0" y="{SPL_H*0.055+3:.0f}" width="{SPL_W}" height="1" fill="#ffffff" opacity=".9"/>
<rect x="0" y="{SPL_H*0.905:.0f}" width="{SPL_W}" height="2" fill="#c3cbd4" opacity=".5"/>
<rect x="0" y="{SPL_H*0.905+3:.0f}" width="{SPL_W}" height="1" fill="#ffffff" opacity=".85"/>
{windows}
<rect width="{SPL_W}" height="{SPL_H}" fill="url(#vign)" pointer-events="none"/>
</svg>
</div>
<div class="tw-mark">
<div class="t">TripWise AI</div>
<div class="s">Plan smarter &middot; Travel further</div>
</div>
</div>
""")


def _flatten(markup: str) -> str:
    """Strip leading indentation from every line.

    Streamlit renders this through its Markdown parser, which treats any line
    indented four spaces or more as a code block and would print the markup as
    text instead of rendering it.
    """
    return "\n".join(line.lstrip() for line in markup.splitlines() if line.strip())

# ==========================================================================
# APPLICATION — TripWise AI — an AI travel platform.
# ==========================================================================








# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_catalogue():
    return load_catalogue_file()


@st.cache_resource(show_spinner=False)
def get_recommender(_df, fingerprint: str):
    """fingerprint keeps the cache keyed to the catalogue, not the frame object."""
    return Recommender(_df)


def chart(fig) -> None:
    """Render a figure full width across Streamlit versions.

    `use_container_width` is deprecated in favour of `width`, but older installs
    do not accept `width`, so try the current API and fall back.
    """
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def card_payload(row, prefs: dict) -> dict:
    """Flatten a matched row into exactly what the card template needs."""
    name, code = airport_of(row)
    airport = None
    if name and code:
        airport = f"{name} ({code})"
    elif name or code:
        airport = name or code

    tastes = tastes_of(row)
    temp = float(row.get("temp_avg_yearly", 20) or 20)
    tier = int(float(row.get("budget_level_encoded", 2) or 2))

    return {
        "city": clean(row.get("city")) or "Unknown",
        "country": clean(row.get("country")) or "",
        "match": f"{float(row.get('match', 0)):.0f}",
        "art": destination_art(str(row.get("city", "x")), tastes, temp),
        "tier": BUDGET_LABEL.get(min(3, max(1, tier)), "Mid-range"),
        "daily": daily_cost(row),
        "temp": f"{temp:.0f}",
        "style": trip_style(row),
        "region": region_of(row),
        "why": explain(row, prefs),
        "airport": airport,
        "season": best_season(row),
        "climate": climate_summary(row),
        "hotel": clean(row.get("HotelName")),
        "attractions": attractions_of(row),
        "tip": travel_tip(row),
    }


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #

def page_home(df, is_real: bool, missing: list[str]) -> None:
    html(navbar("home"))
    html(hero(catalogue_stats(df)))

    if not is_real:
        note = (
            f"Missing columns: {', '.join(missing)}." if missing
            else "tripwise_data.csv was not found beside app.py."
        )
        st.info(
            f"Running on the built-in demo catalogue. {note} Export final_df from "
            "the notebook to run TripWise on your own data.",
            icon="ℹ️",
        )

    html(section_head(
        "How it works", "Three steps to a shortlist you trust",
        "No account, no browsing through hundreds of listings. Describe how you "
        "travel and the ranking does the narrowing.", anchor="features"))
    html('<div class="tw-grid tw-grid--3">'
         + feature_card("compass", "Describe your trip",
                           "Nine travel dimensions, a budget tier and a climate target. "
                           "It takes about a minute and you can revise any of it.", 1)
         + feature_card("sparkle", "See a ranked shortlist",
                           "Destinations are ordered by how closely they match what you "
                           "described, each one explaining its own reasoning.", 2)
         + feature_card("wallet", "Plan around real numbers",
                           "Daily spend, best season, nearest airport and local tips — "
                           "enough to judge a trip before you commit to it.", 3)
         + "</div>" + close_section())

    html(section_head(
        "Capabilities", "Everything you need to travel smarter",
        "Built for travellers who want a decision, not a search results page."))
    html('<div class="tw-grid tw-grid--3">'
         + feature_card("globe", "Global catalogue",
                           "Destinations across every inhabited region, each carrying "
                           "climate, budget and accommodation data.", 1)
         + feature_card("wallet", "Costs before booking",
                           "Per-day estimates from budget tier and accommodation "
                           "standard, scaled to your party and trip length.", 2)
         + feature_card("sun", "Season guidance",
                           "The months worth travelling in, inferred from latitude and "
                           "climate rather than guessed.", 3)
         + feature_card("plane", "Nearest airport",
                           "The gateway airport and its code for every destination that "
                           "has one on record.", 4)
         + feature_card("chart", "Readable insights",
                           "Plain-language readings of your results — climate fit, cost "
                           "spread and how your matches cluster.", 5)
         + feature_card("shield", "Honest about gaps",
                           "When a detail is missing from the catalogue TripWise says so "
                           "instead of inventing a plausible answer.", 6)
         + "</div>" + close_section())

    html(section_head(
        "About", "Advanced recommendation technology, quietly",
        "You should not need to understand the machinery to trust the result.",
        anchor="about"))
    html(about_block() + close_section())

    html(section_head("Contact", "Ready when you are", "", anchor="contact"))
    html(contact_block() + close_section())
    html(footer())


def page_planner(df, is_real: bool) -> None:
    html(navbar("planner"))
    html("""
<div class="tw-section" style="margin-top:8px;">
  <div class="tw-section__head tw-rise">
    <span class="tw-eyebrow">AI Planner</span>
    <h2>Tell us how you travel</h2>
    <p>Rate what matters to you. 1 means you are indifferent, 5 means it would
       shape the whole trip.</p>
  </div>
</div>""")

    with st.form("planner", border=False):
        html('<div class="tw-panel">')

        html('<div class="tw-fieldset__title">What you care about</div>')
        answers: dict = {}
        cols = st.columns(3, gap="large")
        for i, (key, label, hint) in enumerate(TASTES):
            with cols[i % 3]:
                answers[key] = st.slider(label, 1, 5, 3, help=hint, key=f"t_{key}")

        html('<div class="tw-fieldset__title">Budget and climate</div>')
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            answers["budget"] = st.select_slider(
                "Budget level", options=[1, 2, 3], value=2,
                format_func=lambda v: BUDGET_LABEL[v])
        with c2:
            answers["temp"] = st.slider("Preferred average temperature (°C)", -5, 40, 22)
        with c3:
            answers["stars"] = st.slider("Minimum accommodation standard", 1.0, 5.0, 4.0, 0.5)

        html('<div class="tw-fieldset__title">Where and how long</div>')
        c4, c5, c6 = st.columns(3, gap="large")
        with c4:
            region_names = ["Anywhere"] + [REGION_LABEL[c] for c in REGION_COLS]
            picked = st.selectbox("Preferred region", region_names)
            answers["region"] = next(
                (c for c in REGION_COLS if REGION_LABEL[c] == picked), None)
        with c5:
            length = st.radio("Trip length", ["Short trip", "One week"], horizontal=True)
            answers["trip_length"] = "short" if length == "Short trip" else "week"
            answers["nights"] = 4 if length == "Short trip" else 7
        with c6:
            answers["travellers"] = st.number_input("Travellers", 1, 12, 2)

        answers["needs_airport"] = st.checkbox(
            "Only show destinations with a nearby airport", value=True)

        html("</div>")
        st.markdown("<div style='height:1.3rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Find my destinations  →", type="primary")

    if not submitted:
        html(footer())
        return

    prefs = build_preferences(df, answers)
    pool = df
    if answers["needs_airport"] and "has_airport" in df.columns:
        gated = df[df.apply(lambda r: any(airport_of(r)), axis=1)]
        if len(gated) >= 6:
            pool = gated

    with st.spinner("Ranking destinations…"):
        model = get_recommender(pool, f"{len(pool)}-{tuple(pool.columns)[:3]}")
        results = model.rank(prefs, top_n=6)

    render_results(results, prefs, answers)


def render_results(results, prefs: dict, answers: dict) -> None:
    if results.empty:
        st.warning("No destinations matched. Try widening the region or climate.")
        return

    html(section_head(
        "Your matches", "Where you should go",
        f"Ranked against your answers. Estimates assume {answers['travellers']} "
        f"traveller(s) over {answers['nights']} nights."))
    cards = "".join(
        destination_card(card_payload(row, prefs), delay=(i % 6) + 1)
        for i, (_, row) in enumerate(results.iterrows())
    )
    html(f'<div class="tw-grid tw-grid--3">{cards}</div>' + close_section())

    readings = insights(results, prefs, answers)
    if readings:
        html(section_head(
            "AI insights", "What your results say",
            "Read from the shortlist as a whole, not any single destination."))
        html('<div class="tw-grid tw-grid--2">'
             + "".join(insight_card(t, b, (i % 6) + 1)
                       for i, (t, b) in enumerate(readings))
             + "</div>" + close_section())

    render_charts(results, answers)
    html(footer())


def render_charts(results, answers: dict) -> None:
    import plotly.express as px

    html(section_head("Comparison", "Your shortlist side by side",
                         "The same six destinations, measured two ways."))
    left, right = st.columns(2, gap="large")
    axis = dict(showgrid=True, gridcolor="rgba(9,16,32,.07)", zeroline=False,
                tickfont=dict(color="#55637A", size=12),
                title_font=dict(color="#55637A", size=12))
    layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="Inter, sans-serif", color="#080D18", size=13),
                  margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False)

    with left:
        html('<div class="tw-card tw-chart"><h4>Match strength</h4>'
             '<p>How closely each destination sits to your answers.</p>')
        fig = px.bar(results.sort_values("match"), x="match", y="city", orientation="h",
                     color="match", color_continuous_scale=["#BAE6FD", "#0EA5E9", "#2563EB"])
        fig.update_layout(**layout, xaxis=dict(**axis, title="Match %"),
                          yaxis=dict(**axis, title=""), coloraxis_showscale=False)
        fig.update_traces(hovertemplate="%{y}: %{x:.0f}%<extra></extra>")
        chart(fig)
        html("</div>")

    with right:
        costs = results.assign(
            daily=[daily_cost(r) for _, r in results.iterrows()])
        nights, people = answers["nights"], answers["travellers"]
        costs["total"] = costs["daily"] * nights * people
        html('<div class="tw-card tw-chart"><h4>Estimated trip cost</h4>'
             f'<p>{people} traveller(s), {nights} nights, before flights.</p>')
        fig = px.bar(costs.sort_values("total"), x="total", y="city", orientation="h",
                     color="total", color_continuous_scale=["#BAE6FD", "#0EA5E9", "#2563EB"])
        fig.update_layout(**layout, xaxis=dict(**axis, title="US$"),
                          yaxis=dict(**axis, title=""), coloraxis_showscale=False)
        fig.update_traces(hovertemplate="%{y}: $%{x:,.0f}<extra></extra>")
        chart(fig)
        html("</div>")

    html(close_section())


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #

def main() -> None:
    params = st.query_params
    view = params.get("view", "home")

    html(THEME_CSS)
    df, is_real, missing = load_catalogue()

    # The splash is painted over the page that renders beneath it in this same
    # run, then fades itself out — so there is never a blank frame, no blocking
    # sleep and no second rerun. It plays only on a bare first load; every link
    # in the app carries a query param, so navigation never replays it.
    if len(params) == 0 and not st.session_state.get("seen_splash", False):
        st.session_state["seen_splash"] = True
        html(build_splash())

    if view == "planner":
        page_planner(df, is_real)
    else:
        page_home(df, is_real, missing)


if __name__ == "__main__":
    main()
