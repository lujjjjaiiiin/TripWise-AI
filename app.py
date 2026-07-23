"""
TripWise AI — Premium Streamlit Travel Assistant
Splash screen (SVG heart-path flight animation) -> Main App (Home / nav pages)
Destination Explorer page runs the real Cosine Similarity recommendation engine.
"""

import time
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
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


SPLASH_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@600;800&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{
  display:grid;place-items:center;
  background:linear-gradient(180deg,#f1f4f7 0%,#e3e9ef 42%,#ced6df 100%);
  animation:out .8s ease 4.5s forwards;
}
@keyframes out{to{opacity:0}}

.scene{width:min(96vw,1240px);aspect-ratio:1200/620;
  animation:scenein 1.2s cubic-bezier(.2,.7,.3,1) both}
@keyframes scenein{from{opacity:0;transform:scale(1.05)}to{opacity:1;transform:none}}
svg{width:100%;height:100%;display:block;
  filter:drop-shadow(0 30px 64px rgba(66,80,96,.22))}

/* shades: hold closed, then glide up out of the opening */
.shade{transform:translateY(0);animation:up 2.2s cubic-bezier(.72,0,.18,1) forwards}
@keyframes up{to{transform:translateY(-386px)}}

/* clouds drift sideways at different speeds for parallax */
.drift{animation-name:dr;animation-timing-function:ease-in-out;
  animation-iteration-count:infinite;animation-direction:alternate}
@keyframes dr{from{transform:translateX(calc(var(--dx) * -1))}
              to{transform:translateX(var(--dx))}}

.mark{position:absolute;left:0;right:0;bottom:6%;text-align:center;opacity:0;
  animation:mk 1s cubic-bezier(.2,.7,.3,1) 2.8s both}
.mark .t{font-family:'Outfit',sans-serif;font-weight:800;
  font-size:clamp(1.5rem,3.6vw,2.45rem);letter-spacing:-.005em;
  background:linear-gradient(96deg,#0EA5E9,#2563EB);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
  color:#2563EB;
  filter:drop-shadow(0 0 22px rgba(14,165,233,.45)) drop-shadow(0 2px 3px rgba(255,255,255,.95))}
.mark .s{font-family:'Inter',sans-serif;margin-top:.5rem;color:#5A6B7E;font-weight:500;
  font-size:clamp(.62rem,1.35vw,.8rem);letter-spacing:.34em;text-transform:uppercase;
  text-shadow:0 1px 2px rgba(255,255,255,.9)}
@keyframes mk{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}

@media (prefers-reduced-motion:reduce){
  .shade{animation-duration:.01s}
  .drift{animation:none}
  body{animation:out .5s linear 2s forwards}
}
</style></head><body>
<div class="scene">
<svg viewBox="0 0 1200 620" xmlns="http://www.w3.org/2000/svg">
<defs>
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
  <linearGradient id="bezel" x1=".25" y1="0" x2=".75" y2="1">
    <stop offset="0" stop-color="#ffffff"/><stop offset=".5" stop-color="#f4f6f8"/>
    <stop offset="1" stop-color="#dbe1e7"/>
  </linearGradient>
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
  <linearGradient id="mDeepg" x1="0" y1="0" x2="0" y2="1"><stop offset="40%" stop-color="#fff" stop-opacity="0"/><stop offset="66%" stop-color="#fff" stop-opacity="1"/><stop offset="100%" stop-color="#fff" stop-opacity="1"/></linearGradient><mask id="mDeep" maskUnits="userSpaceOnUse" x="-40" y="0" width="300" height="326"><rect x="-40" y="0" width="300" height="326" fill="url(#mDeepg)"/></mask><linearGradient id="mDeckg" x1="0" y1="0" x2="0" y2="1"><stop offset="44%" stop-color="#fff" stop-opacity="0"/><stop offset="70%" stop-color="#fff" stop-opacity="1"/><stop offset="100%" stop-color="#fff" stop-opacity="1"/></linearGradient><mask id="mDeck" maskUnits="userSpaceOnUse" x="-40" y="0" width="300" height="326"><rect x="-40" y="0" width="300" height="326" fill="url(#mDeckg)"/></mask><linearGradient id="mLowg" x1="0" y1="0" x2="0" y2="1"><stop offset="58%" stop-color="#fff" stop-opacity="0"/><stop offset="84%" stop-color="#fff" stop-opacity="1"/><stop offset="100%" stop-color="#fff" stop-opacity="1"/></linearGradient><mask id="mLow" maskUnits="userSpaceOnUse" x="-40" y="0" width="300" height="326"><rect x="-40" y="0" width="300" height="326" fill="url(#mLowg)"/></mask><linearGradient id="mHighg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#fff" stop-opacity="1"/><stop offset="30%" stop-color="#fff" stop-opacity="0"/><stop offset="100%" stop-color="#fff" stop-opacity="0"/></linearGradient><mask id="mHigh" maskUnits="userSpaceOnUse" x="-40" y="0" width="300" height="326"><rect x="-40" y="0" width="300" height="326" fill="url(#mHighg)"/></mask>
  <filter id="cA_0" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.009" numOctaves="5" seed="11" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -1.9 0 0 0 1.18"/></filter><filter id="cB_0" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.017" numOctaves="6" seed="5" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -2.5 0 0 0 1.45"/></filter><filter id="cC_0" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.034" numOctaves="5" seed="29" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -3.0 0 0 0 1.72"/></filter><filter id="cD_0" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.013" numOctaves="4" seed="41" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -2.6 0 0 0 1.3"/></filter><filter id="cA_1" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.009" numOctaves="5" seed="71" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -1.9 0 0 0 1.18"/></filter><filter id="cB_1" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.017" numOctaves="6" seed="65" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -2.5 0 0 0 1.45"/></filter><filter id="cC_1" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.034" numOctaves="5" seed="89" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -3.0 0 0 0 1.72"/></filter><filter id="cD_1" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.013" numOctaves="4" seed="101" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -2.6 0 0 0 1.3"/></filter><filter id="cA_2" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.009" numOctaves="5" seed="131" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -1.9 0 0 0 1.18"/></filter><filter id="cB_2" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.017" numOctaves="6" seed="125" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -2.5 0 0 0 1.45"/></filter><filter id="cC_2" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.034" numOctaves="5" seed="149" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -3.0 0 0 0 1.72"/></filter><filter id="cD_2" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="0.013" numOctaves="4" seed="161" stitchTiles="stitch"/><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  -2.6 0 0 0 1.3"/></filter>
</defs>
  <rect width="1200" height="620" fill="url(#cabin)"/>
  <rect x="0" y="34" width="1200" height="2" fill="#c3cbd4" opacity=".55"/>
  <rect x="0" y="37" width="1200" height="1" fill="#ffffff" opacity=".9"/>
  <rect x="0" y="561" width="1200" height="2" fill="#c3cbd4" opacity=".5"/>
  <rect x="0" y="564" width="1200" height="1" fill="#ffffff" opacity=".85"/>

  <g transform="translate(72,78)">
    <rect x="0" y="0" width="300" height="430" rx="96" ry="78" fill="url(#bezel)"
          filter="url(#winShadow)"/>
    <rect x="0" y="0" width="300" height="430" rx="96" ry="78" fill="none"
          stroke="#aab3bd" stroke-width="1.1" opacity=".5"/>
    <rect x="24" y="24" width="252" height="382" rx="76" ry="60" fill="url(#well)"/>
    <rect x="32" y="34" width="236" height="362" rx="68" ry="54"
          fill="#5c6672" opacity=".5"/>

    <clipPath id="glass0">
      <rect x="40" y="46" width="220" height="326" rx="62" ry="50"/>
    </clipPath>

    <g clip-path="url(#glass0)">
      <rect x="40" y="46" width="220" height="326" fill="url(#sky)"/>
      <g transform="translate(40,46)"><g class="drift" style="animation-duration:26s;animation-delay:-0s;--dx:16px"><rect x="0" y="46" width="300" height="326" filter="url(#cA_0)" mask="url(#mDeep)" opacity="0.85" transform="translate(0,0)"/></g><g class="drift" style="animation-duration:19s;animation-delay:-3s;--dx:26px"><rect x="0" y="46" width="300" height="326" filter="url(#cB_0)" mask="url(#mDeck)" opacity="1.0" transform="translate(0,0)"/></g><g class="drift" style="animation-duration:13s;animation-delay:-6s;--dx:38px"><rect x="0" y="46" width="300" height="326" filter="url(#cC_0)" mask="url(#mLow)" opacity="0.9" transform="translate(0,0)"/></g><g class="drift" style="animation-duration:34s;animation-delay:-9s;--dx:12px"><rect x="0" y="46" width="300" height="326" filter="url(#cD_0)" mask="url(#mHigh)" opacity="0.28" transform="translate(0,0)"/></g></g>
      <ellipse cx="202.8" cy="156.84" rx="154.0" ry="136.92"
               fill="url(#sun)" opacity=".55"/>

      <!-- shade: starts closed, glides up out of the opening -->
      <g class="shade" style="animation-delay:1.0s">
        <rect x="36" y="44" width="228" height="330" fill="url(#shade)"/>
        <rect x="36" y="332" width="228" height="42" fill="url(#lipG)"/>
        <rect x="120.0" y="346" width="60" height="9" rx="4.5"
              fill="url(#gripG)"/>
      </g>

      <!-- glass depth + reflection, always above the shade -->
      <rect x="40" y="46" width="220" height="26" fill="#33404f" opacity=".34"/>
      <rect x="40" y="46" width="13" height="326" fill="#33404f" opacity=".2"/>
      <rect x="247" y="46" width="13" height="326" fill="#33404f" opacity=".2"/>
      <path d="M40 296 L190 46 L250 46 L40 372 Z"
            fill="#fff" opacity=".08"/>
    </g>

    <path d="M58 52 C100 20 200 20 242 52
             C200 34 100 34 58 52 Z" fill="url(#lipHi)"/>
  </g>
  <g transform="translate(450,78)">
    <rect x="0" y="0" width="300" height="430" rx="96" ry="78" fill="url(#bezel)"
          filter="url(#winShadow)"/>
    <rect x="0" y="0" width="300" height="430" rx="96" ry="78" fill="none"
          stroke="#aab3bd" stroke-width="1.1" opacity=".5"/>
    <rect x="24" y="24" width="252" height="382" rx="76" ry="60" fill="url(#well)"/>
    <rect x="32" y="34" width="236" height="362" rx="68" ry="54"
          fill="#5c6672" opacity=".5"/>

    <clipPath id="glass1">
      <rect x="40" y="46" width="220" height="326" rx="62" ry="50"/>
    </clipPath>

    <g clip-path="url(#glass1)">
      <rect x="40" y="46" width="220" height="326" fill="url(#sky)"/>
      <g transform="translate(40,46)"><g class="drift" style="animation-duration:28s;animation-delay:-1s;--dx:16px"><rect x="0" y="46" width="300" height="326" filter="url(#cA_1)" mask="url(#mDeep)" opacity="0.85" transform="translate(0,0)"/></g><g class="drift" style="animation-duration:21s;animation-delay:-4s;--dx:26px"><rect x="0" y="46" width="300" height="326" filter="url(#cB_1)" mask="url(#mDeck)" opacity="1.0" transform="translate(0,0)"/></g><g class="drift" style="animation-duration:15s;animation-delay:-7s;--dx:38px"><rect x="0" y="46" width="300" height="326" filter="url(#cC_1)" mask="url(#mLow)" opacity="0.9" transform="translate(0,0)"/></g><g class="drift" style="animation-duration:36s;animation-delay:-10s;--dx:12px"><rect x="0" y="46" width="300" height="326" filter="url(#cD_1)" mask="url(#mHigh)" opacity="0.28" transform="translate(0,0)"/></g></g>
      <ellipse cx="202.8" cy="156.84" rx="154.0" ry="136.92"
               fill="url(#sun)" opacity=".55"/>

      <!-- shade: starts closed, glides up out of the opening -->
      <g class="shade" style="animation-delay:1.12s">
        <rect x="36" y="44" width="228" height="330" fill="url(#shade)"/>
        <rect x="36" y="332" width="228" height="42" fill="url(#lipG)"/>
        <rect x="120.0" y="346" width="60" height="9" rx="4.5"
              fill="url(#gripG)"/>
      </g>

      <!-- glass depth + reflection, always above the shade -->
      <rect x="40" y="46" width="220" height="26" fill="#33404f" opacity=".34"/>
      <rect x="40" y="46" width="13" height="326" fill="#33404f" opacity=".2"/>
      <rect x="247" y="46" width="13" height="326" fill="#33404f" opacity=".2"/>
      <path d="M40 296 L190 46 L250 46 L40 372 Z"
            fill="#fff" opacity=".08"/>
    </g>

    <path d="M58 52 C100 20 200 20 242 52
             C200 34 100 34 58 52 Z" fill="url(#lipHi)"/>
  </g>
  <g transform="translate(828,78)">
    <rect x="0" y="0" width="300" height="430" rx="96" ry="78" fill="url(#bezel)"
          filter="url(#winShadow)"/>
    <rect x="0" y="0" width="300" height="430" rx="96" ry="78" fill="none"
          stroke="#aab3bd" stroke-width="1.1" opacity=".5"/>
    <rect x="24" y="24" width="252" height="382" rx="76" ry="60" fill="url(#well)"/>
    <rect x="32" y="34" width="236" height="362" rx="68" ry="54"
          fill="#5c6672" opacity=".5"/>

    <clipPath id="glass2">
      <rect x="40" y="46" width="220" height="326" rx="62" ry="50"/>
    </clipPath>

    <g clip-path="url(#glass2)">
      <rect x="40" y="46" width="220" height="326" fill="url(#sky)"/>
      <g transform="translate(40,46)"><g class="drift" style="animation-duration:30s;animation-delay:-2s;--dx:16px"><rect x="0" y="46" width="300" height="326" filter="url(#cA_2)" mask="url(#mDeep)" opacity="0.85" transform="translate(0,0)"/></g><g class="drift" style="animation-duration:23s;animation-delay:-5s;--dx:26px"><rect x="0" y="46" width="300" height="326" filter="url(#cB_2)" mask="url(#mDeck)" opacity="1.0" transform="translate(0,0)"/></g><g class="drift" style="animation-duration:17s;animation-delay:-8s;--dx:38px"><rect x="0" y="46" width="300" height="326" filter="url(#cC_2)" mask="url(#mLow)" opacity="0.9" transform="translate(0,0)"/></g><g class="drift" style="animation-duration:38s;animation-delay:-11s;--dx:12px"><rect x="0" y="46" width="300" height="326" filter="url(#cD_2)" mask="url(#mHigh)" opacity="0.28" transform="translate(0,0)"/></g></g>
      <ellipse cx="202.8" cy="156.84" rx="154.0" ry="136.92"
               fill="url(#sun)" opacity=".55"/>

      <!-- shade: starts closed, glides up out of the opening -->
      <g class="shade" style="animation-delay:1.24s">
        <rect x="36" y="44" width="228" height="330" fill="url(#shade)"/>
        <rect x="36" y="332" width="228" height="42" fill="url(#lipG)"/>
        <rect x="120.0" y="346" width="60" height="9" rx="4.5"
              fill="url(#gripG)"/>
      </g>

      <!-- glass depth + reflection, always above the shade -->
      <rect x="40" y="46" width="220" height="26" fill="#33404f" opacity=".34"/>
      <rect x="40" y="46" width="13" height="326" fill="#33404f" opacity=".2"/>
      <rect x="247" y="46" width="13" height="326" fill="#33404f" opacity=".2"/>
      <path d="M40 296 L190 46 L250 46 L40 372 Z"
            fill="#fff" opacity=".08"/>
    </g>

    <path d="M58 52 C100 20 200 20 242 52
             C200 34 100 34 58 52 Z" fill="url(#lipHi)"/>
  </g>
  <rect width="1200" height="620" fill="url(#vign)" pointer-events="none"/>
</svg>
</div>
<div class="mark">
  <div class="t">TripWise AI</div>
  <div class="s">Plan smarter &middot; Travel further</div>
</div>
</body></html>"""


SPLASH_CHROME_CSS = """
<style>
header, #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stSidebar"]{display:none !important;}
[data-testid="stAppViewContainer"] .main .block-container{padding:0 !important;max-width:100% !important;}
.stApp{background:#070c16 !important;}
iframe{position:fixed !important; inset:0 !important; width:100vw !important; height:100vh !important;
       border:0 !important; z-index:99999 !important;}
</style>
"""


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
        .tw-section-title{
            font-weight:800; font-size:2rem; color:var(--text-dark);
            margin: 2.6rem 0 1.2rem 0; letter-spacing:-.015em;
            text-shadow: 0 1px 0 rgba(255,255,255,.85);
        }
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
            font-size: 3.6rem; font-weight:800; color:var(--text-dark);
            line-height:1.16; margin-bottom:1.2rem; letter-spacing:-.02em;
            text-shadow: 0 1px 0 rgba(255,255,255,.9), 0 10px 34px rgba(37,99,235,.14);
            animation: fadeUp .9s ease both;
        }
        /* the product name itself carries the brand gradient and a soft glow */
        .tw-hero h1 .tw-glow{
            background: linear-gradient(96deg, var(--sky), var(--blue));
            -webkit-background-clip:text; background-clip:text;
            -webkit-text-fill-color:transparent; color:var(--blue);
            filter: drop-shadow(0 0 20px rgba(14,165,233,.4));
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

        /* Floating travel icons. clamp() lets them grow on desktop without
           overflowing narrow screens; flex-wrap is the safety net below that. */
        .tw-illustration-row{
            display:flex; justify-content:center; align-items:center; flex-wrap:wrap;
            gap: clamp(1.2rem, 3.5vw, 3rem);
            font-size: clamp(3.2rem, 8vw, 7rem);
            line-height:1; margin: 2rem 0 3rem 0;
        }
        .tw-illustration-row span{
            filter: drop-shadow(0 14px 26px rgba(37,99,235,.28));
        }

        .tw-icon-circle{
            width: clamp(88px, 9vw, 116px); height: clamp(88px, 9vw, 116px);
            border-radius:28px;
            background: linear-gradient(135deg, var(--sky), var(--blue));
            display:flex; align-items:center; justify-content:center;
            font-size: clamp(2.6rem, 4vw, 3.4rem); line-height:1;
            margin-bottom:1.1rem; color:white;
            box-shadow: 0 16px 34px rgba(14,165,233,.34);
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

        /* ---------------- NAV / SECTION-LINK CARDS (Home -> Insights / Models) ---------------- */
        .tw-navcard{
            display:block; text-decoration:none; height:100%;
        }
        .tw-navcard-inner{
            background: rgba(219,234,254,0.6);
            backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(30,58,138,0.15); border-radius: 24px;
            padding: 2rem; height:100%; box-shadow: var(--shadow);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SPLASH SCREEN
# ----------------------------------------------------------------------------
def render_splash():
    """Airplane-window boarding splash rendered via components.html (an iframe).

    components.html reliably renders full HTML/CSS/JS. st.markdown does not: its
    Markdown parser treats indented HTML as a code block and prints it as text,
    which is why the raw CSS was showing on screen.

    Shades start closed, hold ~1s, then glide up to reveal the clouds. The photo
    frame is a fixed PNG with the glass punched out, so only the shades and the
    drifting clouds move (subtle in-flight parallax).
    """
    components.html(SPLASH_HTML, height=760, scrolling=False)


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
            <h1>✈️ Plan Smarter with<br/><span class="tw-glow">TripWise AI</span></h1>
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

    # Instead of rendering the Insights charts and the Models explanation inline,
    # link out to their own dedicated pages — keeps Home short and focused.
    st.markdown('<div class="tw-section-title">Explore more</div>', unsafe_allow_html=True)
    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown(
            """
            <div class="tw-navcard-inner">
                <div class="tw-icon-circle">📊</div>
                <div style="font-weight:800; font-size:1.3rem; margin-bottom:.5rem; color:#0F172A;">Key Insights</div>
                <div class="tw-muted" style="font-size:1.02rem; line-height:1.6;">
                    Charts on destinations by region, budget level, and climate across our dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Insights →", use_container_width=True, key="go_insights"):
            go_to("Insights")
    with nc2:
        st.markdown(
            """
            <div class="tw-navcard-inner">
                <div class="tw-icon-circle">🤖</div>
                <div style="font-weight:800; font-size:1.3rem; margin-bottom:.5rem; color:#0F172A;">Models Used</div>
                <div class="tw-muted" style="font-size:1.02rem; line-height:1.6;">
                    How Cosine Similarity and K-Means power the recommendation engine.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Models →", use_container_width=True, key="go_models"):
            go_to("Models Used")


# ----------------------------------------------------------------------------
# PAGE: KEY INSIGHTS (own page, linked from Home)
# ----------------------------------------------------------------------------
def page_insights(df: pd.DataFrame):
    st.markdown('<div class="tw-section-title">📊 Key Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tw-muted">A quick look at the destinations in our dataset.</p>',
        unsafe_allow_html=True,
    )

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
            font=dict(color="#0F172A", size=15, family="Inter, sans-serif"),
            title_font=dict(color="#0F172A", size=19, family="Outfit, sans-serif"),
            legend=dict(font=dict(color="#0F172A", size=13)),
            margin=dict(t=60, b=20, l=20, r=20), height=380,
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
            font=dict(color="#0F172A", size=15, family="Inter, sans-serif"),
            title_font=dict(color="#0F172A", size=19, family="Outfit, sans-serif"),
            xaxis=dict(color="#0F172A", tickfont=dict(color="#0F172A", size=13), gridcolor="rgba(15,23,42,0.08)"),
            yaxis=dict(color="#0F172A", tickfont=dict(color="#0F172A", size=13), gridcolor="rgba(15,23,42,0.08)"),
            showlegend=False, margin=dict(t=60, b=20, l=20, r=20), height=380,
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
        font=dict(color="#0F172A", size=15, family="Inter, sans-serif"),
        title_font=dict(color="#0F172A", size=19, family="Outfit, sans-serif"),
        xaxis=dict(color="#0F172A", tickfont=dict(color="#0F172A", size=13), gridcolor="rgba(15,23,42,0.08)"),
        yaxis=dict(color="#0F172A", tickfont=dict(color="#0F172A", size=13), gridcolor="rgba(15,23,42,0.08)"),
        coloraxis_showscale=False, margin=dict(t=60, b=20, l=20, r=20), height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE: MODELS USED (own page, linked from Home)
# ----------------------------------------------------------------------------
def page_models_used():
    st.markdown('<div class="tw-section-title">🤖 Models Used</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tw-muted">The two techniques behind TripWise AI\'s recommendations.</p>',
        unsafe_allow_html=True,
    )

    models = [
        {
            "icon": "🎯",
            "name": "Cosine Similarity",
            "purpose": "Recommends destinations that best match an individual traveler's stated preferences.",
            "how": "Every destination and every user profile is represented as a vector of scaled features (culture, nature, budget, climate...). Cosine Similarity measures the angle between the user's vector and each destination's vector — the smaller the angle, the closer the match.",
            "why": "It works well with structured, interpretable features and doesn't require historical user ratings, which this dataset doesn't have.",
            "contribution": "It powers the core \"Find my destinations\" engine on the Destination Explorer page.",
        },
        {
            "icon": "🧩",
            "name": "K-Means Clustering",
            "purpose": "Groups destinations with similar characteristics into clusters, revealing natural travel-style patterns in the data.",
            "how": "The algorithm partitions all destinations into K groups by minimizing the distance between each destination and its cluster's center, based on the same scaled features used for recommendations.",
            "why": "It's a simple, well-established unsupervised method suited for exploratory pattern discovery when there are no predefined labels.",
            "contribution": "It helps surface destination \"types\" (e.g. beach & budget-friendly vs. cultural & luxury) that complement the direct recommendations.",
        },
    ]

    cols = st.columns(2)
    for c, m in zip(cols, models):
        with c:
            st.markdown(
                f"""
                <div class="tw-glass" style="min-height:100%;">
                    <div class="tw-icon-circle">{m['icon']}</div>
                    <div style="font-weight:800; font-size:1.35rem; margin-bottom:.8rem; color:#0F172A;">{m['name']}</div>
                    <div style="margin-bottom:.7rem;"><span style="font-weight:700; color:#2563EB;">Purpose — </span><span class="tw-muted">{m['purpose']}</span></div>
                    <div style="margin-bottom:.7rem;"><span style="font-weight:700; color:#2563EB;">How it works — </span><span class="tw-muted">{m['how']}</span></div>
                    <div style="margin-bottom:.7rem;"><span style="font-weight:700; color:#2563EB;">Why we chose it — </span><span class="tw-muted">{m['why']}</span></div>
                    <div><span style="font-weight:700; color:#2563EB;">Contribution — </span><span class="tw-muted">{m['contribution']}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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


def _airport_of(row):
    """Best available airport identity for a row, as (name, code).

    The notebook fills cities that found no airport match with "Unknown", so a
    row may carry an "Unknown" name, a has_airport flag of 0, or no airport
    columns at all if the CSV was exported without them. Any of those means
    there is nothing to show, and each is reported the same way.
    """
    name = _clean(row.get("name"))
    code = _clean(row.get("iata")) or _clean(row.get("icao"))
    return (name, code) if (name or code) else (None, None)


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

        missing = [c for c in ("name", "iata") if c not in df.columns]
        if missing:
            st.warning(
                "No nearest airport can be shown: tripwise_data.csv has no "
                + " or ".join(f"{c}" for c in missing)
                + " column. Re-export final_df from the notebook with the "
                "airport columns included."
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
            airport_name, airport_iata = _airport_of(row)
            has_air = airport_name is not None or airport_iata is not None

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
                label = airport_name or "Airport"
                airport_html = (
                    f'<div class="tw-detail"><span class="tw-detail-ic">🛫</span>'
                    f'<span><b>Nearest airport</b> — {label}{code}</span></div>'
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

    # First open only: show the boarding splash full-screen, then hand off to the
    # homepage. The component is an iframe, so we blow it up to cover the viewport
    # and hide Streamlit's chrome while it plays; during the splash nothing else is
    # rendered, so targeting iframe here is safe (no charts exist yet).
    if not st.session_state.splash_done:
        st.markdown(SPLASH_CHROME_CSS, unsafe_allow_html=True)
        render_splash()
        time.sleep(5.2)
        st.session_state.splash_done = True
        st.session_state.fade_home = True
        st.rerun()
        st.stop()

    # gentle one-time fade of the homepage right after the splash
    if st.session_state.pop("fade_home", False):
        st.markdown(
            "<style>@keyframes twHomeIn{from{opacity:0}to{opacity:1}}"
            ".block-container{animation:twHomeIn .7s ease both;}</style>",
            unsafe_allow_html=True,
        )

    render_sidebar()
    render_back_button()

    if st.session_state.page == "Home":
        df = load_data()
        page_home(df)
    elif st.session_state.page == "Destination Explorer":
        df = load_data()
        page_destination_explorer(df)
    elif st.session_state.page == "Insights":
        df = load_data()
        page_insights(df)
    elif st.session_state.page == "Models Used":
        page_models_used()
    else:
        icon, desc = PAGE_CONTENT[st.session_state.page]
        page_placeholder(st.session_state.page, icon, desc)


if __name__ == "__main__":
    main()
