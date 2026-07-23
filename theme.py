"""The whole visual language in one place.

Two jobs: erase Streamlit's default chrome, then supply a design system the rest
of the app builds against. Nothing here knows about travel data — it is purely
tokens and components, so a change to a radius or a shadow lands everywhere at
once instead of being copy-pasted across pages.
"""

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Plus+Jakarta+Sans:wght@400;500;600;700;800&"
    "family=Inter:wght@400;450;500;600&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)

CSS = """
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
/* Streamlit's header sits at the top of the viewport with its own stacking
   context and swallows clicks aimed at the fixed navbar underneath it. */
[data-testid="stHeader"]{ display:none !important; }

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
  position:fixed; top:0; left:0; right:0; height:var(--nav-h); z-index:2147483000;
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
  position:relative; pointer-events:auto; text-decoration:none; color:var(--slate);
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
