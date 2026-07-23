"""
Layer test — finds which part of TripWise breaks interaction.

The plain Streamlit test passed, so the browser and the deployment are fine and
the fault is inside the application. TripWise adds two things on top of plain
widgets: a stylesheet, and cards built from injected HTML. This file switches
them on one at a time.

Self-contained on purpose — no package folder, no data file, nothing to upload
beyond this single file.

HOW TO USE

Put ?level=1 at the end of the page address, press Enter, then press the button
and click the tabs. Repeat for 2 and 3:

    ...streamlit.app/?level=1     plain widgets       (expected to work)
    ...streamlit.app/?level=2     + the stylesheet
    ...streamlit.app/?level=3     + custom HTML cards

The first level where the count stops rising is the culprit. The level is read
from the address bar so it can be changed even if clicking is broken.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="TripWise layer test", page_icon="🔬", layout="wide")

STYLESHEET = """
<style>
@import url('__FONTS__');

:root{
  --ink:#0A1220;
  --slate:#5A6B80;
  --slate-2:#8B9AAC;
  --sky:#0EA5E9;
  --blue:#2563EB;
  --line:rgba(10,18,32,.09);
  --card:rgba(255,255,255,.78);
  --r-lg:22px; --r-md:14px; --r-sm:10px;
  --sh-1:0 1px 2px rgba(10,18,32,.04), 0 6px 20px rgba(10,18,32,.06);
  --sh-2:0 20px 46px rgba(37,99,235,.13);
  --ease:cubic-bezier(.22,.75,.28,1);
}

/* ---- page surface ---------------------------------------------------- */
.stApp{
  background:
    radial-gradient(900px 520px at 6% -8%, rgba(14,165,233,.13), transparent 60%),
    radial-gradient(760px 480px at 96% 0%, rgba(37,99,235,.11), transparent 58%),
    linear-gradient(180deg,#FBFDFF 0%,#F2F6FB 55%,#EBF1F8 100%);
  background-attachment:fixed;
}
#MainMenu, footer{ visibility:hidden; }
.block-container{ max-width:1120px; padding-top:2.2rem; padding-bottom:5rem; }

html, body, .stApp, [class*="css"]{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  color:var(--ink); -webkit-font-smoothing:antialiased;
}
::selection{ background:rgba(14,165,233,.2); }

/* ---- type ------------------------------------------------------------ */
.tw-display{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  letter-spacing:-.035em; line-height:1.05; color:var(--ink);
  font-size:clamp(2.1rem,5vw,3.6rem); margin:.6rem 0 1rem;
}
.tw-grad{
  background:linear-gradient(96deg,var(--sky),var(--blue));
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; color:var(--blue);
}
.tw-eyebrow{
  font-family:'JetBrains Mono',ui-monospace,monospace; font-size:.68rem;
  letter-spacing:.22em; text-transform:uppercase; color:var(--blue);
}
.tw-lede{ color:var(--slate); font-size:1.04rem; line-height:1.7; max-width:640px; }
.tw-h2{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  font-size:clamp(1.5rem,3vw,2.1rem); letter-spacing:-.028em;
  margin:.4rem 0 .5rem; color:var(--ink);
}
.tw-sub{ color:var(--slate); font-size:.98rem; line-height:1.65; margin:0 0 .4rem; }
.tw-mono{ font-family:'JetBrains Mono',ui-monospace,monospace; }

/* ---- brand row ------------------------------------------------------- */
.tw-brand{ display:flex; align-items:center; gap:.6rem; margin-bottom:.2rem; }
.tw-brand svg{ width:28px; height:28px; }
.tw-brand b{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.15rem;
  letter-spacing:-.02em;
}

/* ---- stats ----------------------------------------------------------- */
.tw-stats{
  display:flex; gap:clamp(1.4rem,4vw,3rem); flex-wrap:wrap;
  padding:1.4rem 0 .2rem; border-top:1px solid var(--line); margin-top:1.8rem;
}
.tw-stat b{
  display:block; font-family:'JetBrains Mono',ui-monospace,monospace;
  font-weight:500; font-size:clamp(1.3rem,2.6vw,1.75rem);
  letter-spacing:-.03em; color:var(--ink);
}
.tw-stat span{ font-size:.76rem; color:var(--slate); }

/* ---- cards ----------------------------------------------------------- */
.tw-card{
  background:var(--card);
  backdrop-filter:blur(16px) saturate(1.35);
  -webkit-backdrop-filter:blur(16px) saturate(1.35);
  border:1px solid rgba(255,255,255,.86);
  border-radius:var(--r-lg); box-shadow:var(--sh-1);
  transition:transform .34s var(--ease), box-shadow .34s var(--ease);
  height:100%; overflow:hidden;
}
.tw-card:hover{ transform:translateY(-4px); box-shadow:var(--sh-2); }

.tw-feat{ padding:1.6rem 1.4rem; }
.tw-feat__icon{
  width:60px; height:60px; border-radius:18px; display:grid; place-items:center;
  background:linear-gradient(140deg,var(--sky),var(--blue));
  box-shadow:0 12px 26px rgba(14,165,233,.3); margin-bottom:1rem;
}
.tw-feat__icon svg{ width:28px; height:28px; fill:none; stroke:#fff;
  stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.tw-feat h3{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.06rem;
  margin:0 0 .45rem; letter-spacing:-.015em; color:var(--ink);
}
.tw-feat p{ color:var(--slate); font-size:.92rem; line-height:1.65; margin:0; }

/* ---- destination card ------------------------------------------------ */
.tw-dest__art{ position:relative; height:172px; overflow:hidden; }
.tw-dest__art svg{ width:100%; height:100%; display:block; }
.tw-dest__scrim{
  position:absolute; inset:auto 0 0 0; height:66%;
  background:linear-gradient(180deg,transparent,rgba(6,12,24,.7));
}
.tw-dest__place{ position:absolute; left:16px; bottom:12px; right:90px; }
.tw-dest__place h3{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:#fff;
  font-size:1.26rem; letter-spacing:-.02em; margin:0; line-height:1.15;
  text-shadow:0 2px 12px rgba(0,0,0,.42);
}
.tw-dest__place span{ color:rgba(255,255,255,.88); font-size:.8rem; }
.tw-dest__match{
  position:absolute; right:12px; top:12px; padding:.36rem .62rem; border-radius:99px;
  background:rgba(255,255,255,.93);
  font-family:'JetBrains Mono',ui-monospace,monospace; font-size:.74rem;
  color:var(--blue); box-shadow:0 4px 14px rgba(0,0,0,.18);
}
.tw-dest__body{ padding:1.1rem 1.25rem 1.3rem; display:grid; gap:.9rem; }

.tw-facts{ display:grid; grid-template-columns:repeat(3,1fr); gap:.4rem; }
.tw-facts span{
  display:block; font-family:'JetBrains Mono',ui-monospace,monospace;
  font-size:.58rem; letter-spacing:.13em; text-transform:uppercase;
  color:var(--slate-2); margin-bottom:.15rem;
}
.tw-facts b{ font-size:.92rem; font-weight:600; color:var(--ink); }

.tw-why{
  border-left:2px solid var(--sky); padding:.1rem 0 .1rem .8rem;
  color:var(--slate); font-size:.88rem; line-height:1.6;
}
.tw-chips{ display:flex; flex-wrap:wrap; gap:.35rem; }
.tw-chip{
  padding:.26rem .62rem; border-radius:99px; font-size:.72rem; font-weight:500;
  background:rgba(14,165,233,.11); color:#0369A1;
}
.tw-chip--muted{ background:rgba(10,18,32,.05); color:var(--slate); }

.tw-meta{ border-top:1px solid var(--line); padding-top:.8rem; display:grid; gap:.45rem; }
.tw-meta__row{ display:flex; gap:.55rem; align-items:flex-start;
  font-size:.84rem; color:var(--slate); line-height:1.5; }
.tw-meta__row svg{ width:14px; height:14px; flex:0 0 auto; margin-top:3px;
  stroke:var(--sky); fill:none; stroke-width:1.9;
  stroke-linecap:round; stroke-linejoin:round; }
.tw-meta__row b{ color:var(--ink); font-weight:600; }

/* ---- insight --------------------------------------------------------- */
.tw-insight{ padding:1.3rem 1.4rem; display:flex; gap:1rem; align-items:flex-start; }
.tw-insight__dot{
  width:38px; height:38px; border-radius:12px; flex:0 0 auto; display:grid;
  place-items:center; background:linear-gradient(140deg,rgba(14,165,233,.16),rgba(37,99,235,.16));
  border:1px solid rgba(37,99,235,.16);
}
.tw-insight__dot svg{ width:18px; height:18px; stroke:var(--blue); fill:none;
  stroke-width:1.9; stroke-linecap:round; stroke-linejoin:round; }
.tw-insight h4{
  font-family:'Plus Jakarta Sans',sans-serif; font-size:.95rem; font-weight:700;
  margin:0 0 .3rem; color:var(--ink);
}
.tw-insight p{ margin:0; font-size:.89rem; line-height:1.65; color:var(--slate); }

/* ---- Streamlit widgets, styled in place ------------------------------ */
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button{
  font-family:'Inter',sans-serif; font-weight:600; font-size:.94rem;
  border-radius:99px; padding:.66rem 1.5rem; border:1px solid var(--line);
  transition:transform .22s var(--ease), box-shadow .22s var(--ease);
}
.stButton>button[kind="primary"], .stFormSubmitButton>button{
  background:linear-gradient(96deg,var(--sky),var(--blue));
  color:#fff; border:none; box-shadow:0 10px 24px rgba(37,99,235,.3);
}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover{
  transform:translateY(-2px); box-shadow:0 16px 34px rgba(37,99,235,.4);
}
.stButton>button[kind="secondary"]{ background:rgba(255,255,255,.85); color:var(--ink); }
.stButton>button[kind="secondary"]:hover{ border-color:var(--sky); color:var(--blue); }

/* tabs carry the navigation, so they get a little more presence */
.stTabs [data-baseweb="tab-list"]{
  gap:.3rem; border-bottom:1px solid var(--line); padding-bottom:.2rem;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Inter',sans-serif; font-weight:600; font-size:.95rem;
  color:var(--slate); padding:.6rem 1.1rem; border-radius:12px 12px 0 0;
}
.stTabs [aria-selected="true"]{ color:var(--blue) !important; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--blue); height:2.5px; }

[data-testid="stSlider"] label, .stSelectbox label, .stRadio label,
.stCheckbox label, .stNumberInput label, .stMultiSelect label{
  font-size:.86rem !important; font-weight:600 !important; color:var(--ink) !important;
}
[data-testid="stThumbValue"]{
  font-family:'JetBrains Mono',ui-monospace,monospace !important;
  font-size:.72rem !important; color:#fff !important;
  background:var(--blue) !important; padding:.08rem .42rem !important;
  border-radius:7px !important;
}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"]{
  font-family:'JetBrains Mono',ui-monospace,monospace !important;
  font-size:.64rem !important; color:var(--slate-2) !important;
}
div[data-baseweb="select"] > div{
  background:rgba(255,255,255,.9) !important; border-radius:var(--r-sm) !important;
  border:1px solid var(--line) !important; min-height:2.8rem !important;
}
div[role="radiogroup"] label{
  background:rgba(255,255,255,.85); border:1px solid var(--line);
  border-radius:99px; padding:.38rem .9rem !important; margin:0 .4rem 0 0 !important;
}
[data-testid="stMetricValue"]{
  font-family:'JetBrains Mono',ui-monospace,monospace; font-weight:500;
}
[data-testid="stExpander"] details{
  border:1px solid var(--line); border-radius:var(--r-md);
  background:rgba(255,255,255,.65);
}

/* ---- entrance -------------------------------------------------------- */
.tw-rise{ animation:twRise .6s var(--ease) both; }
@keyframes twRise{ from{ opacity:0; transform:translateY(14px); } to{ opacity:1; transform:none; } }

@media (max-width:640px){
  .tw-facts{ grid-template-columns:repeat(2,1fr); }
  .tw-dest__place{ right:80px; }
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{ animation-duration:.01ms !important; transition-duration:.01ms !important; }
  .tw-card:hover{ transform:none; }
}
</style>
"""

ICONS = {
    "compass": '<circle cx="12" cy="12" r="9"/><path d="m15.6 8.4-2 5.2-5.2 2 2-5.2z"/>',
    "sparkle": '<path d="M12 3v3.6M12 17.4V21M3 12h3.6M17.4 12H21M5.6 5.6l2.6 2.6M15.8 15.8l2.6 2.6M18.4 5.6l-2.6 2.6M8.2 15.8l-2.6 2.6"/><circle cx="12" cy="12" r="3"/>',
    "wallet": '<path d="M3 7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M16 11h5v4h-5a2 2 0 0 1 0-4z"/>',
    "sun": '<circle cx="12" cy="12" r="4.2"/><path d="M12 2v2.4M12 19.6V22M2 12h2.4M19.6 12H22M4.9 4.9l1.7 1.7M17.4 17.4l1.7 1.7M19.1 4.9l-1.7 1.7M6.6 17.4l-1.7 1.7"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3.5 9h17M3.5 15h17M12 3a15 15 0 0 1 0 18A15 15 0 0 1 12 3z"/>',
    "shield": '<path d="M12 3l7 3v6c0 4.2-2.9 7.6-7 9-4.1-1.4-7-4.8-7-9V6z"/><path d="m9.2 12 2 2 3.6-4"/>',
    "plane": '<path d="M2 13.4 21 4l-4.6 16.2-4.1-6.1z"/><path d="m12.3 14.1-3.4 6.9-1-5"/>',
    "pin": '<path d="M12 21s-7-5.6-7-11a7 7 0 1 1 14 0c0 5.4-7 11-7 11z"/><circle cx="12" cy="10" r="2.6"/>',
    "calendar": '<rect x="3.5" y="5" width="17" height="16" rx="2.5"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
    "tag": '<path d="M3 12.5V5a2 2 0 0 1 2-2h7.5L21 11.5 12.5 20z"/><circle cx="7.8" cy="7.8" r="1.4"/>',
    "bulb": '<path d="M9.2 17h5.6M10 21h4"/><path d="M12 3a6 6 0 0 1 3.6 10.8c-.5.4-.8 1-.8 1.6H9.2c0-.6-.3-1.2-.8-1.6A6 6 0 0 1 12 3z"/>',
    "layers": '<path d="M12 3 3 8l9 5 9-5z"/><path d="m3 13 9 5 9-5M3 18l9 5 9-5"/>',
}


def icon(name: str) -> str:
    return f'<svg viewBox="0 0 24 24">{ICONS.get(name, ICONS["sparkle"])}</svg>'


def card(icon_name: str, title: str, body: str) -> str:
    """One feature card, byte-for-byte the markup the real app injects."""
    return f"""
<div class="tw-card tw-feat tw-rise">
<div class="tw-feat__icon">{icon(icon_name)}</div>
<h3>{title}</h3>
<p>{body}</p>
</div>"""


def current_level() -> int:
    raw = st.query_params.get("level", "1")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else "1"
    try:
        return max(1, min(3, int(str(raw).strip())))
    except (TypeError, ValueError):
        return 1


LEVEL = current_level()

# ---------------------------------------------------------------- level 2
if LEVEL >= 2:
    st.markdown(STYLESHEET, unsafe_allow_html=True)

st.title(f"🔬 Layer test — level {LEVEL}")
st.caption({
    1: "Plain widgets only. Nothing from TripWise.",
    2: "Plus the TripWise stylesheet.",
    3: "Plus custom HTML cards.",
}[LEVEL])
st.markdown("Change the number at the end of the address: "
            "`?level=1`, `?level=2`, `?level=3`.")

st.divider()

# ------------------------------------------- the same test at every level
st.subheader("Button")
st.session_state.setdefault("count", 0)
if st.button("Press me", type="primary"):
    st.session_state.count += 1

st.header(f"Count: {st.session_state.count}")
if st.session_state.count:
    st.success(f"Working at level {LEVEL}.")
else:
    st.info("Press the button. The number should go up.")

st.divider()

st.subheader("Tabs")
one, two, three = st.tabs(["Tab one", "Tab two", "Tab three"])
with one:
    st.success("You are on TAB ONE")
with two:
    st.warning("You are on TAB TWO")
with three:
    st.error("You are on TAB THREE")

st.divider()

# ---------------------------------------------------------------- level 3
if LEVEL >= 3:
    st.subheader("Custom HTML cards")
    cols = st.columns(3, gap="medium")
    for col, (ic, title, body) in zip(cols, [
        ("compass", "First card", "Injected HTML, exactly as the real app does it."),
        ("wallet", "Second card", "If the button stopped counting, this is the cause."),
        ("sun", "Third card", "Each card is a single st.markdown call."),
    ]):
        with col:
            st.markdown(card(ic, title, body), unsafe_allow_html=True)
    st.divider()

st.caption("Report the highest level where the count still rises, and the first "
           "level where it stops. That identifies the layer at fault.")
