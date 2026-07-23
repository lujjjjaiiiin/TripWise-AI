"""
TripWise AI — an AI travel platform.

    streamlit run app.py

Python does the thinking: loading the catalogue, ranking destinations, deriving
seasons and costs and insights. Everything the visitor actually sees is custom
markup from `tripwise.ui`, styled by `tripwise.theme`. Streamlit is left with
two jobs — serving the page and collecting form input — and its own interface
is hidden.

    tripwise/theme.py   design tokens and every CSS rule
    tripwise/ui.py      HTML component builders
    tripwise/art.py     generated destination artwork
    tripwise/engine.py  ranking and travel intelligence
    tripwise/data.py    catalogue loading and validation
    tripwise/splash.py  the opening animation
"""

from __future__ import annotations

import os
import sys

# `streamlit run` does not guarantee the app's own folder is on the import path,
# so put it there before importing the package. Without this the app only starts
# when launched from its own directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components

from tripwise import data, engine, theme, ui
from tripwise.art import destination_art
from tripwise.splash import build as build_splash

st.set_page_config(
    page_title="TripWise AI — Intelligent travel planning",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SPLASH_CHROME = """
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
.stApp{background:#e3e9ef!important;}
iframe{position:fixed!important;inset:0!important;width:100vw!important;
       height:100vh!important;border:0!important;z-index:99999!important;}
</style>
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_catalogue():
    return data.load()


@st.cache_resource(show_spinner=False)
def get_recommender(_df, fingerprint: str):
    """fingerprint keeps the cache keyed to the catalogue, not the frame object."""
    return engine.Recommender(_df)


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
    name, code = engine.airport_of(row)
    airport = None
    if name and code:
        airport = f"{name} ({code})"
    elif name or code:
        airport = name or code

    tastes = engine.tastes_of(row)
    temp = float(row.get("temp_avg_yearly", 20) or 20)
    tier = int(float(row.get("budget_level_encoded", 2) or 2))

    return {
        "city": engine.clean(row.get("city")) or "Unknown",
        "country": engine.clean(row.get("country")) or "",
        "match": f"{float(row.get('match', 0)):.0f}",
        "art": destination_art(str(row.get("city", "x")), tastes, temp),
        "tier": engine.BUDGET_LABEL.get(min(3, max(1, tier)), "Mid-range"),
        "daily": engine.daily_cost(row),
        "temp": f"{temp:.0f}",
        "style": engine.trip_style(row),
        "region": engine.region_of(row),
        "why": engine.explain(row, prefs),
        "airport": airport,
        "season": engine.best_season(row),
        "climate": engine.climate_summary(row),
        "hotel": engine.clean(row.get("HotelName")),
        "attractions": engine.attractions_of(row),
        "tip": engine.travel_tip(row),
    }


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #

def page_home(df, is_real: bool, missing: list[str]) -> None:
    html(ui.navbar("home"))
    html(ui.hero(data.catalogue_stats(df)))

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

    html(ui.section_head(
        "How it works", "Three steps to a shortlist you trust",
        "No account, no browsing through hundreds of listings. Describe how you "
        "travel and the ranking does the narrowing.", anchor="features"))
    html('<div class="tw-grid tw-grid--3">'
         + ui.feature_card("compass", "Describe your trip",
                           "Nine travel dimensions, a budget tier and a climate target. "
                           "It takes about a minute and you can revise any of it.", 1)
         + ui.feature_card("sparkle", "See a ranked shortlist",
                           "Destinations are ordered by how closely they match what you "
                           "described, each one explaining its own reasoning.", 2)
         + ui.feature_card("wallet", "Plan around real numbers",
                           "Daily spend, best season, nearest airport and local tips — "
                           "enough to judge a trip before you commit to it.", 3)
         + "</div>" + ui.close_section())

    html(ui.section_head(
        "Capabilities", "Everything you need to travel smarter",
        "Built for travellers who want a decision, not a search results page."))
    html('<div class="tw-grid tw-grid--3">'
         + ui.feature_card("globe", "Global catalogue",
                           "Destinations across every inhabited region, each carrying "
                           "climate, budget and accommodation data.", 1)
         + ui.feature_card("wallet", "Costs before booking",
                           "Per-day estimates from budget tier and accommodation "
                           "standard, scaled to your party and trip length.", 2)
         + ui.feature_card("sun", "Season guidance",
                           "The months worth travelling in, inferred from latitude and "
                           "climate rather than guessed.", 3)
         + ui.feature_card("plane", "Nearest airport",
                           "The gateway airport and its code for every destination that "
                           "has one on record.", 4)
         + ui.feature_card("chart", "Readable insights",
                           "Plain-language readings of your results — climate fit, cost "
                           "spread and how your matches cluster.", 5)
         + ui.feature_card("shield", "Honest about gaps",
                           "When a detail is missing from the catalogue TripWise says so "
                           "instead of inventing a plausible answer.", 6)
         + "</div>" + ui.close_section())

    html(ui.section_head(
        "About", "Advanced recommendation technology, quietly",
        "You should not need to understand the machinery to trust the result.",
        anchor="about"))
    html(ui.about_block() + ui.close_section())

    html(ui.section_head("Contact", "Ready when you are", "", anchor="contact"))
    html(ui.contact_block() + ui.close_section())
    html(ui.footer())


def page_planner(df, is_real: bool) -> None:
    html(ui.navbar("planner"))
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
        for i, (key, label, hint) in enumerate(engine.TASTES):
            with cols[i % 3]:
                answers[key] = st.slider(label, 1, 5, 3, help=hint, key=f"t_{key}")

        html('<div class="tw-fieldset__title">Budget and climate</div>')
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            answers["budget"] = st.select_slider(
                "Budget level", options=[1, 2, 3], value=2,
                format_func=lambda v: engine.BUDGET_LABEL[v])
        with c2:
            answers["temp"] = st.slider("Preferred average temperature (°C)", -5, 40, 22)
        with c3:
            answers["stars"] = st.slider("Minimum accommodation standard", 1.0, 5.0, 4.0, 0.5)

        html('<div class="tw-fieldset__title">Where and how long</div>')
        c4, c5, c6 = st.columns(3, gap="large")
        with c4:
            region_names = ["Anywhere"] + [engine.REGION_LABEL[c] for c in engine.REGION_COLS]
            picked = st.selectbox("Preferred region", region_names)
            answers["region"] = next(
                (c for c in engine.REGION_COLS if engine.REGION_LABEL[c] == picked), None)
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
        html(ui.footer())
        return

    prefs = engine.build_preferences(df, answers)
    pool = df
    if answers["needs_airport"] and "has_airport" in df.columns:
        gated = df[df.apply(lambda r: any(engine.airport_of(r)), axis=1)]
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

    html(ui.section_head(
        "Your matches", "Where you should go",
        f"Ranked against your answers. Estimates assume {answers['travellers']} "
        f"traveller(s) over {answers['nights']} nights."))
    cards = "".join(
        ui.destination_card(card_payload(row, prefs), delay=(i % 6) + 1)
        for i, (_, row) in enumerate(results.iterrows())
    )
    html(f'<div class="tw-grid tw-grid--3">{cards}</div>' + ui.close_section())

    readings = engine.insights(results, prefs, answers)
    if readings:
        html(ui.section_head(
            "AI insights", "What your results say",
            "Read from the shortlist as a whole, not any single destination."))
        html('<div class="tw-grid tw-grid--2">'
             + "".join(ui.insight_card(t, b, (i % 6) + 1)
                       for i, (t, b) in enumerate(readings))
             + "</div>" + ui.close_section())

    render_charts(results, answers)
    html(ui.footer())


def render_charts(results, answers: dict) -> None:
    import plotly.express as px

    html(ui.section_head("Comparison", "Your shortlist side by side",
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
            daily=[engine.daily_cost(r) for _, r in results.iterrows()])
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

    html(ui.close_section())


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #

def main() -> None:
    params = st.query_params
    view = params.get("view", "home")

    # The splash plays only on a bare first load. Any navigation carries a query
    # param, so returning to the landing page never replays it.
    first_visit = len(params) == 0 and not st.session_state.get("seen_splash")
    if first_visit:
        st.session_state.seen_splash = True
        html(SPLASH_CHROME)
        components.html(build_splash(), height=760, scrolling=False)
        return

    html(theme.CSS)
    df, is_real, missing = load_catalogue()

    if view == "planner":
        page_planner(df, is_real)
    else:
        page_home(df, is_real, missing)


if __name__ == "__main__":
    main()
