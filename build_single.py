"""
TripWise AI — an AI travel platform.

    streamlit run app.py

This module is the application layer only: routing, caching, collecting form
input and assembling pages. It holds no ranking logic and no markup — those live
in the package, so this file stays small enough to read in one sitting.

    tripwise/config.py       schema, labels and tunables
    tripwise/errors.py       logging and failure containment
    tripwise/loader.py       catalogue loading, validation, coercion
    tripwise/airports.py     airport resolution with geographic fallback
    tripwise/models.py       fitted scaler and K-Means bundle
    tripwise/recommender.py  preference vector, cosine ranking, diversification
    tripwise/insights.py     cost, season, explanations, travel intelligence
    tripwise/validation.py   user input validation
    tripwise/theme.py        design tokens and every CSS rule
    tripwise/ui.py           HTML component builders
    tripwise/art.py          generated destination artwork
    tripwise/splash.py       the opening animation
"""

from __future__ import annotations

import os
import sys

# `streamlit run` does not guarantee the app's own folder is on the import path,
# so put it there before importing the package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from tripwise import config, insights, loader, models, recommender, theme, ui, validation
from tripwise.airports import AirportIndex, coverage
from tripwise.art import destination_art
from tripwise.errors import guard, to_float, to_int
from tripwise.splash import build as build_splash

st.set_page_config(
    page_title="TripWise AI — Intelligent travel planning",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
# cached resources
# --------------------------------------------------------------------------- #

def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_catalogue() -> tuple[pd.DataFrame, loader.CatalogueReport]:
    """Read and clean the catalogue. Cached on the data, not per session."""
    return loader.load()


@st.cache_resource(show_spinner=False)
def fitted_models(fingerprint: str) -> models.ModelBundle:
    """Scaler and K-Means, fitted once per catalogue.

    `fingerprint` keys the cache to the catalogue's shape rather than to the
    DataFrame, which is unhashable and changes identity on every rerun.
    """
    df, _ = load_catalogue()
    return models.fit(df)


@st.cache_resource(show_spinner=False)
def airport_index(fingerprint: str) -> AirportIndex:
    """Searchable airport index, built once per catalogue."""
    df, _ = load_catalogue()
    return AirportIndex(df)


def fingerprint(df: pd.DataFrame) -> str:
    """A cheap, stable identity for a catalogue."""
    return f"{len(df)}x{len(df.columns)}:{hash(tuple(sorted(df.columns)))}"


# --------------------------------------------------------------------------- #
# view models
# --------------------------------------------------------------------------- #

def card_payload(row, prefs: dict, airports: AirportIndex) -> dict:
    """Flatten a matched row into exactly what the card template needs."""
    airport = airports.resolve(row)
    tastes = insights.tastes_of(row)
    temp = to_float(row.get("temp_avg_yearly", 20), 20.0)
    tier = to_int(row.get("budget_level_encoded", 2), 2, 1, 3)

    style = insights.trip_style(row)
    profile = row.get("profile")
    if profile and profile not in style:
        style = [profile] + style[:2]

    hotel = row.get("HotelName")
    if hotel is not None and pd.isna(hotel):
        hotel = None

    return {
        "city": str(row.get("city", "Unknown")),
        "country": str(row.get("country", "") or ""),
        "match": f"{to_float(row.get('match', 0), 0.0):.0f}",
        "art": destination_art(str(row.get("city", "x")), tastes, temp),
        "tier": config.BUDGET_LABEL[tier],
        "daily": insights.daily_cost(row),
        "temp": f"{temp:.0f}",
        "style": style,
        "region": insights.region_of(row),
        "why": insights.explain(row, prefs),
        "airport": airport.label if airport else None,
        "season": insights.best_season(row),
        "climate": insights.climate_summary(row),
        "hotel": hotel,
        "attractions": insights.attractions_of(row),
        "tip": insights.travel_tip(row),
    }


def data_notice(report: loader.CatalogueReport, airports: AirportIndex,
                df: pd.DataFrame) -> None:
    """Tell the operator what the catalogue can and cannot support."""
    if not report.is_real:
        detail = report.error or (
            f"Missing required columns: {', '.join(report.missing_required)}."
            if report.missing_required
            else f"{config.CSV_NAME} was not found beside app.py."
        )
        st.info(f"Running on the built-in demo catalogue. {detail} "
                "Export final_df from the notebook to use your own data.", icon="ℹ️")
        return

    if not len(airports):
        st.warning(
            "No airport data found in the catalogue, so no airports can be shown. "
            "Re-export final_df including the `name` and `iata` columns.",
            icon="⚠️",
        )
        return

    direct, total = coverage(df, airports)
    if total and direct < total:
        st.caption(
            f"Catalogue: {report.rows_out:,} destinations from {report.source}. "
            f"{direct:,} carry a matched airport; the rest resolve to their nearest "
            f"airport from the {len(airports):,} in the dataset."
        )


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #

def page_home(df: pd.DataFrame, report: loader.CatalogueReport,
              airports: AirportIndex) -> None:
    html(ui.navbar("home"))
    html(ui.hero(loader.stats(df)))
    data_notice(report, airports, df)

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
         + ui.feature_card("plane", "Nearest airport, always",
                           "Every destination resolves to a gateway airport — by direct "
                           "match where one exists, by distance otherwise.", 4)
         + ui.feature_card("layers", "Genuinely different options",
                           "Results are spread across distinct destination profiles, so "
                           "a shortlist offers real alternatives.", 5)
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


def collect_answers() -> tuple[dict, bool]:
    """Render the planner form and return the raw answers plus submit state."""
    with st.form("planner", border=False):
        html('<div class="tw-panel">')
        html('<div class="tw-fieldset__title">What you care about</div>')

        answers: dict = {}
        cols = st.columns(3, gap="large")
        for i, (key, label, hint) in enumerate(config.TASTES):
            with cols[i % 3]:
                answers[key] = st.slider(label, config.TASTE_MIN, config.TASTE_MAX, 3,
                                         help=hint, key=f"t_{key}")

        html('<div class="tw-fieldset__title">Budget and climate</div>')
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            answers["budget"] = st.select_slider(
                "Budget level", options=[1, 2, 3], value=2,
                format_func=lambda v: config.BUDGET_LABEL[v])
        with c2:
            answers["temp"] = st.slider("Preferred average temperature (°C)", -5, 40, 22)
        with c3:
            answers["stars"] = st.slider("Minimum accommodation standard",
                                         config.STARS_MIN, config.STARS_MAX, 4.0, 0.5)

        html('<div class="tw-fieldset__title">Where and how long</div>')
        c4, c5, c6 = st.columns(3, gap="large")
        with c4:
            names = ["Anywhere"] + [config.REGION_LABEL[c] for c in config.REGION_COLS]
            picked = st.selectbox("Preferred region", names)
            answers["region"] = next(
                (c for c in config.REGION_COLS if config.REGION_LABEL[c] == picked), None)
        with c5:
            length = st.radio("Trip length", ["Short trip", "One week"], horizontal=True)
            answers["trip_length"] = "short" if length == "Short trip" else "week"
        with c6:
            answers["travellers"] = st.number_input(
                "Travellers", config.TRAVELLERS_MIN, config.TRAVELLERS_MAX, 2)

        answers["nights"] = st.slider("Nights", config.NIGHTS_MIN, 30,
                                      4 if answers["trip_length"] == "short" else 7)
        answers["needs_airport"] = st.checkbox(
            "Prefer destinations with a nearby airport", value=True)

        html("</div>")
        st.markdown("<div style='height:1.3rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Find my destinations  →", type="primary")

    return answers, submitted


def page_planner(df: pd.DataFrame, report: loader.CatalogueReport,
                 airports: AirportIndex) -> None:
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

    raw_answers, submitted = collect_answers()
    if not submitted:
        html(ui.footer())
        return

    blocking = validation.validate_catalogue(df)
    if blocking:
        st.error(blocking, icon="⚠️")
        html(ui.footer())
        return

    checked = validation.validate_answers(raw_answers)
    for notice in checked.notices:
        st.warning(notice, icon="⚠️")
    answers = checked.answers

    pool, pool_notices = validation.validate_pool(df, answers)
    for notice in pool_notices:
        st.info(notice, icon="ℹ️")

    result = None
    with guard("recommendation") as failure:
        bundle = fitted_models(fingerprint(df))
        prefs = recommender.build_preference_vector(df, answers)
        with st.spinner("Ranking destinations…"):
            result = recommender.recommend(
                df, bundle, prefs,
                top_n=config.DEFAULT_TOP_N,
                pool_index=pool.index.to_numpy(),
            )
    if failure or result is None:
        st.error("Something went wrong while ranking destinations. "
                 "Adjust a preference and try again.", icon="⚠️")
        html(ui.footer())
        return

    if result.empty:
        st.warning("No destinations matched. Try widening the region or climate.",
                   icon="ℹ️")
        html(ui.footer())
        return

    render_results(result, prefs, answers, airports)


def render_results(result: recommender.Recommendation, prefs: dict,
                   answers: dict, airports: AirportIndex) -> None:
    frame = result.frame

    html(ui.section_head(
        "Your matches", "Where you should go",
        f"Ranked against your answers from {result.considered:,} destinations. "
        f"Estimates assume {answers['travellers']} traveller(s) over "
        f"{answers['nights']} nights."))

    cards = []
    for i, (_, row) in enumerate(frame.iterrows()):
        with guard(f"card for {row.get('city')}"):
            cards.append(ui.destination_card(card_payload(row, prefs, airports),
                                             delay=(i % 6) + 1))
    html(f'<div class="tw-grid tw-grid--3">{"".join(cards)}</div>' + ui.close_section())

    readings: list = []
    with guard("insights"):
        readings = insights.insights(frame, prefs, answers, result.profiles)
    if readings:
        html(ui.section_head(
            "AI insights", "What your results say",
            "Read from the shortlist as a whole, not any single destination."))
        html('<div class="tw-grid tw-grid--2">'
             + "".join(ui.insight_card(t, b, (i % 6) + 1)
                       for i, (t, b) in enumerate(readings))
             + "</div>" + ui.close_section())

    with guard("charts") as failed:
        render_charts(frame, answers)
    if failed:
        st.caption("Charts are unavailable for this result set.")

    html(ui.footer())


def chart(fig) -> None:
    """Render a figure full width across Streamlit versions."""
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def render_charts(frame: pd.DataFrame, answers: dict) -> None:
    import plotly.express as px

    html(ui.section_head("Comparison", "Your shortlist side by side",
                         "The same destinations, measured two ways."))
    left, right = st.columns(2, gap="large")
    axis = dict(showgrid=True, gridcolor="rgba(9,16,32,.07)", zeroline=False,
                tickfont=dict(color="#55637A", size=12),
                title_font=dict(color="#55637A", size=12))
    layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="Inter, sans-serif", color="#080D18", size=13),
                  margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False)
    scale = ["#BAE6FD", "#0EA5E9", "#2563EB"]

    with left:
        html('<div class="tw-card tw-chart"><h4>Match strength</h4>'
             '<p>How closely each destination sits to your answers.</p>')
        fig = px.bar(frame.sort_values("match"), x="match", y="city", orientation="h",
                     color="match", color_continuous_scale=scale)
        fig.update_layout(**layout, xaxis=dict(**axis, title="Match %"),
                          yaxis=dict(**axis, title=""), coloraxis_showscale=False)
        fig.update_traces(hovertemplate="%{y}: %{x:.0f}%<extra></extra>")
        chart(fig)
        html("</div>")

    with right:
        costs = frame.assign(daily=[insights.daily_cost(r) for _, r in frame.iterrows()])
        costs["total"] = costs["daily"] * answers["nights"] * answers["travellers"]
        html('<div class="tw-card tw-chart"><h4>Estimated trip cost</h4>'
             f'<p>{answers["travellers"]} traveller(s), {answers["nights"]} nights, '
             'before flights.</p>')
        fig = px.bar(costs.sort_values("total"), x="total", y="city", orientation="h",
                     color="total", color_continuous_scale=scale)
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

    html(theme.CSS)

    df = report = airports = None
    with guard("startup") as failure:
        df, report = load_catalogue()
        airports = airport_index(fingerprint(df))
    if failure or df is None:
        st.error("TripWise could not start: the destination catalogue failed to load. "
                 "Check that tripwise_data.csv sits beside app.py.", icon="⚠️")
        return

    # The splash paints over the page rendering beneath it in this same run, then
    # fades itself out — no blank frame, no blocking sleep, no second rerun. It
    # plays only on a bare first load; every link carries a query param.
    if len(params) == 0 and not st.session_state.get("seen_splash", False):
        st.session_state["seen_splash"] = True
        html(build_splash())

    if view == "planner":
        page_planner(df, report, airports)
    else:
        page_home(df, report, airports)


if __name__ == "__main__":
    main()
