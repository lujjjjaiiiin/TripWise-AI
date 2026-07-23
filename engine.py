"""Recommendation engine and the travel intelligence layered on top of it.

The ranking is unchanged in spirit from the notebook: standardise the feature
matrix, place the traveller's answers in that space, sort by cosine similarity.
Everything else here turns a matched row into something a person can act on —
why it matched, when to go, what it costs, what to do when they arrive.

All of it degrades gracefully: any column the CSV happens not to carry is simply
omitted rather than raising.
"""

from __future__ import annotations

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
