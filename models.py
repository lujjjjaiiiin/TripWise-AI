"""Schema and tunables.

Every column name, label and magic number the backend depends on lives here, so
adapting TripWise to a differently-shaped catalogue is a matter of editing this
file rather than hunting through the pipeline.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# catalogue schema
# --------------------------------------------------------------------------- #

TASTES: list[tuple[str, str, str]] = [
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
TASTE_KEYS: list[str] = [k for k, _, _ in TASTES]
TASTE_LABEL: dict[str, str] = {k: lbl for k, lbl, _ in TASTES}

REGIONS: list[tuple[str, str]] = [
    ("region_africa", "Africa"),
    ("region_asia", "Asia"),
    ("region_europe", "Europe"),
    ("region_middle_east", "Middle East"),
    ("region_north_america", "North America"),
    ("region_oceania", "Oceania"),
    ("region_south_america", "South America"),
]
REGION_COLS: list[str] = [c for c, _ in REGIONS]
REGION_LABEL: dict[str, str] = dict(REGIONS)

NUMERIC_EXTRAS: list[str] = [
    "has_airport", "is_short_trip", "is_one_week", "temp_avg_yearly",
    "budget_level_encoded", "HotelRating_encoded", "rating_was_unknown",
]

FEATURE_COLS: list[str] = (
    ["latitude", "longitude"] + TASTE_KEYS + NUMERIC_EXTRAS + REGION_COLS
)

# the catalogue is unusable without these
REQUIRED_COLS: list[str] = [
    "city", "country", "latitude", "longitude",
    "temp_avg_yearly", "budget_level_encoded",
] + TASTE_KEYS

# present -> richer cards, absent -> those rows are simply omitted
OPTIONAL_COLS: list[str] = [
    "name", "iata", "icao", "HotelName", "Attractions", "Description",
    "latitude_airport", "longitude_airport",
]

# values the notebook writes in place of a missing string
NULL_TOKENS: frozenset[str] = frozenset(
    {"", "unknown", "not specified", "nan", "none", "n/a", "na", "-", "null"}
)

# --------------------------------------------------------------------------- #
# domain constants
# --------------------------------------------------------------------------- #

BUDGET_LABEL: dict[int, str] = {1: "Budget", 2: "Mid-range", 3: "Luxury"}
BUDGET_DAILY: dict[int, int] = {1: 55, 2: 130, 3: 290}

TASTE_MIN, TASTE_MAX = 1, 5
TEMP_MIN, TEMP_MAX = -20, 50
STARS_MIN, STARS_MAX = 1.0, 5.0
NIGHTS_MIN, NIGHTS_MAX = 1, 60
TRAVELLERS_MIN, TRAVELLERS_MAX = 1, 12

DEFAULT_TOP_N = 6

# --------------------------------------------------------------------------- #
# model tunables
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42

# k scales with catalogue size but stays in a range that produces readable groups
KMEANS_MIN_K, KMEANS_MAX_K = 3, 8
KMEANS_ROWS_PER_CLUSTER = 14

# at most this many results may come from one destination profile, so a
# shortlist cannot collapse into six variations of the same place
MAX_PER_CLUSTER = 2

# how far a fallback airport may be before it stops being useful
AIRPORT_MAX_KM = 450.0

CSV_NAME = "tripwise_data.csv"
