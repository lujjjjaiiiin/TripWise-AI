"""
TripWise AI — an AI travel platform (single-file build).

    streamlit run app.py

Generated from the modular project by build_single.py: identical code with the
package inlined, so it deploys without a folder alongside it. Banners mark the
module each section came from; edit the modules, not this file.

Every control is a real Streamlit widget. Custom HTML is used only to display
cards and headings, never to build interaction.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
import re
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from html import escape
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="TripWise AI — Intelligent travel planning",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================================================
# CONFIGURATION — Schema and tunables.
# ==========================================================================

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

# Bumping this invalidates every cached resource. Streamlit keeps cached objects
# across a hot reload, so an instance built by a previous version of a class can
# survive into new code that expects fields it does not have. Any change to a
# cached return type must bump this.
CACHE_VERSION = "5.0.0"

# Shown in the app so the running build can be identified from a screenshot.
BUILD = "5.0.0"

# ==========================================================================
# ERROR HANDLING — Logging and failure containment.
# ==========================================================================

_CONFIGURED = False


def get_logger(name: str = "tripwise") -> logging.Logger:
    """Module logger, configured once for the process."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED = True
    return logger


log = get_logger()

T = TypeVar("T")


def safe(default: T, label: str = "") -> Callable:
    """Decorator: log and return `default` instead of raising.

    Used on the per-row derivations that feed the cards, where one malformed
    value should cost a single detail rather than the whole result set.
    """
    def outer(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> T:
            try:
                return fn(*args, **kwargs)
            except Exception:
                log.warning("%s failed, using fallback", label or fn.__name__, exc_info=True)
                return default
        return inner
    return outer


@contextmanager
def guard(label: str):
    """Context manager for a whole page section.

    Yields a one-element list; it stays empty on success and holds the exception
    on failure, letting the caller render an apology in place of the section.
    """
    box: list[Exception] = []
    try:
        yield box
    except Exception as exc:                      # noqa: BLE001 - deliberate boundary
        log.error("%s failed", label, exc_info=True)
        box.append(exc)


def to_float(value: Any, default: float = 0.0) -> float:
    """Coerce anything to a float without raising."""
    try:
        if value is None:
            return default
        out = float(value)
        return default if out != out else out     # reject NaN
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0, lo: int | None = None, hi: int | None = None) -> int:
    """Coerce to int, optionally clamped to a range."""
    out = int(round(to_float(value, default)))
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out

# ==========================================================================
# AIRPORT RESOLUTION — Airport resolution.
# ==========================================================================

EARTH_RADIUS_KM = 6371.0088


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
    return None if text.lower() in NULL_TOKENS else text


def normalise(text: str | None) -> str:
    """Fold a place name to a comparable key.

    Strips accents, case and punctuation, so ``Al-'Ula``, ``AlUla`` and
    ``al ula`` all collapse to ``alula``.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", ascii_only.lower())


@dataclass(frozen=True)
class Airport:
    """One resolved airport, and how it was arrived at."""
    name: str | None
    code: str | None
    distance_km: float | None
    how: str = "direct"                      # direct | name | distance

    @property
    def exact(self) -> bool:
        return self.how in ("direct", "name")

    @property
    def label(self) -> str:
        """Display string, honest about how far away the airport is."""
        if self.name and self.code:
            base = f"{self.name} ({self.code})"
        else:
            base = self.name or self.code or "Unnamed airport"
        if self.exact or not self.distance_km:
            return base
        if self.distance_km <= AIRPORT_MAX_KM:
            return f"{base} — about {self.distance_km:,.0f} km away"
        return f"{base} — {self.distance_km:,.0f} km away, closest in the catalogue"

    def __bool__(self) -> bool:
        return bool(self.name or self.code)


NO_AIRPORT = Airport(None, None, None, "none")


def _haversine(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from one point to many, vectorised."""
    p1, p2 = np.radians(lat), np.radians(lats)
    dphi = p2 - p1
    dlambda = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class AirportIndex:
    """Every airport the catalogue knows, searchable by name or position.

    Built once per catalogue and cached, because it scans the whole frame.
    """

    def __init__(self, df: pd.DataFrame):
        self._lats = np.empty(0)
        self._lons = np.empty(0)
        self._names: list[str | None] = []
        self._codes: list[str | None] = []
        self._by_city: dict[str, int] = {}
        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        if "name" not in df.columns and "iata" not in df.columns:
            log.warning("catalogue carries no airport columns; airports unavailable")
            return

        # airport coordinates land in *_airport when the merge collides with the
        # destination's own lat/lon; otherwise the destination position is a
        # fair proxy for a city's own airport
        lat_col = "latitude_airport" if "latitude_airport" in df.columns else "latitude"
        lon_col = "longitude_airport" if "longitude_airport" in df.columns else "longitude"

        seen: set[tuple[str | None, str | None]] = set()
        lats: list[float] = []
        lons: list[float] = []

        for row in df.itertuples(index=False):
            name = clean(getattr(row, "name", None))
            code = clean(getattr(row, "iata", None)) or clean(getattr(row, "icao", None))
            if not (name or code):
                continue

            lat = to_float(getattr(row, lat_col, None), float("nan"))
            lon = to_float(getattr(row, lon_col, None), float("nan"))
            if lat != lat or lon != lon or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                continue

            key = (name, code)
            if key in seen:
                continue
            seen.add(key)

            position = len(self._names)
            self._names.append(name)
            self._codes.append(code)
            lats.append(lat)
            lons.append(lon)

            # remember which city this airport served, for the name retry
            for token in (normalise(getattr(row, "city", None)), normalise(name)):
                if token and token not in self._by_city:
                    self._by_city[token] = position

        self._lats = np.asarray(lats, dtype=float)
        self._lons = np.asarray(lons, dtype=float)
        log.info("airport index built: %d airports, %d name keys",
                 len(self._names), len(self._by_city))

    def __len__(self) -> int:
        return len(self._names)

    def _at(self, i: int, distance: float | None, how: str) -> Airport:
        return Airport(self._names[i], self._codes[i], distance, how)

    @safe(NO_AIRPORT, "airport name lookup")
    def by_name(self, city: str | None) -> Airport:
        """Recover a match the exact string join missed."""
        i = self._by_city.get(normalise(city))
        return self._at(i, 0.0, "name") if i is not None else NO_AIRPORT

    @safe(NO_AIRPORT, "nearest airport lookup")
    def nearest(self, lat: float, lon: float) -> Airport:
        """Closest indexed airport to a coordinate.

        Always answers when the index holds anything; the distance carries the
        caveat rather than a silent refusal.
        """
        if not len(self) or lat != lat or lon != lon:
            return NO_AIRPORT
        distances = _haversine(lat, lon, self._lats, self._lons)
        i = int(np.argmin(distances))
        km = float(distances[i])
        if km > AIRPORT_MAX_KM:
            log.debug("nearest airport is %.0f km away — sparse airport coverage", km)
        return self._at(i, km, "distance")

    @safe(NO_AIRPORT, "airport resolution")
    def resolve(self, row) -> Airport:
        """The airport for one destination, via the three-step ladder."""
        name = clean(row.get("name"))
        code = clean(row.get("iata")) or clean(row.get("icao"))
        if name or code:
            return Airport(name, code, 0.0, "direct")

        found = self.by_name(row.get("city"))
        if found:
            return found

        return self.nearest(
            to_float(row.get("latitude"), float("nan")),
            to_float(row.get("longitude"), float("nan")),
        )


def coverage(df: pd.DataFrame, index: AirportIndex) -> tuple[int, int]:
    """(rows resolved by direct join, rows total) — for the data-health notice."""
    if not len(df):
        return 0, 0
    direct = sum(
        1 for _, r in df.iterrows()
        if clean(r.get("name")) or clean(r.get("iata")) or clean(r.get("icao"))
    )
    return direct, len(df)

# ==========================================================================
# CATALOGUE — Catalogue loading and validation.
# ==========================================================================

# city, country, region, iata, airport name, lat, lon,
# cul adv nat bea nig cui wel urb sec, budget, temp, stars
_DEMO: list[tuple] = [
    ("Paris", "France", "region_europe", "CDG", "Charles de Gaulle", 48.86, 2.35, 5,2,2,1,4,5,3,5,1, 3, 12.3, 4.2),
    ("Barcelona", "Spain", "region_europe", "BCN", "Barcelona El Prat", 41.39, 2.17, 4,3,2,4,5,5,3,4,1, 2, 17.1, 4.0),
    ("Reykjavik", "Iceland", "region_europe", "KEF", "Keflavik International", 64.15, -21.94, 3,5,5,1,3,3,4,2,5, 3, 5.2, 4.0),
    ("Santorini", "Greece", "region_europe", "JTR", "Santorini National", 36.39, 25.46, 3,2,3,5,3,4,5,2,4, 3, 18.4, 4.4),
    ("Vienna", "Austria", "region_europe", "VIE", "Vienna International", 48.21, 16.37, 5,2,3,1,3,4,4,4,2, 3, 11.0, 4.3),
    ("Lisbon", "Portugal", "region_europe", "LIS", "Humberto Delgado", 38.72, -9.14, 4,3,3,4,4,4,3,4,2, 2, 17.5, 4.1),
    ("Tokyo", "Japan", "region_asia", "HND", "Haneda", 35.68, 139.69, 5,2,2,1,5,5,3,5,1, 3, 16.0, 4.3),
    ("Bali", "Indonesia", "region_asia", "DPS", "Ngurah Rai International", -8.41, 115.19, 4,4,5,5,4,4,5,2,4, 1, 26.6, 4.1),
    ("Bangkok", "Thailand", "region_asia", "BKK", "Suvarnabhumi", 13.76, 100.50, 4,3,2,2,5,5,4,5,1, 1, 28.4, 4.0),
    ("Maldives", "Maldives", "region_asia", "MLE", "Velana International", 4.18, 73.51, 1,3,4,5,1,3,5,1,5, 3, 28.1, 4.6),
    ("Kyoto", "Japan", "region_asia", "KIX", "Kansai International", 35.01, 135.77, 5,2,4,1,2,5,4,2,3, 3, 15.9, 4.3),
    ("Kathmandu", "Nepal", "region_asia", "KTM", "Tribhuvan International", 27.72, 85.32, 5,5,5,1,2,3,3,2,4, 1, 18.5, 3.5),
    ("Dubai", "UAE", "region_middle_east", "DXB", "Dubai International", 25.20, 55.27, 2,4,1,4,4,4,4,5,1, 3, 28.0, 4.5),
    ("AlUla", "Saudi Arabia", "region_middle_east", "ULH", "Prince Abdul Majeed", 26.61, 37.92, 5,5,5,1,1,3,4,1,5, 3, 26.0, 4.4),
    ("Istanbul", "Turkey", "region_middle_east", "IST", "Istanbul Airport", 41.01, 28.98, 5,2,2,2,4,5,3,5,1, 2, 15.0, 4.1),
    ("Muscat", "Oman", "region_middle_east", "MCT", "Muscat International", 23.59, 58.41, 4,4,4,4,1,4,4,2,4, 2, 28.3, 4.2),
    ("Marrakesh", "Morocco", "region_africa", "RAK", "Marrakesh Menara", 31.63, -7.99, 5,4,3,1,3,5,4,3,2, 2, 20.3, 4.0),
    ("Cape Town", "South Africa", "region_africa", "CPT", "Cape Town International", -33.92, 18.42, 4,5,5,5,4,5,3,4,3, 2, 17.0, 4.2),
    ("Zanzibar", "Tanzania", "region_africa", "ZNZ", "Abeid Amani Karume", -6.16, 39.20, 3,4,4,5,2,3,4,1,4, 2, 26.4, 4.0),
    ("Cairo", "Egypt", "region_africa", "CAI", "Cairo International", 30.04, 31.24, 5,3,1,1,3,4,2,4,1, 1, 22.1, 3.7),
    ("New York", "United States", "region_north_america", "JFK", "John F. Kennedy", 40.71, -74.01, 5,2,2,2,5,5,3,5,1, 3, 12.9, 4.1),
    ("Banff", "Canada", "region_north_america", "YYC", "Calgary International", 51.18, -115.57, 2,5,5,1,2,3,4,1,5, 3, 3.0, 4.3),
    ("Mexico City", "Mexico", "region_north_america", "MEX", "Benito Juarez", 19.43, -99.13, 5,3,3,1,5,5,3,5,1, 2, 17.0, 4.0),
    ("Vancouver", "Canada", "region_north_america", "YVR", "Vancouver International", 49.28, -123.12, 3,5,5,3,3,4,4,4,3, 3, 11.0, 4.2),
    ("Rio de Janeiro", "Brazil", "region_south_america", "GIG", "Galeao", -22.91, -43.17, 3,4,4,5,5,4,3,4,2, 2, 23.8, 4.0),
    ("Cusco", "Peru", "region_south_america", "CUZ", "Alejandro Velasco Astete", -13.53, -71.97, 5,5,5,1,3,4,3,2,4, 1, 12.3, 3.9),
    ("Patagonia", "Chile", "region_south_america", "PNT", "Teniente Julio Gallardo", -51.73, -72.51, 2,5,5,1,1,3,3,1,5, 3, 6.5, 4.0),
    ("Cartagena", "Colombia", "region_south_america", "CTG", "Rafael Nunez", 10.39, -75.51, 4,3,3,5,4,4,4,3,3, 2, 27.8, 4.1),
    ("Sydney", "Australia", "region_oceania", "SYD", "Kingsford Smith", -33.87, 151.21, 3,4,4,5,4,5,4,5,2, 3, 18.3, 4.2),
    ("Queenstown", "New Zealand", "region_oceania", "ZQN", "Queenstown Airport", -45.03, 168.66, 2,5,5,2,3,4,4,2,4, 3, 10.6, 4.3),
    ("Fiji", "Fiji", "region_oceania", "NAN", "Nadi International", -17.71, 177.44, 2,4,5,5,2,3,5,1,5, 3, 25.4, 4.3),
]


# Exports rename things. Each canonical column is accepted under any of these
# spellings, so a catalogue that calls its airport column "airport_name" or
# "IATA" still resolves instead of silently losing the feature.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name", "airport_name", "airportname", "airport", "Airport",
             "AirportName", "name_airport", "nearest_airport"),
    "iata": ("iata", "iata_code", "IATA", "IATA_code", "iatacode",
             "airport_code", "code"),
    "icao": ("icao", "icao_code", "ICAO", "icaocode"),
    "city": ("city", "City", "cityName", "city_name", "destination"),
    "country": ("country", "Country", "countyName", "country_name"),
    "HotelName": ("HotelName", "hotel_name", "hotelname", "hotel"),
    "Attractions": ("Attractions", "attractions", "nearby_attractions"),
    "latitude_airport": ("latitude_airport", "airport_latitude", "lat_airport"),
    "longitude_airport": ("longitude_airport", "airport_longitude", "lon_airport"),
}


def _apply_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rename recognised alternative spellings to the canonical column names."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    renames: dict[str, str] = {}
    found: list[str] = []

    for canonical, options in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for option in options:
            actual = lowered.get(option.lower())
            if actual and actual not in renames:
                renames[actual] = canonical
                found.append(f"{actual} -> {canonical}")
                break

    return (df.rename(columns=renames), found) if renames else (df, found)


@dataclass
class CatalogueReport:
    """What happened during load — surfaced in the UI, not swallowed."""
    is_real: bool = True
    source: str = ""
    rows_in: int = 0
    rows_out: int = 0
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    coerced: list[str] = field(default_factory=list)
    aliased: list[str] = field(default_factory=list)
    columns_seen: list[str] = field(default_factory=list)
    dropped_rows: int = 0
    error: str = ""

    @property
    def healthy(self) -> bool:
        return self.is_real and not self.missing_required and not self.error

    def get(self, field_name: str, default=None):
        """Read a field that a cached older instance may predate.

        Streamlit keeps cached objects across hot reloads, so a report built by
        a previous version of this class can outlive it. Readers use this rather
        than attribute access so a stale instance degrades instead of raising.
        """
        return getattr(self, field_name, default)


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
        row.update({c: (1 if c == region else 0) for c in REGION_COLS})
        rows.append(row)
    return pd.DataFrame(rows)


def _coerce(df: pd.DataFrame, report: CatalogueReport) -> pd.DataFrame:
    """Force every modelling column to a sane number, recording what changed."""
    df = df.copy()

    for col in TASTE_KEYS:
        before = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(3).clip(TASTE_MIN, TASTE_MAX)
        if not before.equals(df[col]):
            report.coerced.append(col)

    for col, lo, hi, fill in (
        ("latitude", -90, 90, 0.0),
        ("longitude", -180, 180, 0.0),
        ("temp_avg_yearly", TEMP_MIN, TEMP_MAX, 20.0),
        ("budget_level_encoded", 1, 3, 2),
        ("HotelRating_encoded", 1, 5, 3),
    ):
        if col not in df.columns:
            df[col] = fill
            continue
        before = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill).clip(lo, hi)
        if not before.equals(df[col]):
            report.coerced.append(col)

    for col in NUMERIC_EXTRAS + REGION_COLS:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in OPTIONAL_COLS:
        if col not in df.columns:
            report.missing_optional.append(col)

    # a row with no name or no position cannot be shown or mapped
    before = len(df)
    df = df[df["city"].notna() & (df["city"].astype(str).str.strip() != "")]
    df = df.dropna(subset=["latitude", "longitude"])
    report.dropped_rows = before - len(df)

    if "country" not in df.columns:
        df["country"] = ""
    df["country"] = df["country"].fillna("")

    # one row per destination; the notebook emits one row per hotel
    subset = [c for c in ("city", "country") if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")

    return df.reset_index(drop=True)


def load_catalogue_file(path: str | None = None) -> tuple[pd.DataFrame, CatalogueReport]:
    """Load, validate and clean the catalogue.

    Falls back to a built-in demo catalogue rather than failing, so the app is
    always usable and the report says which one is in play.
    """
    report = CatalogueReport()

    candidates = [path] if path else []
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates += [os.path.join(here, CSV_NAME), CSV_NAME]

    found = next((p for p in candidates if p and os.path.exists(p)), None)
    if not found:
        log.warning("%s not found; using the demo catalogue", CSV_NAME)
        df = _coerce(_demo_frame(), report)
        report.is_real, report.source = False, "built-in demo catalogue"
        report.rows_in = report.rows_out = len(df)
        return df, report

    try:
        raw = pd.read_csv(found)
    except Exception as exc:                        # noqa: BLE001 - user file
        log.error("could not read %s", found, exc_info=True)
        df = _coerce(_demo_frame(), report)
        report.is_real, report.source = False, "built-in demo catalogue"
        report.error = f"{CSV_NAME} could not be read ({exc.__class__.__name__})."
        report.rows_in = report.rows_out = len(df)
        return df, report

    report.rows_in = len(raw)
    report.columns_seen = [str(c) for c in raw.columns]
    raw, aliased = _apply_aliases(raw)
    report.aliased = aliased
    if aliased:
        log.info("column aliases applied: %s", ", ".join(aliased))

    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        log.error("%s is missing required columns: %s", CSV_NAME, missing)
        df = _coerce(_demo_frame(), report)
        report.is_real, report.source = False, "built-in demo catalogue"
        report.missing_required = missing
        report.rows_out = len(df)
        return df, report

    df = _coerce(raw, report)
    report.source, report.rows_out = os.path.basename(found), len(df)
    log.info("catalogue loaded: %d rows from %s (%d dropped)",
             report.rows_out, report.source, report.dropped_rows)
    return df, report


def feature_frame(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The numeric matrix the models train on, and the columns it came from."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].to_numpy(dtype=float), cols


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
# MODELS — The fitted model bundle.
# ==========================================================================

def choose_k(n_rows: int) -> int:
    """Enough clusters to be discriminating, few enough to stay readable."""
    if n_rows < KMEANS_MIN_K * 2:
        return max(1, min(KMEANS_MIN_K, n_rows))
    k = n_rows // KMEANS_ROWS_PER_CLUSTER
    return int(np.clip(k, KMEANS_MIN_K, KMEANS_MAX_K))


def _name_cluster(profile: dict[str, float], mean: dict[str, float]) -> str:
    """Label a cluster by the tastes it over-indexes on versus the catalogue."""
    lifts = sorted(
        ((k, profile[k] - mean.get(k, 3.0)) for k in TASTE_KEYS),
        key=lambda kv: -kv[1],
    )
    top = [TASTE_LABEL[k] for k, lift in lifts[:2] if lift > 0.35]
    if not top:
        return "All-rounder"
    return " & ".join(top)


@dataclass
class ModelBundle:
    """Everything fitted against one catalogue."""
    columns: list[str]
    scaler: StandardScaler
    matrix: np.ndarray                     # scaled catalogue, rows aligned to df
    kmeans: KMeans | None
    labels: np.ndarray                     # cluster id per row
    cluster_names: dict[int, str]
    norms: np.ndarray                      # row norms, precomputed for scoring

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_names)

    def transform(self, values: dict) -> np.ndarray:
        """Put one preference dict into the same space as the catalogue."""
        row = pd.DataFrame([{c: values.get(c, 0) for c in self.columns}])[self.columns]
        return self.scaler.transform(row.to_numpy(dtype=float))

    def cluster_of(self, index: int) -> int:
        return int(self.labels[index]) if index < len(self.labels) else -1

    def cluster_name(self, index: int) -> str:
        return self.cluster_names.get(self.cluster_of(index), "All-rounder")


def fit_models(df: pd.DataFrame) -> ModelBundle:
    """Fit the scaler and the clustering against a catalogue."""
    raw, columns = feature_frame(df)
    if raw.size == 0:
        raise ValueError("catalogue has no usable feature columns")

    scaler = StandardScaler()
    matrix = scaler.fit_transform(raw)

    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-9

    k = choose_k(len(df))
    kmeans: KMeans | None = None
    labels = np.zeros(len(df), dtype=int)
    names: dict[int, str] = {0: "All-rounder"}

    if k > 1:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(matrix)

        # name each cluster from its members' real (unscaled) taste averages
        catalogue_mean = {c: float(df[c].mean()) for c in TASTE_KEYS if c in df.columns}
        names = {}
        for cid in range(k):
            members = df.loc[labels == cid]
            if members.empty:
                names[cid] = "All-rounder"
                continue
            profile = {c: float(members[c].mean()) for c in TASTE_KEYS if c in members.columns}
            names[cid] = _name_cluster(profile, catalogue_mean)

        # disambiguate collisions so two clusters never share a name
        seen: dict[str, int] = {}
        for cid, label in list(names.items()):
            if label in seen:
                seen[label] += 1
                names[cid] = f"{label} {seen[label]}"
            else:
                seen[label] = 1

    log.info("models fitted: %d features, %d clusters (%s)",
             len(columns), max(1, k), ", ".join(sorted(set(names.values()))))

    return ModelBundle(
        columns=columns, scaler=scaler, matrix=matrix, kmeans=kmeans,
        labels=labels, cluster_names=names, norms=norms,
    )

# ==========================================================================
# RECOMMENDER — The recommendation pipeline.
# ==========================================================================

@dataclass
class Recommendation:
    """The ranked result plus how it was arrived at."""
    frame: pd.DataFrame
    profiles: list[str] = field(default_factory=list)
    considered: int = 0
    diversified: bool = False

    @property
    def empty(self) -> bool:
        return self.frame.empty


def build_preference_vector(df: pd.DataFrame, answers: dict) -> dict:
    """Translate validated answers into the model's feature space.

    Coordinates are set to the catalogue centroid rather than zero: zero is a
    real place in the Atlantic, and after scaling it would drag every result
    towards whatever is nearest to it.
    """
    prefs = {c: 0.0 for c in FEATURE_COLS}

    for key in TASTE_KEYS:
        prefs[key] = to_float(answers.get(key, 3), 3.0)

    prefs["latitude"] = float(df["latitude"].mean()) if "latitude" in df else 0.0
    prefs["longitude"] = float(df["longitude"].mean()) if "longitude" in df else 0.0
    prefs["temp_avg_yearly"] = to_float(answers.get("temp", 22), 22.0)
    prefs["budget_level_encoded"] = to_float(answers.get("budget", 2), 2.0)
    prefs["HotelRating_encoded"] = to_float(answers.get("stars", 4), 4.0)
    prefs["rating_was_unknown"] = 0.0
    prefs["has_airport"] = 1.0 if answers.get("needs_airport", True) else 0.0
    prefs["is_short_trip"] = 1.0 if answers.get("trip_length") == "short" else 0.0
    prefs["is_one_week"] = 1.0 if answers.get("trip_length") == "week" else 0.0

    region = answers.get("region")
    if region in REGION_COLS:
        prefs[region] = 1.0

    return prefs


def _cosine(bundle: ModelBundle, user_scaled: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Cosine similarity of one vector against selected catalogue rows.

    Uses the row norms precomputed at fit time rather than recomputing the whole
    matrix per request.
    """
    user = user_scaled.ravel()
    user_norm = float(np.linalg.norm(user)) or 1e-9
    dots = bundle.matrix[rows] @ user
    return dots / (bundle.norms[rows] * user_norm)


def _diversify(order: np.ndarray, clusters: np.ndarray, top_n: int,
               cap: int = MAX_PER_CLUSTER) -> tuple[list[int], bool]:
    """Walk the ranking, capping how many results share a profile.

    If the cap cannot fill the shortlist — a small or homogeneous catalogue —
    the remaining places are filled from the ranking in order, so the caller
    always gets `top_n` results where they exist.
    """
    picked: list[int] = []
    used: dict[int, int] = {}
    for idx in order:
        cid = int(clusters[idx])
        if used.get(cid, 0) >= cap:
            continue
        picked.append(int(idx))
        used[cid] = used.get(cid, 0) + 1
        if len(picked) == top_n:
            return picked, True

    applied = len(picked) > 0
    for idx in order:
        if len(picked) == top_n:
            break
        if int(idx) not in picked:
            picked.append(int(idx))
    return picked, applied and len(used) > 1


def recommend(df: pd.DataFrame, bundle: ModelBundle, prefs: dict,
              top_n: int = DEFAULT_TOP_N, pool_index: np.ndarray | None = None) -> Recommendation:
    """Rank destinations for one traveller.

    `pool_index` restricts scoring to a subset of the catalogue (a region
    filter, say) while still using the model fitted on everything, so the
    scaling stays stable no matter how the pool is narrowed.
    """
    if df.empty:
        return Recommendation(frame=df.head(0))

    rows = np.arange(len(df)) if pool_index is None else np.asarray(pool_index, dtype=int)
    rows = rows[(rows >= 0) & (rows < len(bundle.matrix))]
    if rows.size == 0:
        log.warning("recommendation pool is empty after bounds checking")
        return Recommendation(frame=df.head(0))

    scores = _cosine(bundle, bundle.transform(prefs), rows)
    order_local = np.argsort(-scores)
    order_global = rows[order_local]

    clusters = bundle.labels
    picked, diversified = _diversify(order_global, clusters, min(top_n, rows.size))

    out = df.iloc[picked].copy()
    lookup = {int(g): float(s) for g, s in zip(order_global, scores[order_local])}
    out["similarity"] = [lookup.get(int(i), 0.0) for i in picked]
    out["match"] = ((out["similarity"] + 1) / 2 * 100).round(1)
    out["profile"] = [bundle.cluster_name(int(i)) for i in picked]
    out = out.reset_index(drop=True)

    return Recommendation(
        frame=out,
        profiles=sorted(set(out["profile"])),
        considered=int(rows.size),
        diversified=diversified,
    )

# ==========================================================================
# TRAVEL INTELLIGENCE — Travel intelligence derived from a matched destination.
# ==========================================================================

def region_of(row) -> str | None:
    for col, label in REGIONS:
        if to_float(row.get(col, 0)) == 1:
            return label
    return None


def tastes_of(row) -> dict:
    return {k: to_float(row.get(k, 3), 3.0) for k in TASTE_KEYS}


@safe(130, "daily cost")
def daily_cost(row) -> int:
    """Per-person daily spend, from budget tier and accommodation standard."""
    tier = to_int(row.get("budget_level_encoded", 2), 2, 1, 3)
    stars = to_float(row.get("HotelRating_encoded", 3), 3.0)
    return int(round(BUDGET_DAILY[tier] * (0.85 + 0.075 * stars), -1))


def trip_cost(row, nights: int, travellers: int) -> int:
    return daily_cost(row) * max(1, nights) * max(1, travellers)


@safe("Climate data unavailable", "climate_summary")
def climate_summary(row) -> str:
    t = to_float(row.get("temp_avg_yearly", 20), 20.0)
    if t >= 28:
        return f"Hot year-round, averaging {t:.0f}°C"
    if t >= 22:
        return f"Warm and settled, around {t:.0f}°C"
    if t >= 15:
        return f"Mild, averaging {t:.0f}°C"
    if t >= 8:
        return f"Cool, around {t:.0f}°C — pack layers"
    return f"Cold, averaging {t:.0f}°C"


@safe("Year-round", "best_season")
def best_season(row) -> str:
    """Infer a sensible window from latitude and average temperature."""
    lat = to_float(row.get("latitude", 0), 0.0)
    t = to_float(row.get("temp_avg_yearly", 20), 20.0)
    north = lat >= 0
    if abs(lat) < 15:                       # tropics: the dry months matter, not heat
        return "December to March, outside the rains"
    if t >= 26:                             # hot climates are best off-peak
        return "November to March" if north else "May to September"
    if t <= 8:                              # cold climates: either peak summer or snow season
        return "June to August, or December for snow"
    return "April to June and September to October" if north else \
           "October to December and March to May"


@safe(["Balanced"], "trip_style")
def trip_style(row) -> list[str]:
    tastes = tastes_of(row)
    ranked = sorted(tastes.items(), key=lambda kv: -kv[1])
    return [TASTE_LABEL[k] for k, v in ranked[:3] if v >= 3.5] or ["Balanced"]


@safe("This destination matched your overall preferences.", "explain")
def explain(row, prefs: dict) -> str:
    """Say plainly why this place surfaced, using the traveller's own answers."""
    tastes = tastes_of(row)
    strong = []
    for key in TASTE_KEYS:
        want = to_float(prefs.get(key, 3), 3.0)
        has = tastes.get(key, 3)
        if want >= 4 and has >= 4:
            strong.append(TASTE_LABEL[key].lower())

    city = clean(row.get("city")) or "This destination"
    tier = BUDGET_LABEL.get(int(to_float(row.get("budget_level_encoded", 2), 2.0)), "Mid-range")

    if strong:
        wanted = ", ".join(strong[:2]) if len(strong) <= 2 else \
            f"{', '.join(strong[:2])} and {strong[2]}"
        lead = f"You asked for {wanted}, and {city} scores highly on all of it"
    else:
        lead = f"{city} sits closest to the overall balance you described"

    temp_gap = abs(to_float(row.get("temp_avg_yearly", 20), 20.0) - to_float(prefs.get("temp_avg_yearly", 22), 22.0))
    climate = "with a climate close to your target" if temp_gap <= 4 else \
              "though the climate runs a little off your target"
    return f"{lead}, {climate}. It fits a {tier.lower()} budget."


@safe([], "attractions_of")
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


@safe("Book accommodation before flights.", "travel_tip")
def travel_tip(row) -> str:
    """A short, situation-specific pointer built from the row's own numbers."""
    tastes = tastes_of(row)
    t = to_float(row.get("temp_avg_yearly", 20), 20.0)
    tier = int(to_float(row.get("budget_level_encoded", 2), 2.0))
    code = clean(row.get("iata"))

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

def travel_insights(results: pd.DataFrame, prefs: dict, answers: dict,
             profiles: list[str] | None = None) -> list[tuple[str, str]]:
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

    if profiles and len(profiles) > 1:
        listed = ", ".join(profiles[:-1]) + f" and {profiles[-1]}"
        out.append((
            "Distinct options",
            f"Your shortlist covers {len(profiles)} different destination profiles — "
            f"{listed} — so these are genuine alternatives rather than variations "
            "on one idea.",
        ))
    elif profiles:
        out.append((
            "A clear direction",
            f"Every match falls into the same profile ({profiles[0]}), which means "
            "your preferences point somewhere specific. Choose on cost and season.",
        ))

    with_air = sum(1 for _, r in results.iterrows()
                   if clean(r.get("name")) or clean(r.get("iata")))
    if with_air < len(results):
        out.append((
            "Getting there",
            f"{len(results) - with_air} of these have no major airport recorded in the "
            "dataset, which usually means a connecting flight plus ground transfer.",
        ))
    return out

# ==========================================================================
# VALIDATION — Validation of everything that arrives from the form.
# ==========================================================================

@dataclass
class Validated:
    """Cleaned answers plus anything that had to be corrected."""
    answers: dict
    notices: list[str] = field(default_factory=list)
    blocking: str = "" 

    @property
    def ok(self) -> bool:
        return not self.blocking


def _clamp(value, lo, hi, default, label, notices, integer=False):
    raw = to_int(value, default, lo, hi) if integer else to_float(value, default)
    if not integer:
        raw = min(hi, max(lo, raw))
    original = to_float(value, default)
    if abs(original - raw) > 1e-9:
        notices.append(f"{label} was adjusted to {raw:g} (allowed {lo:g}–{hi:g}).")
    return raw


def validate_answers(answers: dict | None) -> Validated:
    """Coerce the planner's answers into a safe, complete set."""
    notices: list[str] = []
    src = answers or {}
    if not isinstance(src, dict):
        log.warning("answers arrived as %s, not a dict", type(src).__name__)
        src = {}

    out: dict = {}
    for key in TASTE_KEYS:
        out[key] = _clamp(src.get(key, 3), TASTE_MIN, TASTE_MAX, 3,
                          key.title(), notices, integer=True)

    out["budget"] = to_int(src.get("budget", 2), 2, 1, 3)
    out["temp"] = _clamp(src.get("temp", 22), TEMP_MIN, TEMP_MAX, 22,
                         "Temperature", notices)
    out["stars"] = _clamp(src.get("stars", 4), STARS_MIN, STARS_MAX, 4.0,
                          "Accommodation standard", notices)
    out["nights"] = _clamp(src.get("nights", 7), NIGHTS_MIN, NIGHTS_MAX, 7,
                           "Nights", notices, integer=True)
    out["travellers"] = _clamp(src.get("travellers", 2), TRAVELLERS_MIN,
                               TRAVELLERS_MAX, 2, "Travellers", notices, integer=True)

    length = str(src.get("trip_length", "week")).lower()
    out["trip_length"] = length if length in {"short", "week"} else "week"

    region = src.get("region")
    if region and region not in REGION_COLS:
        notices.append("That region is not in the catalogue, so all regions were searched.")
        region = None
    out["region"] = region

    out["needs_airport"] = bool(src.get("needs_airport", True))
    return Validated(answers=out, notices=notices)


def validate_pool(df: pd.DataFrame, answers: dict, minimum: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """Apply the hard filters, relaxing any that would empty the catalogue.

    A filter that leaves nothing to rank is worse than no filter, so each one is
    reverted with an explanation rather than returning an empty result.
    """
    notices: list[str] = []
    pool = df

    region = answers.get("region")
    if region and region in pool.columns:
        filtered = pool[pool[region] == 1]
        if len(filtered) >= minimum:
            pool = filtered
        else:
            notices.append(
                f"Only {len(filtered)} destination(s) in {REGION_LABEL.get(region, 'that region')}, "
                "so the search was widened to every region."
            )

    if pool.empty:
        notices.append("No destinations matched those filters, so all were considered.")
        return df, notices

    return pool, notices


def validate_catalogue(df: pd.DataFrame) -> str:
    """A blocking message if the catalogue cannot support a recommendation."""
    if df is None or df.empty:
        return "The destination catalogue is empty, so nothing can be ranked."
    if len(df) < 2:
        return "The catalogue holds a single destination — at least two are needed to rank."
    return ""

# ==========================================================================
# STYLING — Visual styling — dark navy travel-tech dashboard.
# ==========================================================================

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Outfit:wght@400;500;600;700;800&"
    "family=Inter:wght@400;500;600&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)

THEME_CSS = """
<style>
@import url('__FONTS__');

/* ======================================================================
   TOKENS
   ==================================================================== */
:root{
  --navy-900:#060B16;
  --navy-800:#0A1222;
  --navy-700:#0E1A2E;
  --navy-600:#13233C;
  --navy-500:#1A2E4C;

  --text:#E9F0FA;
  --text-2:#A7B8D0;
  --text-3:#6F84A3;

  --cyan:#22D3EE;
  --sky:#38BDF8;
  --blue:#3B82F6;
  --deep:#2563EB;

  --line:rgba(120,160,220,.14);
  --line-2:rgba(120,160,220,.24);

  --glow-soft:0 0 24px rgba(34,211,238,.18);
  --sh-card:0 1px 2px rgba(0,0,0,.3), 0 18px 44px rgba(0,0,0,.42);
  --sh-lift:0 28px 68px rgba(0,0,0,.55), 0 0 40px rgba(56,189,248,.12);

  --r-xl:28px; --r-lg:22px; --r-md:14px; --r-sm:10px;
  --ease:cubic-bezier(.22,.75,.28,1);
}

/* ======================================================================
   PAGE
   ==================================================================== */
.stApp{
  background:
    radial-gradient(900px 520px at 8% -6%,  rgba(34,211,238,.10), transparent 60%),
    radial-gradient(820px 500px at 94% 2%,  rgba(59,130,246,.13), transparent 58%),
    radial-gradient(700px 620px at 50% 112%, rgba(56,189,248,.07), transparent 62%),
    linear-gradient(180deg,#060B16 0%,#0A1222 45%,#0C1526 100%);
  background-attachment:fixed;
}
#MainMenu, footer{ visibility:hidden; }
.block-container{ max-width:1180px; padding-top:1.5rem; padding-bottom:6rem; }

html, body, .stApp, [class*="css"]{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  color:var(--text); -webkit-font-smoothing:antialiased;
}
::selection{ background:rgba(34,211,238,.3); }
:focus-visible{ outline:2px solid var(--cyan); outline-offset:3px; border-radius:8px; }

/* Streamlit writes body text in its own colour; pull it onto the dark ground */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{
  color:var(--text-2);
}
.stMarkdown strong, [data-testid="stMarkdownContainer"] strong{ color:var(--text); }
h1, h2, h3, h4, h5, h6{ color:var(--text) !important; }

/* ======================================================================
   TYPE
   ==================================================================== */
.tw-display{
  font-family:'Outfit',sans-serif; font-weight:800;
  letter-spacing:-.04em; line-height:1.02; color:var(--text);
  font-size:clamp(2.5rem,6vw,4.6rem); margin:1rem 0 1.2rem;
}
.tw-grad{
  background:linear-gradient(100deg,var(--cyan) 0%,var(--sky) 45%,var(--blue) 100%);
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; color:var(--sky);
  filter:drop-shadow(0 0 28px rgba(34,211,238,.35));
}
.tw-eyebrow{
  font-family:'JetBrains Mono',ui-monospace,monospace; font-size:.66rem;
  letter-spacing:.26em; text-transform:uppercase; color:var(--cyan);
  display:inline-flex; align-items:center; gap:.6rem;
}
.tw-eyebrow::before{
  content:""; width:26px; height:1px;
  background:linear-gradient(90deg,var(--cyan),transparent);
}
.tw-lede{ color:var(--text-2); font-size:clamp(1rem,1.5vw,1.14rem);
  line-height:1.72; max-width:580px; }
.tw-h2{
  font-family:'Outfit',sans-serif; font-weight:700;
  font-size:clamp(1.6rem,3.2vw,2.4rem); letter-spacing:-.03em;
  margin:.5rem 0 .55rem; color:var(--text); line-height:1.1;
}
.tw-sub{ color:var(--text-3); font-size:.96rem; line-height:1.6; margin:0; }
.tw-mono{ font-family:'JetBrains Mono',ui-monospace,monospace; }

/* ======================================================================
   BRAND + HERO
   ==================================================================== */
.tw-brand{ display:flex; align-items:center; gap:.65rem; margin-bottom:.3rem; }
.tw-brand svg{ width:30px; height:30px;
  filter:drop-shadow(0 0 12px rgba(34,211,238,.5)); }
.tw-brand b{
  font-family:'Outfit',sans-serif; font-weight:800; font-size:1.24rem;
  letter-spacing:-.025em; color:var(--text);
}

.tw-hero{ position:relative; padding:1.2rem 0 .4rem; }
.tw-hero__inner{ max-width:720px; }
.tw-hero__plane{
  position:absolute; right:-1%; top:0; width:min(42%,420px);
  pointer-events:none; animation:twFloat 8s ease-in-out infinite;
}
.tw-hero__plane svg{ width:100%; height:auto; overflow:visible; }
@keyframes twFloat{ 0%,100%{ transform:translateY(0) } 50%{ transform:translateY(-14px) } }
@media (max-width:920px){ .tw-hero__plane{ display:none } }

/* ======================================================================
   METRICS
   ==================================================================== */
.tw-stats{
  display:flex; gap:clamp(1.2rem,3.5vw,2.6rem); flex-wrap:wrap;
  padding:1.5rem 0 .3rem; border-top:1px solid var(--line); margin-top:1.8rem;
}
.tw-stat b{
  display:block; font-family:'JetBrains Mono',ui-monospace,monospace;
  font-weight:500; font-size:clamp(1.35rem,2.8vw,1.9rem);
  letter-spacing:-.03em;
  background:linear-gradient(100deg,var(--cyan),var(--sky));
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent;
}
.tw-stat span{ font-size:.74rem; color:var(--text-3); letter-spacing:.05em; }

.tw-metrics{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:.8rem; margin:1.6rem 0 .4rem;
}
.tw-metric{
  background:linear-gradient(158deg, rgba(26,46,76,.7), rgba(14,26,46,.82));
  border:1px solid var(--line); border-radius:var(--r-md);
  padding:1.05rem 1.15rem; box-shadow:var(--sh-card);
}
.tw-metric b{
  display:block; font-family:'JetBrains Mono',ui-monospace,monospace;
  font-weight:500; font-size:1.55rem; letter-spacing:-.03em; line-height:1.1;
  background:linear-gradient(100deg,var(--cyan),var(--sky));
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent;
}
.tw-metric span{
  display:block; margin-top:.3rem; font-size:.7rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--text-3);
}

/* ======================================================================
   CARDS
   ==================================================================== */
.tw-card{
  position:relative;
  background:linear-gradient(158deg, rgba(26,46,76,.78) 0%, rgba(14,26,46,.86) 100%);
  backdrop-filter:blur(14px) saturate(1.2);
  -webkit-backdrop-filter:blur(14px) saturate(1.2);
  border:1px solid var(--line);
  border-radius:var(--r-lg); box-shadow:var(--sh-card);
  transition:transform .34s var(--ease), box-shadow .34s var(--ease),
             border-color .34s var(--ease);
  height:100%; overflow:hidden;
}
.tw-card::before{
  content:""; position:absolute; inset:0 0 auto; height:1px;
  background:linear-gradient(90deg,transparent,rgba(56,189,248,.55),transparent);
}
.tw-card:hover{
  transform:translateY(-6px); box-shadow:var(--sh-lift);
  border-color:rgba(56,189,248,.34);
}

.tw-feat{ padding:1.8rem 1.5rem; }
.tw-feat__icon{
  width:64px; height:64px; border-radius:20px; display:grid; place-items:center;
  background:linear-gradient(140deg, rgba(34,211,238,.2), rgba(59,130,246,.28));
  border:1px solid rgba(56,189,248,.34);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.1), 0 0 26px rgba(34,211,238,.14);
  margin-bottom:1.1rem;
}
.tw-feat__icon svg{ width:30px; height:30px; fill:none; stroke:var(--cyan);
  stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round;
  filter:drop-shadow(0 0 8px rgba(34,211,238,.5)); }
.tw-feat h3{
  font-family:'Outfit',sans-serif; font-weight:700; font-size:1.1rem;
  margin:0 0 .45rem; letter-spacing:-.02em; color:var(--text);
}
.tw-feat p{ color:var(--text-2); font-size:.91rem; line-height:1.62; margin:0; }

/* ======================================================================
   DESTINATION CARD
   ==================================================================== */
.tw-dest__art{ position:relative; height:168px; overflow:hidden; }
.tw-dest__art svg{ width:100%; height:100%; display:block; }
.tw-dest__scrim{
  position:absolute; inset:auto 0 0 0; height:72%;
  background:linear-gradient(180deg,transparent,rgba(6,11,22,.9));
}
.tw-dest__place{ position:absolute; left:16px; bottom:12px; right:92px; }
.tw-dest__place h3{
  font-family:'Outfit',sans-serif; font-weight:800; color:#fff;
  font-size:1.24rem; letter-spacing:-.025em; margin:0; line-height:1.15;
  text-shadow:0 2px 14px rgba(0,0,0,.6);
}
.tw-dest__place span{ color:rgba(233,240,250,.78); font-size:.78rem; }
.tw-dest__match{
  position:absolute; right:12px; top:12px; padding:.34rem .64rem; border-radius:99px;
  background:rgba(10,18,34,.82); border:1px solid rgba(56,189,248,.4);
  backdrop-filter:blur(8px);
  font-family:'JetBrains Mono',ui-monospace,monospace; font-size:.72rem;
  color:var(--cyan); box-shadow:0 0 18px rgba(34,211,238,.22);
}
.tw-dest__body{ padding:1.05rem 1.2rem 1.25rem; display:grid; gap:.85rem; }

.tw-facts{ display:grid; grid-template-columns:repeat(3,1fr); gap:.4rem; }
.tw-facts span{
  display:block; font-family:'JetBrains Mono',ui-monospace,monospace;
  font-size:.56rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--text-3); margin-bottom:.18rem;
}
.tw-facts b{ font-size:.9rem; font-weight:600; color:var(--text); }

.tw-why{
  border-left:2px solid var(--cyan); padding:.1rem 0 .1rem .8rem;
  color:var(--text-2); font-size:.86rem; line-height:1.58;
}
.tw-chips{ display:flex; flex-wrap:wrap; gap:.34rem; }
.tw-chip{
  padding:.24rem .6rem; border-radius:99px; font-size:.7rem; font-weight:500;
  background:rgba(34,211,238,.12); color:var(--cyan);
  border:1px solid rgba(34,211,238,.24);
}
.tw-chip--muted{
  background:rgba(120,160,220,.09); color:var(--text-3);
  border-color:var(--line);
}

.tw-meta{ border-top:1px solid var(--line); padding-top:.75rem; display:grid; gap:.42rem; }
.tw-meta__row{ display:flex; gap:.55rem; align-items:flex-start;
  font-size:.82rem; color:var(--text-2); line-height:1.5; }
.tw-meta__row svg{ width:14px; height:14px; flex:0 0 auto; margin-top:3px;
  stroke:var(--sky); fill:none; stroke-width:1.9;
  stroke-linecap:round; stroke-linejoin:round; }
.tw-meta__row b{ color:var(--text); font-weight:600; }

/* ======================================================================
   INSIGHT
   ==================================================================== */
.tw-insight{ padding:1.25rem 1.35rem; display:flex; gap:.95rem; align-items:flex-start; }
.tw-insight__dot{
  width:38px; height:38px; border-radius:12px; flex:0 0 auto; display:grid;
  place-items:center;
  background:linear-gradient(140deg, rgba(34,211,238,.18), rgba(59,130,246,.24));
  border:1px solid rgba(56,189,248,.32);
  box-shadow:0 0 20px rgba(34,211,238,.16);
}
.tw-insight__dot svg{ width:18px; height:18px; stroke:var(--cyan); fill:none;
  stroke-width:1.9; stroke-linecap:round; stroke-linejoin:round; }
.tw-insight h4{
  font-family:'Outfit',sans-serif; font-size:.94rem; font-weight:700;
  margin:0 0 .28rem; color:var(--text);
}
.tw-insight p{ margin:0; font-size:.87rem; line-height:1.62; color:var(--text-2); }

/* ======================================================================
   BUTTONS — gradient, rounded, with a restrained cyan glow
   ==================================================================== */
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button{
  font-family:'Inter',sans-serif; font-weight:600; font-size:.94rem;
  border-radius:99px; padding:.7rem 1.7rem;
  transition:transform .24s var(--ease), box-shadow .24s var(--ease),
             filter .24s var(--ease), border-color .24s var(--ease);
}
.stButton>button[kind="primary"], .stFormSubmitButton>button{
  background:linear-gradient(100deg,var(--cyan) 0%,var(--sky) 48%,var(--blue) 100%);
  color:#04121F !important; border:none;
  box-shadow:0 10px 26px rgba(37,99,235,.34), 0 0 22px rgba(34,211,238,.26),
             inset 0 1px 0 rgba(255,255,255,.28);
}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover{
  transform:translateY(-2px); filter:brightness(1.06);
  box-shadow:0 16px 36px rgba(37,99,235,.42), 0 0 34px rgba(34,211,238,.42),
             inset 0 1px 0 rgba(255,255,255,.34);
}
.stButton>button[kind="primary"]:active, .stFormSubmitButton>button:active{
  transform:translateY(0); filter:brightness(.97);
}
.stButton>button[kind="secondary"]{
  background:rgba(26,46,76,.6); color:var(--text) !important;
  border:1px solid var(--line-2); box-shadow:0 6px 18px rgba(0,0,0,.3);
}
.stButton>button[kind="secondary"]:hover{
  border-color:rgba(56,189,248,.55); color:var(--cyan) !important;
  transform:translateY(-2px); box-shadow:0 10px 26px rgba(0,0,0,.36), var(--glow-soft);
}

/* ======================================================================
   TABS — the navigation, dressed as a glowing pill bar
   ==================================================================== */
.stTabs [data-baseweb="tab-list"]{
  gap:.3rem; border-bottom:none; padding:.4rem; margin-bottom:1.5rem;
  background:rgba(14,26,46,.72);
  backdrop-filter:blur(16px) saturate(1.3);
  -webkit-backdrop-filter:blur(16px) saturate(1.3);
  border:1px solid var(--line);
  border-radius:99px; box-shadow:var(--sh-card); width:fit-content;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Inter',sans-serif; font-weight:600; font-size:.93rem;
  color:var(--text-3); padding:.6rem 1.5rem; border-radius:99px;
  transition:color .24s var(--ease), background .24s var(--ease);
}
.stTabs [data-baseweb="tab"]:hover{ color:var(--text); background:rgba(56,189,248,.1); }
.stTabs [aria-selected="true"]{
  color:#04121F !important;
  background:linear-gradient(100deg,var(--cyan),var(--sky));
  box-shadow:0 8px 22px rgba(34,211,238,.3), 0 0 26px rgba(34,211,238,.24);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }
.stTabs [data-baseweb="tab-panel"]{ padding-top:.5rem; }

/* ======================================================================
   FORM CONTROLS
   ==================================================================== */
[data-testid="stSlider"] label, .stSelectbox label, .stRadio label,
.stCheckbox label, .stNumberInput label, .stMultiSelect label,
.stTextInput label, [data-testid="stWidgetLabel"] p{
  font-size:.85rem !important; font-weight:600 !important;
  color:var(--text-2) !important;
}
[data-testid="stThumbValue"]{
  font-family:'JetBrains Mono',ui-monospace,monospace !important;
  font-size:.7rem !important; color:#04121F !important;
  background:var(--cyan) !important; padding:.08rem .44rem !important;
  border-radius:7px !important; box-shadow:0 0 14px rgba(34,211,238,.45) !important;
}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"]{
  font-family:'JetBrains Mono',ui-monospace,monospace !important;
  font-size:.62rem !important; color:var(--text-3) !important;
}
[data-testid="stSlider"] [role="slider"]{
  box-shadow:0 0 0 3px rgba(34,211,238,.22), 0 0 16px rgba(34,211,238,.4) !important;
}

div[data-baseweb="select"] > div,
.stNumberInput input, .stTextInput input, .stNumberInput [data-baseweb="input"]{
  background:rgba(14,26,46,.86) !important;
  border:1px solid var(--line-2) !important;
  border-radius:var(--r-sm) !important;
  color:var(--text) !important;
  min-height:2.8rem !important;
}
div[data-baseweb="select"] > div:hover,
.stNumberInput input:hover, .stTextInput input:hover{
  border-color:rgba(56,189,248,.5) !important;
}
div[data-baseweb="select"] svg{ fill:var(--text-2) !important; }
div[data-baseweb="select"] [data-baseweb="tag"]{
  background:rgba(34,211,238,.16) !important;
  border:1px solid rgba(34,211,238,.3) !important;
  color:var(--cyan) !important;
}

/* Dropdown menus mount in a portal at the document root, outside .stApp, so
   they keep a white background unless targeted directly. */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] ul, [data-baseweb="menu"], [data-baseweb="menu"] ul{
  background:#0E1A2E !important; border:1px solid var(--line-2) !important;
  box-shadow:0 20px 48px rgba(0,0,0,.6) !important;
}
[data-baseweb="menu"] li, [role="option"], [data-baseweb="menu"] li div{
  color:var(--text) !important;
}
[data-baseweb="menu"] li:hover, [role="option"]:hover{
  background:rgba(56,189,248,.16) !important; color:var(--cyan) !important;
}

div[role="radiogroup"] label{
  background:rgba(26,46,76,.55); border:1px solid var(--line-2);
  border-radius:99px; padding:.4rem .95rem !important; margin:0 .4rem 0 0 !important;
  transition:all .22s var(--ease);
}
div[role="radiogroup"] label:hover{ border-color:rgba(56,189,248,.5); }

/* ======================================================================
   DATA SURFACES
   ==================================================================== */
[data-testid="stMetricValue"]{
  font-family:'JetBrains Mono',ui-monospace,monospace; font-weight:500;
  color:var(--cyan);
}
[data-testid="stMetricLabel"] p{ color:var(--text-3) !important; }

[data-testid="stExpander"] details{
  border:1px solid var(--line); border-radius:var(--r-md);
  background:rgba(14,26,46,.7);
}
[data-testid="stExpander"] summary{ color:var(--text-2) !important; }

[data-testid="stDataFrame"]{
  border:1px solid var(--line); border-radius:var(--r-md); overflow:hidden;
}
[data-testid="stAlert"]{
  background:rgba(14,26,46,.85) !important; border:1px solid var(--line-2) !important;
  border-radius:var(--r-md) !important; color:var(--text) !important;
}
[data-testid="stAlert"] p{ color:var(--text-2) !important; }

hr, [data-testid="stDivider"] hr{ border-color:var(--line) !important; }

/* ======================================================================
   MOTION
   ==================================================================== */
.tw-rise{ animation:twRise .7s var(--ease) both; }
@keyframes twRise{ from{ opacity:0; transform:translateY(18px); } to{ opacity:1; transform:none; } }
.d1{animation-delay:.05s}.d2{animation-delay:.13s}.d3{animation-delay:.21s}
.d4{animation-delay:.29s}.d5{animation-delay:.37s}

@media (max-width:640px){
  .tw-facts{ grid-template-columns:repeat(2,1fr); }
  .tw-dest__place{ right:80px; }
  .stTabs [data-baseweb="tab"]{ padding:.55rem 1rem; font-size:.85rem; }
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{ animation-duration:.01ms !important; transition-duration:.01ms !important; }
  .tw-card:hover{ transform:none; }
}
</style>
""".replace("__FONTS__", FONTS)

# ==========================================================================
# DESTINATION ARTWORK — Generative destination artwork.
# ==========================================================================

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

    # Gradient ids must be unique across the page. Six cards each defining
    # id="sky" is invalid HTML, and the browser resolves every url(#sky) to the
    # first one — so without this suffix all six cards render in the first
    # card's colours.
    uid = hashlib.sha256(city.encode("utf-8")).hexdigest()[:8]

    body = SCENES[scene](pal, rnd).replace("url(#sea)", f"url(#sea{uid})")
    sun = _sun(pal, rnd, low_sun).replace("url(#glow)", f"url(#glow{uid})")

    return (
        f'<svg viewBox="0 0 {ART_W} {ART_H}" xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="xMidYMid slice" class="tw-art">'
        f"<defs>"
        f'<linearGradient id="sky{uid}" x1="0" y1="0" x2="0" y2="1">{stops}</linearGradient>'
        f'<linearGradient id="sea{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{pal["water"]}"/>'
        f'<stop offset="100%" stop-color="{pal["land"][1]}"/></linearGradient>'
        f'<radialGradient id="glow{uid}">'
        f'<stop offset="0%" stop-color="{pal["sun"]}" stop-opacity=".55"/>'
        f'<stop offset="100%" stop-color="{pal["sun"]}" stop-opacity="0"/></radialGradient>'
        f"</defs>"
        f'<rect width="{ART_W}" height="{ART_H}" fill="url(#sky{uid})"/>'
        f'<g fill="#FFFFFF">{clouds}</g>'
        f"{sun}"
        f"{body}"
        f"</svg>"
    )

# ==========================================================================
# DISPLAY MARKUP — HTML builders.
# ==========================================================================

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


def _e(value) -> str:
    return escape(str(value), quote=True)


def brand() -> str:
    return """
<div class="tw-brand">
<svg viewBox="0 0 24 24" fill="none" stroke="url(#bg1)" stroke-width="1.8"
stroke-linecap="round" stroke-linejoin="round">
<defs><linearGradient id="bg1" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/>
</linearGradient></defs>
<path d="M2 13.4 21 4l-4.6 16.2-4.1-6.1z"/><path d="m12.3 14.1-3.4 6.9-1-5"/>
</svg>
<b>TripWise <span class="tw-grad">AI</span></b>
</div>"""


def hero(stats: list[tuple[str, str]]) -> str:
    """The landing headline, with the aircraft drifting alongside it."""
    cells = "".join(
        f'<div class="tw-stat"><b>{_e(v)}</b><span>{_e(label)}</span></div>'
        for v, label in stats
    )
    return f"""
<div class="tw-hero">
<div class="tw-hero__plane">
<svg viewBox="0 0 420 250" fill="none">
<defs>
<linearGradient id="planeBody" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/>
</linearGradient>
<linearGradient id="planeTrail" x1="0" y1="0" x2="1" y2="0">
<stop offset="0%" stop-color="#0EA5E9" stop-opacity="0"/>
<stop offset="100%" stop-color="#2563EB" stop-opacity=".5"/>
</linearGradient>
</defs>
<path d="M12 208 C120 198 208 148 288 70" stroke="url(#planeTrail)"
stroke-width="3" stroke-dasharray="2 12" stroke-linecap="round"/>
<circle cx="298" cy="62" r="52" fill="url(#planeBody)" opacity=".09"/>
<g transform="translate(298,62) rotate(-38)">
<path d="M46 0 L16 -7 L-6 -34 L-18 -34 L-9 -7 L-30 -7 L-40 -17 L-48 -17
L-42 -5 L-42 5 L-48 17 L-40 17 L-30 7 L-9 7 L-18 34 L-6 34 L16 7 Z"
fill="url(#planeBody)"/>
<path d="M46 0 L16 -7 L-9 -7 L-9 7 L16 7 Z" fill="#ffffff" opacity=".3"/>
</g>
<circle cx="12" cy="208" r="6" fill="#2563EB"/>
<circle cx="12" cy="208" r="11" fill="#2563EB" opacity=".2"/>
</svg>
</div>
<div class="tw-hero__inner">
<span class="tw-eyebrow tw-rise d1">Intelligent travel planning</span>
<h1 class="tw-display tw-rise d2">Plan smarter with<br/><span class="tw-grad">TripWise AI</span></h1>
<p class="tw-lede tw-rise d3">Tell us how you like to travel and TripWise ranks
real destinations against your answers &mdash; with the cost, the climate, the
right season and the nearest airport worked out before you commit to anything.</p>
<div class="tw-stats tw-rise d4">{cells}</div>
</div>
</div>"""


def metrics(items: list[tuple[str, str]]) -> str:
    """A row of headline figures — the shape of an answer, before its detail."""
    cells = "".join(
        f'<div class="tw-metric"><b>{_e(value)}</b><span>{_e(label)}</span></div>'
        for value, label in items
    )
    return f'<div class="tw-metrics tw-rise">{cells}</div>'


def heading(eyebrow: str, title: str, sub: str = "") -> str:
    tail = f'<p class="tw-sub">{_e(sub)}</p>' if sub else ""
    return (f'<div class="tw-rise" style="margin:3rem 0 1.6rem;max-width:660px">'
            f'<span class="tw-eyebrow">{_e(eyebrow)}</span>'
            f'<div class="tw-h2">{_e(title)}</div>{tail}</div>')


def feature(icon_name: str, title: str, body: str) -> str:
    return f"""
<div class="tw-card tw-feat tw-rise">
<div class="tw-feat__icon">{icon(icon_name)}</div>
<h3>{_e(title)}</h3>
<p>{_e(body)}</p>
</div>"""


def insight(title: str, body: str) -> str:
    return f"""
<div class="tw-card tw-insight tw-rise">
<div class="tw-insight__dot">{icon('bulb')}</div>
<div><h4>{_e(title)}</h4><p>{_e(body)}</p></div>
</div>"""


def destination(d: dict) -> str:
    """`d` is the flat dict assembled by app.py, never a dataframe row."""
    chips = "".join(f'<span class="tw-chip">{_e(s)}</span>' for s in d["style"])
    if d.get("region"):
        chips += f'<span class="tw-chip tw-chip--muted">{_e(d["region"])}</span>'

    rows = [("plane", "Nearest airport",
             d["airport"] or "None recorded in the catalogue"),
            ("calendar", "Best season", d["season"]),
            ("sun", "Weather", d["climate"])]
    if d.get("hotel"):
        rows.append(("tag", "Suggested stay", d["hotel"]))
    if d.get("attractions"):
        rows.append(("pin", "Nearby", ", ".join(d["attractions"])))
    rows.append(("bulb", "Tip", d["tip"]))

    meta = "".join(
        f'<div class="tw-meta__row">{icon(ic)}<div><b>{_e(label)}</b> &mdash; {_e(val)}</div></div>'
        for ic, label, val in rows
    )

    return f"""
<div class="tw-card tw-rise">
<div class="tw-dest__art">
{d['art']}
<div class="tw-dest__scrim"></div>
<div class="tw-dest__match">{_e(d['match'])}% match</div>
<div class="tw-dest__place"><h3>{_e(d['city'])}</h3><span>{_e(d['country'])}</span></div>
</div>
<div class="tw-dest__body">
<div class="tw-facts">
<div><span>Budget</span><b>{_e(d['tier'])}</b></div>
<div><span>Per day</span><b>${_e(d['daily'])}</b></div>
<div><span>Climate</span><b>{_e(d['temp'])}&deg;C</b></div>
</div>
<div class="tw-chips">{chips}</div>
<div class="tw-why">{_e(d['why'])}</div>
<div class="tw-meta">{meta}</div>
</div>
</div>"""


def about() -> str:
    return f"""
<div class="tw-card tw-feat tw-rise">
<div class="tw-feat__icon">{icon('sparkle')}</div>
<h3>Recommendations you can trace</h3>
<p>TripWise compares your answers against every destination across nine travel
dimensions, climate, budget tier and accommodation standard, then ranks by how
closely each sits to what you described. Results are spread across distinct
destination profiles so a shortlist offers real alternatives, and every card
explains its own reasoning in plain language.</p>
</div>"""


def data_promise() -> str:
    return f"""
<div class="tw-card tw-feat tw-rise">
<div class="tw-feat__icon">{icon('shield')}</div>
<h3>Grounded in real data</h3>
<p>Destinations, airports, accommodation standards and climate averages come
from a curated catalogue rather than generated text, so a recommendation always
points somewhere that exists. When a detail is missing, TripWise says so instead
of inventing one.</p>
</div>"""

# ==========================================================================
# OPENING ANIMATION — Generate the TripWise boarding splash as one self-contained HTML string.
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
.tw-splash, .tw-splash *{{
  /* Decorative only. This is declared with !important on the overlay and every
     descendant so a click always reaches the page beneath, even if the fade
     animation never runs. An overlay that could swallow clicks was the most
     likely cause of an earlier fault, and this makes that impossible. */
  pointer-events:none !important;
}}
.tw-splash{{
  position:fixed; inset:0; z-index:99998; display:grid; place-items:center;
  background:linear-gradient(180deg,#060B16 0%,#0A1222 50%,#0C1526 100%);
  animation:twSplashOut .9s ease 4.4s forwards;
}}
@keyframes twSplashOut{{ to{{ opacity:0; visibility:hidden; display:none; }} }}
.tw-splash .tw-scene{{
  width:min(96vw,1240px); aspect-ratio:{SPL_W}/{SPL_H};
  animation:twSceneIn 1.2s cubic-bezier(.2,.7,.3,1) both;
}}
@keyframes twSceneIn{{ from{{ opacity:0; transform:scale(1.05); }} to{{ opacity:1; transform:none; }} }}
.tw-splash .tw-scene svg{{
  width:100%; height:100%; display:block;
  filter:drop-shadow(0 30px 64px rgba(0,0,0,.6));
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
  background:linear-gradient(100deg,#22D3EE,#38BDF8);
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; color:#2563EB;
  filter:drop-shadow(0 0 26px rgba(34,211,238,.55));
}}
.tw-splash .tw-mark .s{{
  font-family:'Inter',sans-serif; margin-top:.5rem; color:#A7B8D0; font-weight:500;
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
<stop offset="0" stop-color="#3B4E6E"/><stop offset=".5" stop-color="#2C3C58"/>
<stop offset="1" stop-color="#1E2B42"/>
</linearGradient>
<linearGradient id="cabin" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#1B2942"/><stop offset=".42" stop-color="#141F33"/>
<stop offset="1" stop-color="#0D1626"/>
</linearGradient>
<radialGradient id="vign" cx="50%" cy="42%" r="72%">
<stop offset="55%" stop-color="#000" stop-opacity="0"/>
<stop offset="100%" stop-color="#02060E" stop-opacity=".55"/>
</radialGradient>
<filter id="winShadow" x="-25%" y="-25%" width="150%" height="150%">
<feDropShadow dx="0" dy="12" stdDeviation="15" flood-color="#02060E" flood-opacity=".6"/>
</filter>
<linearGradient id="well" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#0E1728"/><stop offset=".28" stop-color="#1A2640"/>
<stop offset=".75" stop-color="#2A3A56"/><stop offset="1" stop-color="#3A4D6E"/>
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
<stop offset="0" stop-color="#1C2942"/><stop offset=".08" stop-color="#2B3B58"/>
<stop offset=".5" stop-color="#334566"/><stop offset=".92" stop-color="#2A3A56"/>
<stop offset="1" stop-color="#1A2740"/>
</linearGradient>
<linearGradient id="lipG" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#22314C"/><stop offset="1" stop-color="#16223A"/>
</linearGradient>
<linearGradient id="gripG" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#48daf0"/><stop offset="1" stop-color="#2b8fb8"/>
</linearGradient>
<linearGradient id="lipHi" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#ffffff" stop-opacity=".92"/>
<stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
</linearGradient>
{deck_masks()}
{''.join(per_window)}
</defs>
<rect width="{SPL_W}" height="{SPL_H}" fill="url(#cabin)"/>
<rect x="0" y="{SPL_H*0.055:.0f}" width="{SPL_W}" height="2" fill="#22D3EE" opacity=".18"/>
<rect x="0" y="{SPL_H*0.055+3:.0f}" width="{SPL_W}" height="1" fill="#3B4E6E" opacity=".5"/>
<rect x="0" y="{SPL_H*0.905:.0f}" width="{SPL_W}" height="2" fill="#22D3EE" opacity=".14"/>
<rect x="0" y="{SPL_H*0.905+3:.0f}" width="{SPL_W}" height="1" fill="#3B4E6E" opacity=".45"/>
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






def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# cached resources
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_catalogue(version: str = CACHE_VERSION):
    """Parse and clean the catalogue once. `version` busts the cache on release."""
    return load_catalogue_file()


@st.cache_resource(show_spinner=False)
def fitted_models(fingerprint: str) -> ModelBundle:
    df, _ = load_catalogue()
    return fit_models(df)


@st.cache_resource(show_spinner=False)
def airport_index(fingerprint: str) -> AirportIndex:
    df, _ = load_catalogue()
    return AirportIndex(df)


def fingerprint(df: pd.DataFrame) -> str:
    """Stable identity for a catalogue, scoped to this release."""
    return (f"{CACHE_VERSION}:{len(df)}x{len(df.columns)}:"
            f"{'|'.join(sorted(str(c) for c in df.columns))[:200]}")


# --------------------------------------------------------------------------- #
# view model
# --------------------------------------------------------------------------- #

def card_payload(row, prefs: dict, airports: AirportIndex) -> dict:
    airport = airports.resolve(row)
    temp = to_float(row.get("temp_avg_yearly", 20), 20.0)
    tier = to_int(row.get("budget_level_encoded", 2), 2, 1, 3)

    style = trip_style(row)
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
        "art": destination_art(str(row.get("city", "x")),
                               tastes_of(row), temp),
        "tier": BUDGET_LABEL[tier],
        "daily": daily_cost(row),
        "temp": f"{temp:.0f}",
        "style": style,
        "region": region_of(row),
        "why": explain(row, prefs),
        "airport": airport.label if airport else None,
        "season": best_season(row),
        "climate": climate_summary(row),
        "hotel": hotel,
        "attractions": attractions_of(row),
        "tip": travel_tip(row),
    }


def data_notice(report, airports: AirportIndex, df: pd.DataFrame) -> None:
    """Report what the catalogue can and cannot support.

    Fields are read through `report.get` because Streamlit may hand back a
    cached report built by an earlier release of the class.
    """
    source = report.get("source", "the catalogue")
    rows = report.get("rows_out", len(df))

    if not report.get("is_real", True):
        missing = report.get("missing_required", [])
        if report.get("error", ""):
            detail = report.get("error", "")
        elif missing:
            detail = "Missing required columns: " + ", ".join(missing) + "."
        else:
            detail = f"{CSV_NAME} was not found beside app.py."
        st.info(f"Running on the built-in demo catalogue. {detail}", icon="ℹ️")
        return

    if not len(airports):
        st.warning(
            f"**No airport data in {source}.** {rows:,} destinations loaded, but "
            "no column holding airport names or codes was found.", icon="⚠️")
        with st.expander("How to fix this"):
            st.markdown(
                "TripWise looks for a **`name`** column and an **`iata`** column, "
                "and accepts the spellings `airport_name`, `airport`, `IATA` and "
                "`iata_code`. They come from the airports merge in the notebook; "
                "if the export kept only the model features, they were dropped.")
            seen = report.get("columns_seen", [])
            if seen:
                st.caption(f"Columns found in {source}:")
                st.code(", ".join(str(c) for c in seen), language="text")
        return

    direct, total = coverage(df, airports)
    if total and direct < total:
        st.caption(f"{rows:,} destinations from {source}. {direct:,} carry a "
                   f"matched airport; the rest resolve to the nearest of "
                   f"{len(airports):,} known airports.")


# --------------------------------------------------------------------------- #
# tabs
# --------------------------------------------------------------------------- #

def tab_home(df: pd.DataFrame, report, airports: AirportIndex) -> None:
    """The landing view.

    Deliberately short. Three claims, a metrics strip and a route into the
    planner — a dashboard earns trust by showing signal, not by explaining
    itself at length.
    """
    html(hero(catalogue_stats(df)))
    data_notice(report, airports, df)

    html(heading("How it works", "Three steps to a shortlist you trust"))
    for col, (ic, title, body) in zip(st.columns(3, gap="medium"), [
        ("compass", "Describe your trip",
         "Nine travel dimensions, a budget tier and a climate target."),
        ("sparkle", "See a ranked shortlist",
         "Ranked by fit, spread across distinct destination profiles."),
        ("wallet", "Plan around real numbers",
         "Cost, season and nearest airport before you commit."),
    ]):
        with col:
            html(feature(ic, title, body))

    html(heading("Why it is trusted", "Advanced recommendation technology"))
    left, right = st.columns(2, gap="medium")
    with left:
        html(about())
    with right:
        html(data_promise())

    diagnostics()


def planner_form() -> tuple[dict, bool]:
    """Collect preferences. Every control is a native widget."""
    with st.form("planner", border=False):
        st.markdown("**What you care about**")
        answers: dict = {}
        cols = st.columns(3, gap="medium")
        for i, (key, label, hint) in enumerate(TASTES):
            with cols[i % 3]:
                answers[key] = st.slider(label, TASTE_MIN, TASTE_MAX,
                                         3, help=hint, key=f"t_{key}")

        st.markdown("**Budget and climate**")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            answers["budget"] = st.select_slider(
                "Budget level", options=[1, 2, 3], value=2,
                format_func=lambda v: BUDGET_LABEL[v])
        with c2:
            answers["temp"] = st.slider("Preferred average temperature (°C)", -5, 40, 22)
        with c3:
            answers["stars"] = st.slider("Minimum accommodation standard",
                                         STARS_MIN, STARS_MAX, 4.0, 0.5)

        st.markdown("**Where and how long**")
        c4, c5, c6 = st.columns(3, gap="medium")
        with c4:
            names = ["Anywhere"] + [REGION_LABEL[c] for c in REGION_COLS]
            picked = st.selectbox("Preferred region", names)
            answers["region"] = next(
                (c for c in REGION_COLS if REGION_LABEL[c] == picked), None)
        with c5:
            answers["nights"] = st.slider("Nights", NIGHTS_MIN, 30, 7)
            answers["trip_length"] = "short" if answers["nights"] <= 4 else "week"
        with c6:
            answers["travellers"] = st.number_input(
                "Travellers", TRAVELLERS_MIN, TRAVELLERS_MAX, 2)

        answers["needs_airport"] = st.checkbox(
            "Prefer destinations with a nearby airport", value=True)

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Find my destinations  →", type="primary")

    return answers, submitted


def tab_planner(df: pd.DataFrame, airports: AirportIndex) -> None:
    html(heading("AI Planner", "Tell us how you travel",
                    "Rate what matters to you. 1 means indifferent, 5 means it "
                    "would shape the whole trip."))

    raw, submitted = planner_form()
    if not submitted:
        st.caption("Set your preferences above, then press Find my destinations.")
        return

    blocking = validate_catalogue(df)
    if blocking:
        st.error(blocking, icon="⚠️")
        return

    checked = validate_answers(raw)
    for note in checked.notices:
        st.warning(note, icon="⚠️")
    answers = checked.answers

    pool, notes = validate_pool(df, answers)
    for note in notes:
        st.info(note, icon="ℹ️")

    result = None
    with guard("recommendation") as failed:
        bundle = fitted_models(fingerprint(df))
        prefs = build_preference_vector(df, answers)
        with st.spinner("Ranking destinations…"):
            result = recommend(df, bundle, prefs,
                                           top_n=DEFAULT_TOP_N,
                                           pool_index=pool.index.to_numpy())
    if failed or result is None:
        st.error("Something went wrong while ranking. Adjust a preference and "
                 "try again.", icon="⚠️")
        return
    if result.empty:
        st.warning("No destinations matched. Try widening the region or climate.",
                   icon="ℹ️")
        return

    st.session_state["last_result"] = result
    st.session_state["last_prefs"] = prefs
    st.session_state["last_answers"] = answers
    render_results(result, prefs, answers, airports)


def render_results(result, prefs: dict, answers: dict, airports: AirportIndex) -> None:
    frame = result.frame

    # headline numbers first — the shape of the answer before the detail
    costs = [daily_cost(r) for _, r in frame.iterrows()]
    nights, people = answers["nights"], answers["travellers"]
    html(metrics([
        (f"{frame['match'].max():.0f}%", "Best match"),
        (f"${min(costs) * nights * people:,}", "From, per trip"),
        (f"{len(result.profiles)}", "Distinct profiles"),
        (f"{frame['temp_avg_yearly'].mean():.0f}°C", "Average climate"),
    ]))

    html(heading("Your matches", "Where you should go",
                    f"Ranked from {result.considered:,} destinations for "
                    f"{people} traveller(s) over {nights} nights."))

    rows = list(frame.iterrows())
    for start in range(0, len(rows), 3):
        for col, (_, row) in zip(st.columns(3, gap="medium"), rows[start:start + 3]):
            with col, guard(f"card for {row.get('city')}"):
                html(destination(card_payload(row, prefs, airports)))

    readings: list = []
    with guard("insights"):
        readings = travel_insights(frame, prefs, answers, result.profiles)
    if readings:
        html(heading("AI insights", "What your results say",
                        "Read from the shortlist as a whole."))
        for start in range(0, len(readings), 2):
            for col, (title, body) in zip(st.columns(2, gap="medium"),
                                          readings[start:start + 2]):
                with col:
                    html(insight(title, body))

    with guard("charts") as failed:
        render_charts(frame, answers)
    if failed:
        st.caption("Charts are unavailable for this result set.")


def chart(fig) -> None:
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def render_charts(frame: pd.DataFrame, answers: dict) -> None:
    import plotly.express as px

    html(heading("Comparison", "Your shortlist side by side"))
    axis = dict(showgrid=True, gridcolor="rgba(120,160,220,.12)", zeroline=False,
                tickfont=dict(color="#6F84A3", size=12),
                title_font=dict(color="#6F84A3", size=12))
    layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="Inter, sans-serif", color="#E9F0FA", size=13),
                  margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False,
                  hoverlabel=dict(bgcolor="#0E1A2E", bordercolor="#22D3EE",
                                  font=dict(color="#E9F0FA")))
    scale = ["#1E3A8A", "#38BDF8", "#22D3EE"]

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Match strength**")
        fig = px.bar(frame.sort_values("match"), x="match", y="city",
                     orientation="h", color="match", color_continuous_scale=scale)
        fig.update_layout(**layout, xaxis=dict(**axis, title="Match %"),
                          yaxis=dict(**axis, title=""), coloraxis_showscale=False)
        fig.update_traces(hovertemplate="%{y}: %{x:.0f}%<extra></extra>")
        chart(fig)
    with right:
        costs = frame.assign(daily=[daily_cost(r) for _, r in frame.iterrows()])
        costs["total"] = costs["daily"] * answers["nights"] * answers["travellers"]
        st.markdown(f"**Estimated trip cost** — {answers['travellers']} "
                    f"traveller(s), {answers['nights']} nights, before flights")
        fig = px.bar(costs.sort_values("total"), x="total", y="city",
                     orientation="h", color="total", color_continuous_scale=scale)
        fig.update_layout(**layout, xaxis=dict(**axis, title="US$"),
                          yaxis=dict(**axis, title=""), coloraxis_showscale=False)
        fig.update_traces(hovertemplate="%{y}: $%{x:,.0f}<extra></extra>")
        chart(fig)


def tab_explore(df: pd.DataFrame, airports: AirportIndex) -> None:
    html(heading("Explore", "Browse the whole catalogue",
                    "Filter, then look at where things land."))

    c1, c2, c3 = st.columns([1.4, 1, 1.2], gap="medium")
    with c1:
        regions = st.multiselect(
            "Region", REGION_COLS,
            format_func=lambda c: REGION_LABEL[c], placeholder="All regions")
    with c2:
        tiers = st.multiselect("Budget", [1, 2, 3],
                               format_func=lambda v: BUDGET_LABEL[v],
                               placeholder="Any budget")
    with c3:
        lo, hi = st.slider("Average temperature (°C)", -10, 40, (-10, 40))

    view = df
    if regions:
        mask = pd.Series(False, index=df.index)
        for col in regions:
            if col in df.columns:
                mask |= df[col] == 1
        view = view[mask]
    if tiers:
        view = view[view["budget_level_encoded"].astype(int).isin(tiers)]
    view = view[(view["temp_avg_yearly"] >= lo) & (view["temp_avg_yearly"] <= hi)]

    st.caption(f"{len(view):,} of {len(df):,} destinations match")
    if view.empty:
        st.info("Nothing matches those filters. Widen the range or clear a filter.")
        return

    map_tab, table_tab = st.tabs(["Map", "Table"])
    with map_tab:
        with guard("map"):
            st.map(view.rename(columns={"latitude": "lat", "longitude": "lon"}),
                   size=30, color="#22D3EE")
    with table_tab:
        cols = ["city", "country", "temp_avg_yearly", "budget_level_encoded"]
        cols += [c for c in ("name", "iata") if c in view.columns]
        cols += TASTE_KEYS
        table = view[[c for c in cols if c in view.columns]].copy()
        table["budget_level_encoded"] = (table["budget_level_encoded"].astype(int)
                                         .map(BUDGET_LABEL))
        table = table.rename(columns={"temp_avg_yearly": "avg °C",
                                      "budget_level_encoded": "budget",
                                      "name": "airport", "iata": "code"})
        st.dataframe(table, hide_index=True)


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #


def version_bar() -> None:
    """Show the running build and a one-click interaction test, unhidden.

    Which version is actually live has been ambiguous more than once, and a
    hidden expander is easy to miss. This sits at the top of the page so the
    build number and whether clicking works can both be read at a glance.
    """
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.caption(f"Build {BUILD}")
    with right:
        st.session_state.setdefault("_clicks", 0)
        if st.button(f"Self-test: {st.session_state['_clicks']}", key="_selftest",
                     help="Press this. The number must go up. If it does not, "
                          "the page is not reaching the server — reload with "
                          "Ctrl+Shift+R."):
            st.session_state["_clicks"] += 1
            st.rerun()


def diagnostics() -> None:
    """A self-test the user can run in their own browser.

    Tab switching and button clicks are handled by Streamlit's own JavaScript.
    When they do nothing, the cause is in the browser — a stale cached bundle, a
    dropped websocket, or scripts being blocked — none of which is visible from
    the server. This makes the answer checkable in one click.
    """
    with st.expander("Having trouble?"):
        st.markdown(
            "**If the number never changes**, the page is not talking to the "
            "server. In order of likelihood:\n\n"
            "1. You are seeing a cached copy — reload with **Ctrl+Shift+R** "
            "(**Cmd+Shift+R** on a Mac).\n"
            "2. The connection dropped — reload the page.\n"
            "3. An extension or strict privacy setting is blocking scripts — "
            "try a private window or another browser.\n\n"
            f"If it still fails, note the build number above ({BUILD}) "
            "so the running version can be confirmed."
        )


def main() -> None:
    html(THEME_CSS)

    # The opening animation paints over the page that renders beneath it in this
    # same run, then removes itself. It is decorative and declared
    # pointer-events:none throughout, so it cannot intercept a click even if the
    # animation never runs. Shown once per session.
    if not st.session_state.get("seen_splash", False):
        st.session_state["seen_splash"] = True
        html(build_splash())

    html(brand())
    version_bar()

    df = report = airports = None
    with guard("startup") as failed:
        df, report = load_catalogue()
        airports = airport_index(fingerprint(df))
    if failed or df is None:
        st.error("TripWise could not start: the destination catalogue failed to "
                 "load. Check that tripwise_data.csv sits beside app.py.",
                 icon="⚠️")
        return

    home, planner, explore = st.tabs(["  Home  ", "  AI Planner  ", "  Explore  "])
    with home:
        tab_home(df, report, airports)
    with planner:
        tab_planner(df, airports)
    with explore:
        tab_explore(df, airports)


if __name__ == "__main__":
    main()
