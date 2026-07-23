"""
TripWise AI — an AI travel platform (single-file build).

    streamlit run app.py

Generated from the modular project by build_single.py: identical code with the
package inlined, so it deploys without a folder alongside it. Banners mark the
module each section came from; edit the modules, not this file.

Python does the thinking — loading and validating the catalogue, resolving
airports, ranking destinations and deriving costs, seasons and insights.
Everything visible is custom markup styled by the design system.
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
CACHE_VERSION = "3.1.0"

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
  padding:22px 22px 90px 22px !important;
}
[data-testid="stVerticalBlock"]{ gap:0 !important; }
html{ scroll-behavior:smooth; }
html{ scroll-padding-top:24px; }

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
   NAVBAR — a Streamlit column row dressed as a nav bar

   The controls are real widgets rather than anchors, so the styling has to
   reach into Streamlit's markup. Everything below scopes to .tw-navwrap so it
   cannot leak into the buttons on the rest of the page.
   ===================================================================== */
.tw-navwrap{
  position:relative; margin:-14px -12px 6px; padding:10px 18px 8px;
  background:rgba(255,255,255,.72);
  backdrop-filter:blur(20px) saturate(1.6);
  -webkit-backdrop-filter:blur(20px) saturate(1.6);
  border:1px solid rgba(255,255,255,.85);
  border-radius:var(--r-lg); box-shadow:var(--sh-1);
}
.tw-brand{ display:flex; align-items:center; gap:.6rem; }
.tw-brand svg{ width:26px; height:26px; flex:0 0 auto; }
.tw-brand b{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.08rem;
  letter-spacing:-.02em; color:var(--ink); white-space:nowrap;
}

/* in-page section links stay as anchors — they only scroll, never navigate */
a.tw-navlink{
  display:block; text-align:center; text-decoration:none; color:var(--slate);
  font-size:.9rem; font-weight:500; padding:.5rem .4rem; border-radius:99px;
  transition:color .22s var(--ease), background .22s var(--ease);
}
a.tw-navlink:hover{ color:var(--blue); background:rgba(9,16,32,.045); }

/* nav buttons: flat until hovered, gradient pill when active */
.tw-navwrap [data-testid="stButton"] > button{
  width:100%; padding:.5rem .4rem; font-size:.9rem; font-weight:500;
  border-radius:99px; box-shadow:none; border:1px solid transparent;
}
.tw-navwrap [data-testid="stButton"] > button[kind="secondary"]{
  background:transparent; color:var(--slate);
}
.tw-navwrap [data-testid="stButton"] > button[kind="secondary"]:hover{
  background:rgba(9,16,32,.045); color:var(--blue); transform:none;
}
.tw-navwrap [data-testid="stButton"] > button[kind="primary"]{
  background:linear-gradient(96deg,var(--sky),var(--blue));
  color:#fff; box-shadow:0 8px 20px rgba(37,99,235,.28);
}
.tw-navwrap [data-testid="stButton"] > button[kind="primary"]:hover{
  transform:translateY(-1px); box-shadow:0 12px 26px rgba(37,99,235,.36);
}
.tw-navwrap [data-testid="stVerticalBlock"]{ gap:0 !important; }

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

.tw-section{ margin:clamp(64px,9vw,116px) 0 0; scroll-margin-top:24px; }
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
  .tw-navwrap [data-testid="stButton"] > button{ font-size:.8rem; padding:.45rem .2rem; }
  a.tw-navlink{ font-size:.8rem; padding:.45rem .2rem; }
  .tw-facts{ grid-template-columns:repeat(2,1fr); }
}
@media (max-width:560px){
  .tw-brand b{ font-size:.9rem; }
  .tw-navwrap [data-testid="stButton"] > button{ font-size:.72rem; }
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
# HTML COMPONENTS — HTML component builders.
# ==========================================================================

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

# (label, view) — Features/About/Contact are in-page anchors on the home view
NAV_ITEMS: list[tuple[str, str]] = [
    ("Home", "home"),
    ("Features", "#features"),
    ("AI Planner", "planner"),
    ("About", "#about"),
    ("Contact", "#contact"),
]


def brand_mark() -> str:
    """The logo, rendered as markup because it never needs to be clickable."""
    return """
<div class="tw-brand">
<svg viewBox="0 0 24 24" fill="none" stroke="url(#navg)" stroke-width="1.8"
stroke-linecap="round" stroke-linejoin="round">
<defs><linearGradient id="navg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/>
</linearGradient></defs>
<path d="M2 13.4 21 4l-4.6 16.2-4.1-6.1z"/><path d="m12.3 14.1-3.4 6.9-1-5"/>
</svg>
<b>TripWise <span class="tw-grad">AI</span></b>
</div>"""


def nav_open() -> str:
    return '<div class="tw-navwrap">'


def nav_close() -> str:
    return "</div>"


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
      <a class="tw-btn tw-btn--ghost" href="#features">See how it works</a>
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
    <a href="#features">Features</a> &nbsp;·&nbsp;
    <a href="#about">About</a> &nbsp;·&nbsp;
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
  /* purely decorative, so it must never intercept a click even mid-animation */
  pointer-events:none;
  position:fixed; inset:0; z-index:99998; display:grid; place-items:center;
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
# cached resources
# --------------------------------------------------------------------------- #

def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_catalogue(cache_version: str = CACHE_VERSION):
    """Read and clean the catalogue, cached on the data rather than per session.

    `cache_version` is part of the key so a release that changes the shape of
    the report cannot be served a stale object from a previous build.
    """
    return load_catalogue_file()


@st.cache_resource(show_spinner=False)
def fitted_models(fingerprint: str) -> ModelBundle:
    """Scaler and K-Means, fitted once per catalogue.

    `fingerprint` keys the cache to the catalogue's shape rather than to the
    DataFrame, which is unhashable and changes identity on every rerun.
    """
    df, _ = load_catalogue()
    return fit_models(df)


@st.cache_resource(show_spinner=False)
def airport_index(fingerprint: str) -> AirportIndex:
    """Searchable airport index, built once per catalogue."""
    df, _ = load_catalogue()
    return AirportIndex(df)


def fingerprint(df: pd.DataFrame) -> str:
    """A cheap, stable identity for a catalogue, scoped to this release."""
    return (f"{CACHE_VERSION}:{len(df)}x{len(df.columns)}:"
            f"{hash(tuple(sorted(str(c) for c in df.columns)))}")


# --------------------------------------------------------------------------- #
# view models
# --------------------------------------------------------------------------- #

def card_payload(row, prefs: dict, airports: AirportIndex) -> dict:
    """Flatten a matched row into exactly what the card template needs."""
    airport = airports.resolve(row)
    tastes = tastes_of(row)
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
        "art": destination_art(str(row.get("city", "x")), tastes, temp),
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


def data_notice(report: CatalogueReport, airports: AirportIndex,
                df: pd.DataFrame) -> None:
    """Tell the operator what the catalogue can and cannot support.

    Every field is read through `report.get`, because Streamlit can serve a
    cached report built by an earlier version of the class, which would not
    carry fields added since.
    """
    source = report.get("source", "the catalogue")
    rows_out = report.get("rows_out", len(df))

    if not report.get("is_real", True):
        missing = report.get("missing_required", [])
        if report.get("error", ""):
            detail = report.get("error", "")
        elif missing:
            detail = "Missing required columns: " + ", ".join(missing) + "."
        else:
            detail = f"{CSV_NAME} was not found beside app.py."
        st.info(f"Running on the built-in demo catalogue. {detail} "
                "Export final_df from the notebook to use your own data.", icon="ℹ️")
        return

    if not len(airports):
        st.warning(
            f"**No airport data in {source}.** The catalogue loaded "
            f"{rows_out:,} destinations, but no column holding airport names or "
            "codes was found, so no airports can be shown.",
            icon="⚠️",
        )
        with st.expander("How to fix this"):
            st.markdown(
                "TripWise looks for a **`name`** column (airport name) and an "
                "**`iata`** column (airport code), and also accepts the common "
                "alternative spellings `airport_name`, `airport`, `IATA` and "
                "`iata_code`.\n\n"
                "These columns come from the airports merge in the notebook. If "
                "the export selected only the model features, they were dropped. "
                "Re-export keeping them, then reload this page."
            )
            seen = report.get("columns_seen", [])
            if seen:
                st.caption(f"Columns found in {source}:")
                st.code(", ".join(str(c) for c in seen), language="text")
        return

    direct, total = coverage(df, airports)
    if total and direct < total:
        st.caption(
            f"Catalogue: {rows_out:,} destinations from {source}. "
            f"{direct:,} carry a matched airport; the rest resolve to their nearest "
            f"airport from the {len(airports):,} in the dataset."
        )


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #


def navigate(view: str) -> None:
    """Switch view without a page reload, preserving session state."""
    st.query_params["view"] = view
    st.rerun()


def render_nav(active: str) -> None:
    """The top bar.

    Navigation runs on real Streamlit buttons rather than anchor tags. Anchors
    inside st.markdown proved unreliable in the deployed environment, and this
    route is better regardless: setting a query param and rerunning switches
    view in place instead of forcing a full page load, so session state and
    cached models survive.
    """
    html(nav_open())
    brand, links, cta = st.columns([3, 7, 2.4], vertical_alignment="center")

    with brand:
        html(brand_mark())

    with links:
        cols = st.columns(len(NAV_ITEMS))
        for col, (label, target) in zip(cols, NAV_ITEMS):
            with col:
                if target.startswith("#"):
                    # in-page section: a link is right here, no navigation needed
                    html(f'<a class="tw-navlink" href="{target}">{label}</a>')
                elif st.button(label, key=f"nav_{target}",
                               type="primary" if active == target else "secondary"):
                    navigate(target)

    with cta:
        if st.button("Start planning", key="nav_cta", type="primary"):
            navigate("planner")

    html(nav_close())


def page_home(df: pd.DataFrame, report: CatalogueReport,
              airports: AirportIndex) -> None:
    render_nav("home")
    html(hero(catalogue_stats(df)))
    cta, _ = st.columns([1, 3])
    with cta:
        if st.button("Start planning  →", key="hero_cta", type="primary"):
            navigate("planner")
    data_notice(report, airports, df)

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
         + feature_card("plane", "Nearest airport, always",
                           "Every destination resolves to a gateway airport — by direct "
                           "match where one exists, by distance otherwise.", 4)
         + feature_card("layers", "Genuinely different options",
                           "Results are spread across distinct destination profiles, so "
                           "a shortlist offers real alternatives.", 5)
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


def collect_answers() -> tuple[dict, bool]:
    """Render the planner form and return the raw answers plus submit state."""
    with st.form("planner", border=False):
        html('<div class="tw-panel">')
        html('<div class="tw-fieldset__title">What you care about</div>')

        answers: dict = {}
        cols = st.columns(3, gap="large")
        for i, (key, label, hint) in enumerate(TASTES):
            with cols[i % 3]:
                answers[key] = st.slider(label, TASTE_MIN, TASTE_MAX, 3,
                                         help=hint, key=f"t_{key}")

        html('<div class="tw-fieldset__title">Budget and climate</div>')
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            answers["budget"] = st.select_slider(
                "Budget level", options=[1, 2, 3], value=2,
                format_func=lambda v: BUDGET_LABEL[v])
        with c2:
            answers["temp"] = st.slider("Preferred average temperature (°C)", -5, 40, 22)
        with c3:
            answers["stars"] = st.slider("Minimum accommodation standard",
                                         STARS_MIN, STARS_MAX, 4.0, 0.5)

        html('<div class="tw-fieldset__title">Where and how long</div>')
        c4, c5, c6 = st.columns(3, gap="large")
        with c4:
            names = ["Anywhere"] + [REGION_LABEL[c] for c in REGION_COLS]
            picked = st.selectbox("Preferred region", names)
            answers["region"] = next(
                (c for c in REGION_COLS if REGION_LABEL[c] == picked), None)
        with c5:
            length = st.radio("Trip length", ["Short trip", "One week"], horizontal=True)
            answers["trip_length"] = "short" if length == "Short trip" else "week"
        with c6:
            answers["travellers"] = st.number_input(
                "Travellers", TRAVELLERS_MIN, TRAVELLERS_MAX, 2)

        answers["nights"] = st.slider("Nights", NIGHTS_MIN, 30,
                                      4 if answers["trip_length"] == "short" else 7)
        answers["needs_airport"] = st.checkbox(
            "Prefer destinations with a nearby airport", value=True)

        html("</div>")
        st.markdown("<div style='height:1.3rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Find my destinations  →", type="primary")

    return answers, submitted


def page_planner(df: pd.DataFrame, report: CatalogueReport,
                 airports: AirportIndex) -> None:
    render_nav("planner")
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
        html(footer())
        return

    blocking = validate_catalogue(df)
    if blocking:
        st.error(blocking, icon="⚠️")
        html(footer())
        return

    checked = validate_answers(raw_answers)
    for notice in checked.notices:
        st.warning(notice, icon="⚠️")
    answers = checked.answers

    pool, pool_notices = validate_pool(df, answers)
    for notice in pool_notices:
        st.info(notice, icon="ℹ️")

    result = None
    with guard("recommendation") as failure:
        bundle = fitted_models(fingerprint(df))
        prefs = build_preference_vector(df, answers)
        with st.spinner("Ranking destinations…"):
            result = recommend(
                df, bundle, prefs,
                top_n=DEFAULT_TOP_N,
                pool_index=pool.index.to_numpy(),
            )
    if failure or result is None:
        st.error("Something went wrong while ranking destinations. "
                 "Adjust a preference and try again.", icon="⚠️")
        html(footer())
        return

    if result.empty:
        st.warning("No destinations matched. Try widening the region or climate.",
                   icon="ℹ️")
        html(footer())
        return

    render_results(result, prefs, answers, airports)


def render_results(result: Recommendation, prefs: dict,
                   answers: dict, airports: AirportIndex) -> None:
    frame = result.frame

    html(section_head(
        "Your matches", "Where you should go",
        f"Ranked against your answers from {result.considered:,} destinations. "
        f"Estimates assume {answers['travellers']} traveller(s) over "
        f"{answers['nights']} nights."))

    cards = []
    for i, (_, row) in enumerate(frame.iterrows()):
        with guard(f"card for {row.get('city')}"):
            cards.append(destination_card(card_payload(row, prefs, airports),
                                             delay=(i % 6) + 1))
    html(f'<div class="tw-grid tw-grid--3">{"".join(cards)}</div>' + close_section())

    readings: list = []
    with guard("insights"):
        readings = travel_insights(frame, prefs, answers, result.profiles)
    if readings:
        html(section_head(
            "AI insights", "What your results say",
            "Read from the shortlist as a whole, not any single destination."))
        html('<div class="tw-grid tw-grid--2">'
             + "".join(insight_card(t, b, (i % 6) + 1)
                       for i, (t, b) in enumerate(readings))
             + "</div>" + close_section())

    with guard("charts") as failed:
        render_charts(frame, answers)
    if failed:
        st.caption("Charts are unavailable for this result set.")

    html(footer())


def chart(fig) -> None:
    """Render a figure full width across Streamlit versions."""
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def render_charts(frame: pd.DataFrame, answers: dict) -> None:
    import plotly.express as px

    html(section_head("Comparison", "Your shortlist side by side",
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
        costs = frame.assign(daily=[daily_cost(r) for _, r in frame.iterrows()])
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

    html(close_section())


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #

def main() -> None:
    params = st.query_params
    view = params.get("view", "home")
    if isinstance(view, (list, tuple)):        # some contexts hand back a list
        view = view[0] if view else "home"
    view = str(view).strip().lower()
    if view not in ("home", "planner"):
        view = "home"

    html(THEME_CSS)

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
