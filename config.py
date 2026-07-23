"""Airport resolution.

Why this module exists
----------------------
The notebook joins airports onto destinations on an exact ``(city, country)``
string match. That join misses whenever the two sources spell a place
differently — ``AlUla`` against ``Al-'Ula``, ``Cape Town`` against ``CAPE
TOWN`` — and unmatched rows are then filled with the literal string
``"Unknown"``. Worse, ``has_airport`` is computed *before* that fill, so a row
can hold a perfectly good airport name while its flag reads 0.

The old display code gated on that flag, which is why nearly every card
reported no airport. Dropping the gate is necessary but not sufficient: the
join itself is lossy.

Resolution here is a three-step ladder, so a valid destination always gets an
answer:

1. **Direct** — the row's own join succeeded, use it.
2. **Normalised name** — retry with case, accents and punctuation stripped,
   which recovers the spelling mismatches the exact join dropped.
3. **Geographic** — the closest airport in the catalogue by great-circle
   distance. Destinations carry coordinates, so this always resolves.

Distance is reported alongside the name, which keeps step 3 honest: a gateway
80 km away and one 3,000 km away are both shown, and the reader can tell them
apart.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AIRPORT_MAX_KM, NULL_TOKENS
from .errors import log, safe, to_float

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
