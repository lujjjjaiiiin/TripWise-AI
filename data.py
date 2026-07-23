"""Loading and validating the destination catalogue.

The app expects `tripwise_data.csv` — the export of the notebook's `final_df` —
beside app.py. If it is absent or missing required columns, a small built-in
catalogue keeps the interface usable and the UI says so plainly rather than
presenting demo numbers as if they were real.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .engine import REGION_COLS, TASTE_KEYS

CSV_NAME = "tripwise_data.csv"

REQUIRED = ["city", "country", "latitude", "longitude", "temp_avg_yearly",
            "budget_level_encoded"] + TASTE_KEYS

# city, country, region col, iata, airport, lat, lon,
# cul adv nat bea nig cui wel urb sec, budget, temp, stars
_DEMO = [
    ("Paris", "France", "region_europe", "CDG", "Charles de Gaulle", 48.86, 2.35, 5,2,2,1,4,5,3,5,1, 3, 12.3, 4.2),
    ("Barcelona", "Spain", "region_europe", "BCN", "Barcelona El Prat", 41.39, 2.17, 4,3,2,4,5,5,3,4,1, 2, 17.1, 4.0),
    ("Reykjavik", "Iceland", "region_europe", "KEF", "Keflavik International", 64.15, -21.94, 3,5,5,1,3,3,4,2,5, 3, 5.2, 4.0),
    ("Santorini", "Greece", "region_europe", "JTR", "Santorini National", 36.39, 25.46, 3,2,3,5,3,4,5,2,4, 3, 18.4, 4.4),
    ("Tokyo", "Japan", "region_asia", "HND", "Haneda", 35.68, 139.69, 5,2,2,1,5,5,3,5,1, 3, 16.0, 4.3),
    ("Bali", "Indonesia", "region_asia", "DPS", "Ngurah Rai International", -8.41, 115.19, 4,4,5,5,4,4,5,2,4, 1, 26.6, 4.1),
    ("Bangkok", "Thailand", "region_asia", "BKK", "Suvarnabhumi", 13.76, 100.50, 4,3,2,2,5,5,4,5,1, 1, 28.4, 4.0),
    ("Maldives", "Maldives", "region_asia", "MLE", "Velana International", 4.18, 73.51, 1,3,4,5,1,3,5,1,5, 3, 28.1, 4.6),
    ("Dubai", "UAE", "region_middle_east", "DXB", "Dubai International", 25.20, 55.27, 2,4,1,4,4,4,4,5,1, 3, 28.0, 4.5),
    ("AlUla", "Saudi Arabia", "region_middle_east", "ULH", "Prince Abdul Majeed", 26.61, 37.92, 5,5,5,1,1,3,4,1,5, 3, 26.0, 4.4),
    ("Istanbul", "Turkey", "region_middle_east", "IST", "Istanbul Airport", 41.01, 28.98, 5,2,2,2,4,5,3,5,1, 2, 15.0, 4.1),
    ("Marrakesh", "Morocco", "region_africa", "RAK", "Marrakesh Menara", 31.63, -7.99, 5,4,3,1,3,5,4,3,2, 2, 20.3, 4.0),
    ("Cape Town", "South Africa", "region_africa", "CPT", "Cape Town International", -33.92, 18.42, 4,5,5,5,4,5,3,4,3, 2, 17.0, 4.2),
    ("Zanzibar", "Tanzania", "region_africa", "ZNZ", "Abeid Amani Karume", -6.16, 39.20, 3,4,4,5,2,3,4,1,4, 2, 26.4, 4.0),
    ("New York", "United States", "region_north_america", "JFK", "John F. Kennedy", 40.71, -74.01, 5,2,2,2,5,5,3,5,1, 3, 12.9, 4.1),
    ("Banff", "Canada", "region_north_america", "YYC", "Calgary International", 51.18, -115.57, 2,5,5,1,2,3,4,1,5, 3, 3.0, 4.3),
    ("Mexico City", "Mexico", "region_north_america", "MEX", "Benito Juarez", 19.43, -99.13, 5,3,3,1,5,5,3,5,1, 2, 17.0, 4.0),
    ("Rio de Janeiro", "Brazil", "region_south_america", "GIG", "Galeao", -22.91, -43.17, 3,4,4,5,5,4,3,4,2, 2, 23.8, 4.0),
    ("Cusco", "Peru", "region_south_america", "CUZ", "Alejandro Velasco Astete", -13.53, -71.97, 5,5,5,1,3,4,3,2,4, 1, 12.3, 3.9),
    ("Patagonia", "Chile", "region_south_america", "PNT", "Teniente Julio Gallardo", -51.73, -72.51, 2,5,5,1,1,3,3,1,5, 3, 6.5, 4.0),
    ("Sydney", "Australia", "region_oceania", "SYD", "Kingsford Smith", -33.87, 151.21, 3,4,4,5,4,5,4,5,2, 3, 18.3, 4.2),
    ("Queenstown", "New Zealand", "region_oceania", "ZQN", "Queenstown Airport", -45.03, 168.66, 2,5,5,2,3,4,4,2,4, 3, 10.6, 4.3),
]


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
        for col in REGION_COLS:
            row[col] = 1 if col == region else 0
        rows.append(row)
    return pd.DataFrame(rows)


def load() -> tuple[pd.DataFrame, bool, list[str]]:
    """Return (catalogue, is_real_data, missing_columns)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CSV_NAME)
    if not os.path.exists(path):
        path = CSV_NAME

    if os.path.exists(path):
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if not missing:
            for col in REGION_COLS + ["has_airport", "is_short_trip", "is_one_week",
                                      "HotelRating_encoded", "rating_was_unknown"]:
                if col not in df.columns:
                    df[col] = 0
            optional = [c for c in ("name", "iata", "HotelName", "Attractions")
                        if c not in df.columns]
            return df, True, optional
        return _demo_frame(), False, missing

    return _demo_frame(), False, []


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
