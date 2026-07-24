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

import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
CACHE_VERSION = "7.1.0"

# Shown in the app so the running build can be identified from a screenshot.
BUILD = "7.1.0"

# ==========================================================================
# TRANSLATION — Interface translation.
# ==========================================================================

LANGUAGES = {"en": "English", "ar": "العربية"}
DEFAULT = "en"

_current = DEFAULT


def set_language(code: str) -> None:
    global _current
    _current = code if code in LANGUAGES else DEFAULT


def language() -> str:
    return _current


def is_rtl() -> bool:
    return _current == "ar"


STRINGS: dict[str, dict[str, str]] = {
    # ---- navigation -------------------------------------------------------
    "nav.home": {"en": "Home", "ar": "الرئيسية"},
    "nav.planner": {"en": "AI Planner", "ar": "المخطِّط الذكي"},
    "nav.explore": {"en": "Explore", "ar": "استكشاف"},
    "nav.section": {"en": "Section", "ar": "القسم"},
    "nav.language": {"en": "Language", "ar": "اللغة"},

    # ---- hero -------------------------------------------------------------
    "hero.badge": {"en": "✦ AI-powered travel planning",
                   "ar": "✦ تخطيط رحلات بالذكاء الاصطناعي"},
    "hero.title_1": {"en": "Plan smarter with", "ar": "خطّط بذكاء مع"},
    "hero.lede": {
        "en": "Describe how you like to travel. TripWise ranks real destinations "
              "against your answers — with cost, climate, season and the nearest "
              "airport already worked out.",
        "ar": "صِف طريقتك في السفر، ويرتّب TripWise وجهات حقيقية حسب إجاباتك — "
              "مع التكلفة والمناخ وأفضل موسم وأقرب مطار، محسوبة مسبقًا.",
    },
    "hero.cta": {"en": "Start planning  →", "ar": "ابدأ التخطيط  ←"},
    "hero.tagline": {"en": "Intelligent destination matching",
                     "ar": "مطابقة ذكية للوجهات"},

    # ---- home stats -------------------------------------------------------
    "stat.destinations": {"en": "Destinations ranked", "ar": "وجهة مُرتَّبة"},
    "stat.countries": {"en": "Countries covered", "ar": "دولة"},
    "stat.regions": {"en": "World regions", "ar": "منطقة عالمية"},
    "stat.dimensions": {"en": "Travel dimensions", "ar": "معيار سفر"},

    # ---- home cards -------------------------------------------------------
    "home.c1.title": {"en": "Tell it how you travel", "ar": "أخبِرنا كيف تسافر"},
    "home.c1.body": {"en": "Nine preferences. About a minute.",
                     "ar": "تسعة تفضيلات، في دقيقة تقريبًا."},
    "home.c2.title": {"en": "Get a ranked shortlist", "ar": "احصل على قائمة مرتّبة"},
    "home.c2.body": {"en": "Real destinations, scored against your answers.",
                     "ar": "وجهات حقيقية، مُقيَّمة حسب إجاباتك."},
    "home.c3.title": {"en": "See the whole picture", "ar": "شاهد الصورة كاملة"},
    "home.c3.body": {"en": "Cost, season and nearest airport for each.",
                     "ar": "التكلفة والموسم وأقرب مطار لكل وجهة."},

    # ---- planner ----------------------------------------------------------
    "plan.eyebrow": {"en": "AI Planner", "ar": "المخطِّط الذكي"},
    "plan.title": {"en": "Tell us how you travel", "ar": "أخبِرنا كيف تسافر"},
    "plan.sub": {"en": "1 means indifferent, 5 means it would shape the whole trip.",
                 "ar": "١ يعني لا يهمّك، و٥ يعني أنه يشكّل الرحلة كلها."},
    "plan.group.tastes": {"en": "What you care about", "ar": "ما الذي يهمّك"},
    "plan.group.budget": {"en": "Budget and climate", "ar": "الميزانية والمناخ"},
    "plan.group.where": {"en": "Where and how long", "ar": "الوجهة والمدة"},
    "plan.budget": {"en": "Low Budget", "ar": "ميزانية محدودة"},
    "plan.temp": {"en": "Preferred average temperature (°C)",
                  "ar": "متوسط الحرارة المفضّل (°م)"},
    "plan.stars": {"en": "Minimum accommodation standard",
                   "ar": "أدنى مستوى للإقامة"},
    "plan.region": {"en": "Preferred region", "ar": "المنطقة المفضّلة"},
    "plan.anywhere": {"en": "Anywhere", "ar": "أي مكان"},
    "plan.nights": {"en": "Nights", "ar": "عدد الليالي"},
    "plan.travellers": {"en": "Travellers", "ar": "عدد المسافرين"},
    "plan.airport": {"en": "Prefer destinations with a nearby airport",
                     "ar": "أفضّل وجهات قريبة من مطار"},
    "plan.submit": {"en": "Find my destinations  →", "ar": "اعرض وجهاتي  ←"},
    "plan.hint": {"en": "Set your preferences above, then press the button.",
                  "ar": "اضبط تفضيلاتك بالأعلى ثم اضغط الزر."},

    # ---- tastes -----------------------------------------------------------
    "taste.culture": {"en": "Culture", "ar": "الثقافة"},
    "taste.adventure": {"en": "Adventure", "ar": "المغامرة"},
    "taste.nature": {"en": "Nature", "ar": "الطبيعة"},
    "taste.beaches": {"en": "Beaches", "ar": "الشواطئ"},
    "taste.nightlife": {"en": "Nightlife", "ar": "الحياة الليلية"},
    "taste.cuisine": {"en": "Food", "ar": "الطعام"},
    "taste.wellness": {"en": "Wellness", "ar": "الاسترخاء"},
    "taste.urban": {"en": "City life", "ar": "حياة المدن"},
    "taste.seclusion": {"en": "Quiet", "ar": "الهدوء"},
    "taste.balanced": {"en": "Balanced", "ar": "متوازن"},

    "hint.culture": {"en": "Museums, old towns, landmarks",
                     "ar": "المتاحف والمدن القديمة والمعالم"},
    "hint.adventure": {"en": "Hiking, diving, adrenaline",
                       "ar": "المشي والغوص والإثارة"},
    "hint.nature": {"en": "Parks, mountains, wildlife",
                    "ar": "الحدائق والجبال والحياة البرية"},
    "hint.beaches": {"en": "Coastline and swimming", "ar": "السواحل والسباحة"},
    "hint.nightlife": {"en": "Bars, music, late nights",
                       "ar": "المقاهي والموسيقى والسهر"},
    "hint.cuisine": {"en": "Restaurants and street food",
                     "ar": "المطاعم وأكل الشوارع"},
    "hint.wellness": {"en": "Spas, retreats, slow days",
                      "ar": "المنتجعات والاسترخاء"},
    "hint.urban": {"en": "Design, shopping, skylines",
                   "ar": "التصميم والتسوّق وناطحات السحاب"},
    "hint.seclusion": {"en": "Few crowds, room to breathe",
                       "ar": "ازدحام أقل ومساحة أوسع"},

    # ---- regions ----------------------------------------------------------
    "region.region_africa": {"en": "Africa", "ar": "أفريقيا"},
    "region.region_asia": {"en": "Asia", "ar": "آسيا"},
    "region.region_europe": {"en": "Europe", "ar": "أوروبا"},
    "region.region_middle_east": {"en": "Middle East", "ar": "الشرق الأوسط"},
    "region.region_north_america": {"en": "North America", "ar": "أمريكا الشمالية"},
    "region.region_oceania": {"en": "Oceania", "ar": "أوقيانوسيا"},
    "region.region_south_america": {"en": "South America", "ar": "أمريكا الجنوبية"},

    # ---- budget tiers -----------------------------------------------------
    "budget.1": {"en": "Budget", "ar": "اقتصادية"},
    "budget.2": {"en": "Mid-range", "ar": "متوسطة"},
    "budget.3": {"en": "Luxury", "ar": "فاخرة"},

    # ---- results ----------------------------------------------------------
    "res.eyebrow": {"en": "Your matches", "ar": "نتائجك"},
    "res.title": {"en": "Where you should go", "ar": "إلى أين تذهب"},
    "res.sub": {"en": "Ranked from {n} destinations for {p} traveller(s) over {k} nights.",
                "ar": "مرتّبة من {n} وجهة، لـ{p} مسافر خلال {k} ليالٍ."},
    "res.metric.match": {"en": "Best match", "ar": "أعلى تطابق"},
    "res.metric.from": {"en": "From, per trip", "ar": "تبدأ من، للرحلة"},
    "res.metric.profiles": {"en": "Distinct profiles", "ar": "أنماط مختلفة"},
    "res.metric.climate": {"en": "Average climate", "ar": "متوسط المناخ"},
    "res.none": {"en": "No destinations matched. Try widening the region or climate.",
                 "ar": "لا توجد نتائج مطابقة. جرّب توسيع المنطقة أو المناخ."},
    "res.error": {"en": "Something went wrong while ranking. Adjust a preference and try again.",
                  "ar": "حدث خطأ أثناء الترتيب. عدّل أحد التفضيلات وحاول مجددًا."},

    # ---- destination card -------------------------------------------------
    "card.match": {"en": "match", "ar": "تطابق"},
    "card.budget": {"en": "Budget", "ar": "الميزانية"},
    "card.perday": {"en": "Per day", "ar": "لليوم"},
    "card.climate": {"en": "Climate", "ar": "المناخ"},
    "card.airport": {"en": "Nearest airport", "ar": "أقرب مطار"},
    "card.season": {"en": "Best season", "ar": "أفضل موسم"},
    "card.weather": {"en": "Weather", "ar": "الطقس"},
    "card.stay": {"en": "Suggested stay", "ar": "إقامة مقترحة"},
    "card.nearby": {"en": "Nearby", "ar": "معالم قريبة"},
    "card.tip": {"en": "Tip", "ar": "نصيحة"},
    "card.noairport": {"en": "None recorded in the catalogue",
                       "ar": "غير مُسجَّل في قاعدة البيانات"},

    # ---- map, insights, charts -------------------------------------------
    "map.eyebrow": {"en": "On the map", "ar": "على الخريطة"},
    "map.title": {"en": "Your shortlist, plotted", "ar": "وجهاتك على الخريطة"},
    "ins.eyebrow": {"en": "AI insights", "ar": "قراءات ذكية"},
    "ins.title": {"en": "What your results say", "ar": "ماذا تقول نتائجك"},
    "chart.eyebrow": {"en": "Comparison", "ar": "مقارنة"},
    "chart.title": {"en": "Your shortlist side by side", "ar": "وجهاتك جنبًا إلى جنب"},
    "chart.match": {"en": "Match strength", "ar": "قوة التطابق"},
    "chart.cost": {"en": "Estimated trip cost", "ar": "التكلفة التقديرية"},
    "chart.cost_sub": {"en": "{p} traveller(s), {k} nights, before flights",
                       "ar": "{p} مسافر، {k} ليالٍ، قبل الطيران"},
    "chart.matchpct": {"en": "Match %", "ar": "نسبة التطابق"},
    "chart.usd": {"en": "US$", "ar": "دولار"},

    # ---- explore ----------------------------------------------------------
    "exp.eyebrow": {"en": "Explore", "ar": "استكشاف"},
    "exp.title": {"en": "Browse the whole catalogue", "ar": "تصفّح كل الوجهات"},
    "exp.region": {"en": "Region", "ar": "المنطقة"},
    "exp.allregions": {"en": "All regions", "ar": "كل المناطق"},
    "exp.budget": {"en": "Budget", "ar": "الميزانية"},
    "exp.anybudget": {"en": "Any budget", "ar": "أي ميزانية"},
    "exp.temp": {"en": "Average temperature (°C)", "ar": "متوسط الحرارة (°م)"},
    "exp.count": {"en": "{a} of {b} destinations match",
                  "ar": "{a} من {b} وجهة مطابقة"},
    "exp.empty": {"en": "Nothing matches those filters. Widen the range or clear a filter.",
                  "ar": "لا نتائج بهذه الفلاتر. وسّع النطاق أو أزِل أحدها."},
    "exp.zoom": {"en": "Zoom to", "ar": "تكبير إلى"},
    "exp.world": {"en": "Whole world", "ar": "العالم كله"},
    "exp.maphint": {"en": "Drag to pan · scroll to zoom · click a country",
                    "ar": "اسحب للتحريك · مرّر للتكبير · اضغط دولة للتفاصيل"},
    "exp.dest": {"en": "destinations", "ar": "وجهة"},
    "exp.bestmatch": {"en": "best match", "ar": "أعلى تطابق"},
    "exp.avgday": {"en": "avg / day", "ar": "متوسط اليوم"},
    "exp.avgtemp": {"en": "avg temp", "ar": "متوسط الحرارة"},
    "exp.table": {"en": "See the data behind the map", "ar": "عرض البيانات"},
    "exp.col.city": {"en": "city", "ar": "المدينة"},
    "exp.col.country": {"en": "country", "ar": "الدولة"},
    "exp.col.temp": {"en": "avg °C", "ar": "متوسط °م"},
    "exp.col.budget": {"en": "budget", "ar": "الميزانية"},
    "exp.col.airport": {"en": "airport", "ar": "المطار"},
    "exp.col.code": {"en": "code", "ar": "الرمز"},

    # ---- footer -----------------------------------------------------------
    "foot.note": {
        "en": "Destination matching, cost estimates and climate guidance from a "
              "curated travel catalogue. Figures are planning estimates — confirm "
              "prices and seasons before booking.",
        "ar": "مطابقة الوجهات وتقديرات التكلفة وإرشادات المناخ من قاعدة بيانات "
              "سياحية منسّقة. الأرقام تقديرية للتخطيط — تأكّد من الأسعار والمواسم "
              "قبل الحجز.",
    },

    # ---- seasons (engine phrases) ----------------------------------------
    "season.tropics": {"en": "December to March, outside the rains",
                       "ar": "من ديسمبر إلى مارس، خارج موسم الأمطار"},
    "season.hot_north": {"en": "November to March", "ar": "من نوفمبر إلى مارس"},
    "season.hot_south": {"en": "May to September", "ar": "من مايو إلى سبتمبر"},
    "season.cold": {"en": "June to August, or December for snow",
                    "ar": "من يونيو إلى أغسطس، أو ديسمبر للثلوج"},
    "season.mild_north": {"en": "April to June and September to October",
                          "ar": "من أبريل إلى يونيو ومن سبتمبر إلى أكتوبر"},
    "season.mild_south": {"en": "October to December and March to May",
                          "ar": "من أكتوبر إلى ديسمبر ومن مارس إلى مايو"},
    "season.year": {"en": "Year-round", "ar": "طوال السنة"},

    # ---- climate (engine phrases) ----------------------------------------
    "clim.hot": {"en": "Hot year-round, averaging {t}°C",
                 "ar": "حار طوال السنة، بمتوسط {t}°م"},
    "clim.warm": {"en": "Warm and settled, around {t}°C",
                  "ar": "دافئ ومستقر، حوالي {t}°م"},
    "clim.mild": {"en": "Mild, averaging {t}°C", "ar": "معتدل، بمتوسط {t}°م"},
    "clim.cool": {"en": "Cool, around {t}°C — pack layers",
                  "ar": "بارد نسبيًا، حوالي {t}°م — خذ ملابس دافئة"},
    "clim.cold": {"en": "Cold, averaging {t}°C", "ar": "بارد، بمتوسط {t}°م"},
    "clim.na": {"en": "Climate data unavailable", "ar": "بيانات المناخ غير متوفرة"},

    # ---- tips (engine phrases) -------------------------------------------
    "tip.hot": {"en": "Plan outdoor time early morning or after sunset — midday heat is punishing.",
                "ar": "اجعل الأنشطة الخارجية صباحًا أو بعد الغروب — حرّ الظهيرة شديد."},
    "tip.cold": {"en": "Daylight is short in winter; book the outdoor activities first.",
                 "ar": "النهار قصير شتاءً؛ احجز الأنشطة الخارجية أولًا."},
    "tip.quiet": {"en": "Getting around is easier with your own transport — arrange it before arrival.",
                  "ar": "التنقّل أسهل بسيارة خاصة — رتّبها قبل الوصول."},
    "tip.night": {"en": "Stay central: the saving on taxis usually beats the cheaper outer hotels.",
                  "ar": "أقم في المركز: ما توفّره من أجرة النقل يفوق فرق سعر الفنادق البعيدة."},
    "tip.culture": {"en": "Buy museum passes online — the queues are the real cost, not the ticket.",
                    "ar": "احجز تذاكر المتاحف عبر الإنترنت — الطوابير أغلى من التذكرة."},
    "tip.flight": {"en": "Flights into {code} are cheapest midweek; avoid Friday and Sunday departures.",
                   "ar": "الرحلات إلى {code} أرخص منتصف الأسبوع؛ تجنّب الجمعة والأحد."},
    "tip.default": {"en": "Book accommodation before flights — availability moves faster than airfare.",
                    "ar": "احجز الإقامة قبل الطيران — التوفّر ينفد أسرع من تغيّر الأسعار."},

    # ---- insight cards ---------------------------------------------------
    "insight.profile.h": {"en": "Your travel profile", "ar": "ملفّك في السفر"},
    "insight.profile.some": {
        "en": "You lean towards {what} at a {tier} budget. That combination is "
              "what ranked these destinations, not their general popularity.",
        "ar": "تميل إلى {what} بميزانية {tier}. هذا المزيج هو ما رتّب هذه "
              "الوجهات، لا شهرتها العامة.",
    },
    "insight.profile.none": {
        "en": "You kept most preferences balanced at a {tier} budget, so the "
              "ranking favours destinations that do many things well rather "
              "than specialising.",
        "ar": "أبقيت أغلب تفضيلاتك متوازنة بميزانية {tier}، فالترتيب يُفضّل "
              "الوجهات الجيدة في أمور كثيرة على المتخصّصة في أمر واحد.",
    },
    "insight.climate.h": {"en": "Climate fit", "ar": "ملاءمة المناخ"},
    "insight.climate.b": {
        "en": "{close} of your {n} matches sit within 4°C of the {target}°C you "
              "asked for. The spread runs {lo}°C to {hi}°C.",
        "ar": "{close} من {n} نتائج ضمن ٤ درجات من {target}°م التي طلبتها. "
              "المدى بين {lo}°م و{hi}°م.",
    },
    "insight.cost.h": {"en": "What it costs", "ar": "التكلفة"},
    "insight.cost.b": {
        "en": "Daily spend across these matches runs ${lo}–${hi} per person. "
              "Over {nights} nights that is roughly ${tlo}–${thi} each, before "
              "flights.",
        "ar": "الإنفاق اليومي بين ${lo} و${hi} للشخص. على مدى {nights} ليالٍ "
              "يصبح نحو ${tlo} إلى ${thi} للفرد، قبل الطيران.",
    },
    "insight.cluster.h": {"en": "Where they cluster", "ar": "أين تتجمّع"},
    "insight.cluster.one": {
        "en": "{share} of {n} matches are in {top}. Booking two of them into one "
              "trip is usually cheaper than two separate journeys.",
        "ar": "{share} من {n} نتائج في {top}. ضمّ اثنتين منها في رحلة واحدة "
              "أرخص عادةً من رحلتين منفصلتين.",
    },
    "insight.cluster.many": {
        "en": "Your matches spread across {k} regions, so there is no obvious "
              "way to combine them — pick on preference, not logistics.",
        "ar": "نتائجك موزّعة على {k} مناطق، فلا توجد طريقة واضحة لدمجها — "
              "اختر بناءً على التفضيل لا على الترتيبات.",
    },
    "insight.profiles.h": {"en": "Distinct options", "ar": "خيارات متمايزة"},
    "insight.profiles.b": {
        "en": "Your shortlist covers {k} different destination profiles — "
              "{listed} — so these are genuine alternatives rather than "
              "variations on one idea.",
        "ar": "قائمتك تغطّي {k} أنماط وجهات مختلفة — {listed} — أي أنها بدائل "
              "حقيقية لا صيغًا مختلفة لفكرة واحدة.",
    },
    "insight.single.h": {"en": "A clear direction", "ar": "اتجاه واضح"},
    "insight.single.b": {
        "en": "Every match falls into the same profile ({p}), which means your "
              "preferences point somewhere specific. Choose on cost and season.",
        "ar": "كل النتائج ضمن النمط نفسه ({p})، ما يعني أن تفضيلاتك تشير إلى "
              "وجهة محدّدة. اختر بناءً على التكلفة والموسم.",
    },
    "insight.air.h": {"en": "Getting there", "ar": "الوصول"},
    "insight.air.b": {
        "en": "{k} of these have no major airport recorded in the dataset, "
              "which usually means a connecting flight plus ground transfer.",
        "ar": "{k} منها بلا مطار رئيسي مسجّل في البيانات، ما يعني عادةً رحلة "
              "ترانزيت مع انتقال برّي.",
    },
    "insight.and": {"en": " and ", "ar": " و"},
    "list.sep": {"en": ", ", "ar": "، "},
    "profile.join": {"en": " & ", "ar": " و"},
    "profile.allrounder": {"en": "All-rounder", "ar": "متعدّد الجوانب"},

    # ---- data notices -----------------------------------------------------
    "note.demo": {"en": "Running on the built-in demo catalogue.",
                  "ar": "يعمل على قاعدة البيانات التجريبية المدمجة."},
    "note.coverage": {"en": "{rows} destinations. {direct} carry a matched airport; "
                            "the rest resolve to the nearest of {total} known airports.",
                      "ar": "{rows} وجهة. {direct} منها لها مطار مطابق، والبقية تُربَط "
                            "بأقرب مطار من أصل {total}."},
}


def t(key: str, **fields) -> str:
    """The phrase for the active language, formatted with `fields`."""
    entry = STRINGS.get(key)
    if not entry:
        return key
    text = entry.get(_current) or entry.get(DEFAULT, key)
    if fields:
        try:
            return text.format(**fields)
        except (KeyError, IndexError, ValueError):
            return text
    return text

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
        (f"{cities:,}", t("stat.destinations")),
        (f"{countries:,}", t("stat.countries")),
        (f"{regions}", t("stat.regions")),
        ("9", t("stat.dimensions")),
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


def _cluster_tastes(profile: dict[str, float], mean: dict[str, float]) -> list[str]:
    """The taste keys a cluster over-indexes on versus the catalogue.

    Keys, not labels. The bundle is cached and outlives a language change, so
    naming has to happen at display time or a cluster fitted in English would
    keep its English name after the user switches to Arabic.
    """
    lifts = sorted(
        ((k, profile[k] - mean.get(k, 3.0)) for k in TASTE_KEYS),
        key=lambda kv: -kv[1],
    )
    return [k for k, lift in lifts[:2] if lift > 0.35]


@dataclass
class ModelBundle:
    """Everything fitted against one catalogue."""
    columns: list[str]
    scaler: StandardScaler
    matrix: np.ndarray                     # scaled catalogue, rows aligned to df
    kmeans: KMeans | None
    labels: np.ndarray                     # cluster id per row
    cluster_tastes: dict[int, list[str]]
    norms: np.ndarray                      # row norms, precomputed for scoring

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_tastes)

    def transform(self, values: dict) -> np.ndarray:
        """Put one preference dict into the same space as the catalogue."""
        row = pd.DataFrame([{c: values.get(c, 0) for c in self.columns}])[self.columns]
        return self.scaler.transform(row.to_numpy(dtype=float))

    def cluster_of(self, index: int) -> int:
        return int(self.labels[index]) if index < len(self.labels) else -1

    def cluster_name(self, index: int) -> str:
        """The cluster's label, translated now rather than at fit time."""
        keys = self.cluster_tastes.get(self.cluster_of(index)) or []
        if not keys:
            return t("profile.allrounder")
        return t("profile.join").join(t(f"taste.{k}") for k in keys)


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
    tastes: dict[int, list[str]] = {0: []}

    if k > 1:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(matrix)

        # characterise each cluster from its members' real (unscaled) averages
        catalogue_mean = {c: float(df[c].mean()) for c in TASTE_KEYS if c in df.columns}
        tastes = {}
        for cid in range(k):
            members = df.loc[labels == cid]
            if members.empty:
                tastes[cid] = []
                continue
            profile = {c: float(members[c].mean())
                       for c in TASTE_KEYS if c in members.columns}
            tastes[cid] = _cluster_tastes(profile, catalogue_mean)

    log.info("models fitted: %d features, %d clusters", len(columns), max(1, k))

    return ModelBundle(
        columns=columns, scaler=scaler, matrix=matrix, kmeans=kmeans,
        labels=labels, cluster_tastes=tastes, norms=norms,
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
    for col, _ in REGIONS:
        if to_float(row.get(col, 0)) == 1:
            return t(f"region.{col}")
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


@safe("", "climate_summary")
def climate_summary(row) -> str:
    temp = to_float(row.get("temp_avg_yearly", 20), 20.0)
    value = f"{temp:.0f}"
    if temp >= 28:
        return t("clim.hot", t=value)
    if temp >= 22:
        return t("clim.warm", t=value)
    if temp >= 15:
        return t("clim.mild", t=value)
    if temp >= 8:
        return t("clim.cool", t=value)
    return t("clim.cold", t=value)


@safe("", "best_season")
def best_season(row) -> str:
    """Infer a sensible window from latitude and average temperature."""
    lat = to_float(row.get("latitude", 0), 0.0)
    temp = to_float(row.get("temp_avg_yearly", 20), 20.0)
    north = lat >= 0
    if abs(lat) < 15:                       # tropics: the dry months matter, not heat
        return t("season.tropics")
    if temp >= 26:                          # hot climates are best off-peak
        return t("season.hot_north") if north else t("season.hot_south")
    if temp <= 8:                           # cold climates: peak summer or snow
        return t("season.cold")
    return t("season.mild_north") if north else t("season.mild_south")


@safe([], "trip_style")
def trip_style(row) -> list[str]:
    tastes = tastes_of(row)
    ranked = sorted(tastes.items(), key=lambda kv: -kv[1])
    return ([t(f"taste.{k}") for k, v in ranked[:3] if v >= 3.5]
            or [t("taste.balanced")])


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


@safe("", "travel_tip")
def travel_tip(row) -> str:
    """A short, situation-specific pointer built from the row's own numbers."""
    tastes = tastes_of(row)
    temp = to_float(row.get("temp_avg_yearly", 20), 20.0)
    tier = int(to_float(row.get("budget_level_encoded", 2), 2.0))
    code = clean(row.get("iata"))

    if temp >= 28:
        return t("tip.hot")
    if temp <= 8:
        return t("tip.cold")
    if tastes.get("seclusion", 3) >= 4:
        return t("tip.quiet")
    if tastes.get("nightlife", 3) >= 4 and tier <= 2:
        return t("tip.night")
    if tastes.get("culture", 3) >= 4:
        return t("tip.culture")
    if code:
        return t("tip.flight", code=code)
    return t("tip.default")


# --------------------------------------------------------------------------- #
# insights across the whole result set
# --------------------------------------------------------------------------- #

def travel_insights(results: pd.DataFrame, prefs: dict, answers: dict,
             profiles: list[str] | None = None) -> list[tuple[str, str]]:
    """Plain-language readings of the result set, as (heading, body) pairs.

    Every phrase comes from the string catalogue, so the readings follow the
    selected language rather than being assembled from English fragments.
    """
    out: list[tuple[str, str]] = []
    if results.empty:
        return out

    joiner, sep = t("insight.and"), t("list.sep")

    # what the traveller asked for
    wanted = [t(f"taste.{k}") for k in TASTE_KEYS if to_float(prefs.get(k, 3), 3.0) >= 4]
    tier = t(f"budget.{to_int(answers.get('budget', 2), 2, 1, 3)}")
    if wanted:
        what = wanted[0] if len(wanted) == 1 else sep.join(wanted[:-1]) + joiner + wanted[-1]
        out.append((t("insight.profile.h"),
                    t("insight.profile.some", what=what, tier=tier)))
    else:
        out.append((t("insight.profile.h"), t("insight.profile.none", tier=tier)))

    # how close the climate lands
    temps = results["temp_avg_yearly"].astype(float)
    target = to_float(answers.get("temp", 22), 22.0)
    out.append((
        t("insight.climate.h"),
        t("insight.climate.b",
          close=int((temps - target).abs().le(4).sum()), n=len(results),
          target=f"{target:.0f}", lo=f"{temps.min():.0f}", hi=f"{temps.max():.0f}"),
    ))

    # what it costs
    costs = [daily_cost(r) for _, r in results.iterrows()]
    nights = to_int(answers.get("nights", 7), 7, 1, 60)
    out.append((
        t("insight.cost.h"),
        t("insight.cost.b", lo=min(costs), hi=max(costs), nights=nights,
          tlo=f"{min(costs) * nights:,}", thi=f"{max(costs) * nights:,}"),
    ))

    # whether they can be combined
    regions = [r for r in (region_of(row) for _, row in results.iterrows()) if r]
    if regions:
        top = max(set(regions), key=regions.count)
        share = regions.count(top)
        out.append((
            t("insight.cluster.h"),
            t("insight.cluster.one", share=share, n=len(results), top=top)
            if share > 1 else
            t("insight.cluster.many", k=len(set(regions))),
        ))

    # how varied the shortlist is
    if profiles and len(profiles) > 1:
        listed = sep.join(profiles[:-1]) + joiner + profiles[-1]
        out.append((t("insight.profiles.h"),
                    t("insight.profiles.b", k=len(profiles), listed=listed)))
    elif profiles:
        out.append((t("insight.single.h"), t("insight.single.b", p=profiles[0])))

    # gaps worth knowing about
    with_air = sum(1 for _, r in results.iterrows()
                   if clean(r.get("name")) or clean(r.get("iata")))
    if with_air < len(results):
        out.append((t("insight.air.h"),
                    t("insight.air.b", k=len(results) - with_air)))
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
# STYLING — Visual styling — the original light blue TripWise identity.
# ==========================================================================

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Outfit:wght@400;500;600;700;800&"
    "family=Cairo:wght@400;500;600;700;800&"
    "family=Inter:wght@400;500;600&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)

THEME_BASE = """
<style>
@import url('__FONTS__');

/* ======================================================================
   TOKENS
   ==================================================================== */
:root{
  --text:#0F172A;
  --text-2:#3F5069;
  --text-3:#64748B;

  --cyan:#0EA5E9;
  --sky:#0EA5E9;
  --blue:#2563EB;
  --deep:#1E3A8A;

  --line:rgba(30,58,138,.13);
  --line-2:rgba(30,58,138,.22);

  --glow-soft:0 0 22px rgba(14,165,233,.2);
  --sh-card:0 1px 2px rgba(9,16,32,.04), 0 10px 30px rgba(37,99,235,.1);
  --sh-lift:0 26px 60px rgba(37,99,235,.2);

  --r-xl:28px; --r-lg:22px; --r-md:14px; --r-sm:12px;
  /* one spacing scale, so gaps and padding never drift apart */
  --gap:18px; --pad:1.5rem; --section:3.2rem;
  --ease:cubic-bezier(.22,.75,.28,1);
}

/* ======================================================================
   PAGE
   ==================================================================== */
.stApp{
  background:
    radial-gradient(900px 520px at 6% -8%, rgba(255,255,255,.55), transparent 62%),
    radial-gradient(760px 480px at 96% 0%, rgba(255,255,255,.4), transparent 58%),
    linear-gradient(180deg,#DCEFFC 0%,#C9E6FA 55%,#BFE0F8 100%);
  background-attachment:fixed;
}
#MainMenu, footer{ visibility:hidden; }
.block-container{ max-width:1180px; padding-top:2rem; padding-bottom:3rem; }

html, body, .stApp, [class*="css"]{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  color:var(--text); -webkit-font-smoothing:antialiased;
}
::selection{ background:rgba(14,165,233,.22); }
:focus-visible{ outline:2px solid var(--blue); outline-offset:3px; border-radius:8px; }

/* Streamlit writes body text in its own colour; pull it onto the dark ground */
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{
  color:var(--text-2);
}
/* Captions default to a pale grey that is hard to read on the cabin blue.
   These carry chart explanations and data notices, so they are darkened. */
.stCaption, .stCaption p, [data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] div,
small, .stMarkdown small{
  color:#3A4A61 !important; font-weight:500 !important;
}
.js-plotly-plot .plotly text, .js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text{ fill:#3A4A61 !important; }
.stMarkdown strong, [data-testid="stMarkdownContainer"] strong{ color:var(--text); }
h1, h2, h3, h4, h5, h6{ color:var(--text) !important; }

/* ======================================================================
   TYPE
   ==================================================================== */
.tw-display{
  font-family:'Outfit',sans-serif; font-weight:800;
  letter-spacing:-.04em; line-height:1.02; color:var(--text);
  font-size:clamp(2.6rem,6.2vw,4.8rem); margin:1.2rem 0 1.3rem;
}
.tw-grad{
  background:linear-gradient(96deg,var(--sky) 0%,var(--blue) 60%,var(--deep) 100%);
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; color:var(--sky);
  filter:drop-shadow(0 0 24px rgba(14,165,233,.34));
}
.tw-eyebrow{
  font-family:'JetBrains Mono',ui-monospace,monospace; font-size:.66rem;
  letter-spacing:.26em; text-transform:uppercase; color:var(--blue);
  display:inline-flex; align-items:center; gap:.6rem;
}
.tw-eyebrow::before{
  content:""; width:26px; height:1px;
  background:linear-gradient(90deg,var(--sky),transparent);
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
/* ---- header band: brand above the nav strip, read as one unit ---------- */
.tw-header{
  display:flex; align-items:baseline; justify-content:space-between;
  flex-wrap:wrap; gap:.6rem; padding:.2rem 0 .9rem;
}
.tw-header__tag{
  font-family:'JetBrains Mono',ui-monospace,monospace; font-size:.66rem;
  letter-spacing:.2em; text-transform:uppercase; color:var(--text-3);
}
.tw-brand{ display:flex; align-items:center; gap:.65rem; }
.tw-brand svg{ width:30px; height:30px; }
.tw-brand b{
  font-family:'Outfit',sans-serif; font-weight:800; font-size:1.24rem;
  letter-spacing:-.025em; color:var(--text);
}

.tw-badge{
  display:inline-block; padding:.42rem 1rem; border-radius:99px;
  background:rgba(14,165,233,.12); border:1px solid rgba(14,165,233,.22);
  color:var(--blue); font-weight:600; font-size:.82rem; letter-spacing:.01em;
}
.tw-cta{
  display:inline-block; margin-top:2rem; padding:.85rem 1.6rem; border-radius:99px;
  background:linear-gradient(96deg,var(--sky),var(--blue)); color:#fff;
  font-weight:600; font-size:.96rem;
  box-shadow:0 12px 30px rgba(37,99,235,.32), 0 0 20px rgba(14,165,233,.2);
}
.tw-cta b{ font-weight:700; }

.tw-hero{ position:relative; padding:1.6rem 0 .6rem; }
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
/* Equal cards on a grid: auto-fit gives every card the same width and the
   same gap at any breakpoint, and a fixed min-height keeps their heights
   equal regardless of how long a label wraps. */
.tw-stats{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:var(--gap); margin:2.4rem 0 .5rem; padding:0;
}
.tw-stat{
  position:relative; overflow:hidden; min-height:172px;
  display:flex; flex-direction:column; align-items:flex-start; gap:.35rem;
  padding:1.4rem 1.3rem 1.5rem;
  background:linear-gradient(160deg, rgba(255,255,255,.85), rgba(219,234,254,.6));
  border:1px solid rgba(255,255,255,.75);
  border-radius:var(--r-lg); box-shadow:var(--sh-card);
  transition:transform .34s var(--ease), box-shadow .34s var(--ease),
             border-color .34s var(--ease);
}
.tw-stat::before{
  content:""; position:absolute; inset:0 0 auto; height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.98),transparent);
}
.tw-stat:hover{
  transform:translateY(-6px);
  box-shadow:var(--sh-lift), 0 0 30px rgba(14,165,233,.22);
  border-color:rgba(14,165,233,.4);
}
.tw-stat__icon{
  width:44px; height:44px; border-radius:14px; display:grid; place-items:center;
  background:linear-gradient(140deg,var(--sky),var(--blue));
  box-shadow:0 10px 22px rgba(14,165,233,.3); margin-bottom:.5rem;
}
.tw-stat__icon svg{ width:22px; height:22px; fill:none; stroke:#fff;
  stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.tw-stat b{
  display:block; font-family:'Outfit',sans-serif; font-weight:800;
  font-size:clamp(2rem,3.6vw,2.7rem); line-height:1; letter-spacing:-.04em;
  background:linear-gradient(96deg,var(--sky),var(--blue));
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent;
}
.tw-stat span{
  font-size:.8rem; font-weight:500; color:var(--text-2); letter-spacing:.01em;
  line-height:1.4;
}
.tw-stat__rule{
  position:absolute; left:1.3rem; bottom:1rem; height:2px; width:0;
  border-radius:2px; background:linear-gradient(90deg,var(--sky),var(--blue));
  transition:width .4s var(--ease);
}
.tw-stat:hover .tw-stat__rule{ width:44px; }

.tw-metrics{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:var(--gap); margin:1.6rem 0 .4rem;
}
.tw-metric{
  background:linear-gradient(160deg, rgba(255,255,255,.88), rgba(219,234,254,.62));
  border:1px solid rgba(255,255,255,.8); border-radius:var(--r-lg);
  padding:1.1rem 1.2rem; box-shadow:var(--sh-card); min-height:104px;
  transition:transform .3s var(--ease), box-shadow .3s var(--ease);
}
.tw-metric:hover{ transform:translateY(-4px);
  box-shadow:var(--sh-lift), 0 0 24px rgba(14,165,233,.18); }
.tw-metric b{
  display:block; font-family:'JetBrains Mono',ui-monospace,monospace;
  font-weight:500; font-size:1.55rem; letter-spacing:-.03em; line-height:1.1;
  background:linear-gradient(96deg,var(--sky),var(--blue));
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
  background:linear-gradient(160deg, rgba(255,255,255,.85), rgba(219,234,254,.6));
  backdrop-filter:blur(16px) saturate(1.35);
  -webkit-backdrop-filter:blur(16px) saturate(1.35);
  border:1px solid rgba(255,255,255,.75);
  border-radius:var(--r-lg); box-shadow:var(--sh-card);
  transition:transform .34s var(--ease), box-shadow .34s var(--ease),
             border-color .34s var(--ease);
  height:100%; overflow:hidden;
}
.tw-card::before{
  content:""; position:absolute; inset:0 0 auto; height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.98),transparent);
}
.tw-card:hover{
  transform:translateY(-6px); box-shadow:var(--sh-lift);
  border-color:rgba(14,165,233,.4);
}

.tw-feat{ padding:var(--pad); min-height:210px; display:flex;
  flex-direction:column; }
.tw-feat__icon{
  width:64px; height:64px; border-radius:20px; display:grid; place-items:center;
  background:linear-gradient(140deg, var(--sky), var(--blue));
  box-shadow:0 14px 30px rgba(14,165,233,.32);
  margin-bottom:1.1rem;
}
.tw-feat__icon svg{ width:30px; height:30px; fill:none; stroke:#fff;
  stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
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
  background:linear-gradient(180deg,transparent,rgba(6,12,24,.7));
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
  background:rgba(255,255,255,.93); border:1px solid rgba(255,255,255,.9);
  backdrop-filter:blur(8px);
  font-family:'JetBrains Mono',ui-monospace,monospace; font-size:.72rem;
  color:var(--blue); box-shadow:0 4px 14px rgba(0,0,0,.16);
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
  border-left:2px solid var(--sky); padding:.1rem 0 .1rem .8rem;
  color:var(--text-2); font-size:.86rem; line-height:1.58;
}
.tw-chips{ display:flex; flex-wrap:wrap; gap:.34rem; }
.tw-chip{
  padding:.24rem .6rem; border-radius:99px; font-size:.7rem; font-weight:500;
  background:rgba(14,165,233,.11); color:#0369A1;
  border:1px solid rgba(14,165,233,.2);
}
.tw-chip--muted{
  background:rgba(9,16,32,.05); color:var(--text-3);
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
.tw-insight{ padding:var(--pad); min-height:132px; display:flex; gap:.95rem; align-items:flex-start; }
.tw-insight__dot{
  width:38px; height:38px; border-radius:12px; flex:0 0 auto; display:grid;
  place-items:center;
  background:linear-gradient(140deg, rgba(14,165,233,.16), rgba(37,99,235,.16));
  border:1px solid rgba(37,99,235,.16);
}
.tw-insight__dot svg{ width:18px; height:18px; stroke:var(--blue); fill:none;
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
  background:linear-gradient(96deg,var(--sky),var(--blue));
  color:#fff !important; border:none;
  box-shadow:0 10px 26px rgba(37,99,235,.32), 0 0 18px rgba(14,165,233,.2);
}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover{
  transform:translateY(-2px); filter:brightness(1.04);
  box-shadow:0 16px 36px rgba(37,99,235,.42), 0 0 28px rgba(14,165,233,.32);
}
.stButton>button[kind="primary"]:active, .stFormSubmitButton>button:active{
  transform:translateY(0); filter:brightness(.97);
}
.stButton>button[kind="secondary"]{
  background:rgba(255,255,255,.86); color:var(--text) !important;
  border:1px solid var(--line-2); box-shadow:var(--sh-card);
}
.stButton>button[kind="secondary"]:hover{
  border-color:var(--sky); color:var(--blue) !important;
  transform:translateY(-2px); box-shadow:var(--sh-lift), var(--glow-soft);
}

/* ======================================================================
   TABS — the navigation, dressed as a glowing pill bar
   ==================================================================== */
.stTabs [data-baseweb="tab-list"]{
  gap:.2rem; padding:0 0 0; margin:0 0 2.4rem;
  background:transparent; width:100%;
  border-bottom:1px solid rgba(30,58,138,.14);
}
.stTabs [data-baseweb="tab"]{
  position:relative; font-family:'Inter',sans-serif; font-weight:600;
  font-size:.95rem; color:var(--text-3); padding:.75rem 1.15rem;
  border-radius:10px 10px 0 0; background:transparent;
  transition:color .24s var(--ease);
}
.stTabs [data-baseweb="tab"]:hover{ color:var(--blue); background:transparent; }
.stTabs [data-baseweb="tab"]::after{
  content:""; position:absolute; left:1.15rem; right:1.15rem; bottom:-1px;
  height:2px; border-radius:2px; background:linear-gradient(90deg,var(--sky),var(--blue));
  transform:scaleX(0); transform-origin:center; transition:transform .28s var(--ease);
}
.stTabs [aria-selected="true"]{ color:var(--blue) !important; background:transparent; }
.stTabs [aria-selected="true"]::after{ transform:scaleX(1); }
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
  font-size:.7rem !important; color:#fff !important;
  background:var(--blue) !important; padding:.08rem .44rem !important;
  border-radius:7px !important;
}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"]{
  font-family:'JetBrains Mono',ui-monospace,monospace !important;
  font-size:.62rem !important; color:var(--text-3) !important;
}
[data-testid="stSlider"] [role="slider"]{
  box-shadow:0 2px 10px rgba(37,99,235,.34) !important;
}

/* Every select, multiselect and input shares one definition, so they cannot
   drift apart as pages are added. */
div[data-baseweb="select"] > div,
.stNumberInput input, .stTextInput input, .stNumberInput [data-baseweb="input"]{
  background:rgba(255,255,255,.92) !important;
  border:1px solid var(--line-2) !important;
  border-radius:var(--r-sm) !important;
  color:var(--text) !important;
  min-height:2.9rem !important;
  font-size:.92rem !important; font-weight:500 !important;
  padding:0 .25rem !important;
  box-shadow:0 2px 8px rgba(30,58,138,.06) !important;
  transition:border-color .2s var(--ease), box-shadow .2s var(--ease) !important;
}
div[data-baseweb="select"] > div:focus-within{
  border-color:var(--blue) !important;
  box-shadow:0 0 0 3px rgba(37,99,235,.14) !important;
}
div[data-baseweb="select"] > div:hover,
.stNumberInput input:hover, .stTextInput input:hover{
  border-color:var(--sky) !important;
}
div[data-baseweb="select"] svg{ fill:var(--text-2) !important; }
div[data-baseweb="select"] [data-baseweb="tag"]{
  background:rgba(14,165,233,.14) !important;
  border:1px solid rgba(14,165,233,.26) !important;
  color:#0369A1 !important;
}

/* Dropdown menus mount in a portal at the document root, outside .stApp, so
   they keep a white background unless targeted directly. */
/* Dropdown surfaces.
   The menu mounts in a portal at the document root, outside .stApp, so it does
   not inherit anything above and has to be styled from scratch — including the
   option text, which otherwise inherits a colour that can land dark-on-dark. */
[data-baseweb="popover"]{ animation:twMenuIn .18s var(--ease) both; }
@keyframes twMenuIn{
  from{ opacity:0; transform:translateY(-6px); }
  to{ opacity:1; transform:none; }
}
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul,
[data-baseweb="menu"], [data-baseweb="menu"] ul{
  background:linear-gradient(160deg,#123B6B,#0E2C52) !important;
  border:1px solid rgba(255,255,255,.16) !important;
  border-radius:var(--r-sm) !important;
  box-shadow:0 20px 50px rgba(15,40,80,.4) !important;
}
[data-baseweb="menu"] li, [role="option"],
[data-baseweb="menu"] li div, [role="option"] div, [role="option"] span{
  color:#FFFFFF !important; background:transparent !important;
  font-weight:500 !important; font-size:.92rem !important;
  padding-top:.42rem !important; padding-bottom:.42rem !important;
}
[data-baseweb="menu"] li:hover, [role="option"]:hover,
[role="option"][aria-selected="true"]{
  background:linear-gradient(96deg,var(--sky),var(--blue)) !important;
}
[data-baseweb="menu"] li:hover *, [role="option"]:hover *,
[role="option"][aria-selected="true"] *{ color:#FFFFFF !important; }

/* the closed control shows the current value — it must read clearly too */
div[data-baseweb="select"] > div, div[data-baseweb="select"] > div *{
  color:var(--text) !important;
}
div[data-baseweb="select"] input{ color:var(--text) !important; }

div[role="radiogroup"] label{
  background:rgba(255,255,255,.88); border:1px solid var(--line-2);
  border-radius:99px; padding:.4rem .95rem !important; margin:0 .4rem 0 0 !important;
  transition:all .22s var(--ease);
}
div[role="radiogroup"] label:hover{ border-color:var(--sky); }

/* ======================================================================
   DATA SURFACES
   ==================================================================== */
[data-testid="stMetricValue"]{
  font-family:'JetBrains Mono',ui-monospace,monospace; font-weight:500;
  color:var(--blue);
}
[data-testid="stMetricLabel"] p{ color:var(--text-3) !important; }

[data-testid="stExpander"] details{
  border:1px solid rgba(255,255,255,.7); border-radius:var(--r-lg);
  background:linear-gradient(160deg, rgba(255,255,255,.78), rgba(219,234,254,.55));
  backdrop-filter:blur(14px) saturate(1.3);
  -webkit-backdrop-filter:blur(14px) saturate(1.3);
  box-shadow:var(--sh-card);
}
[data-testid="stExpander"] summary{ color:var(--text-2) !important; }

[data-testid="stDataFrame"]{
  border:1px solid rgba(255,255,255,.7); border-radius:var(--r-lg);
  overflow:hidden; box-shadow:var(--sh-card);
}
[data-testid="stDataFrame"] [data-testid="stTable"],
[data-testid="stDataFrame"] div[role="grid"]{ background:transparent !important; }
[data-testid="stAlert"]{
  background:linear-gradient(160deg, rgba(255,255,255,.82), rgba(219,234,254,.6)) !important;
  border:1px solid rgba(255,255,255,.72) !important;
  border-radius:var(--r-lg) !important; color:var(--text) !important;
  box-shadow:var(--sh-card) !important;
}
[data-testid="stAlert"] p{ color:var(--text-2) !important; }

hr, [data-testid="stDivider"] hr{ border-color:var(--line) !important; }

/* ======================================================================
   MAP
   ==================================================================== */
.tw-map{
  position:relative; height:var(--map-h,420px); overflow:hidden;
  border-radius:var(--r-lg); border:1px solid rgba(255,255,255,.75);
  box-shadow:var(--sh-card); margin:.4rem 0 .6rem;
}
.tw-map svg{ width:100%; height:100%; display:block; }
.tw-map__label{
  font-family:'Inter',sans-serif; font-size:15px; font-weight:600;
  fill:#0F172A; paint-order:stroke; stroke:#FFFFFF; stroke-width:4px;
  stroke-linejoin:round;
}
.tw-map__halo{ animation:twPulse 3s ease-out infinite; transform-origin:center; }
@keyframes twPulse{
  0%{ opacity:.55; transform:scale(.55); }
  70%{ opacity:0; transform:scale(1.35); }
  100%{ opacity:0; transform:scale(1.35); }
}
.tw-map__pin{
  animation:twDrop .7s var(--ease) both; transform-origin:center;
  cursor:pointer; transition:transform .2s var(--ease);
}
.tw-map__pin:hover{ transform:scale(1.22); }

/* Labels are hidden until hover. Pan and zoom would need JavaScript, which
   st.markdown does not run, so reading a marker happens on hover instead and
   the zoom level is chosen with a Streamlit control. */
.tw-map__label--hover{ opacity:0; transition:opacity .2s var(--ease); }
.tw-map__pin:hover .tw-map__label--hover{ opacity:1; }
@keyframes twDrop{ from{ opacity:0; transform:translateY(-10px); } to{ opacity:1; transform:none; } }
.tw-map__route{ animation:twFlow 22s linear infinite; }
@keyframes twFlow{ to{ stroke-dashoffset:-300; } }

/* ======================================================================
   FOOTER
   ==================================================================== */
.tw-footer{
  margin-top:5rem; padding:2.4rem 0 .6rem;
  border-top:1px solid rgba(30,58,138,.14);
  display:grid; gap:.7rem;
}
.tw-footer__brand{
  font-family:'Outfit',sans-serif; font-weight:800; font-size:1.05rem;
  color:var(--text); letter-spacing:-.02em;
}
.tw-footer__note{
  color:var(--text-3); font-size:.84rem; line-height:1.7; max-width:560px;
}
.tw-footer__meta{
  font-family:'JetBrains Mono',ui-monospace,monospace; font-size:.7rem;
  color:var(--text-3); letter-spacing:.06em;
}

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

# Applied on top of BASE when Arabic is active. Direction is set on the app
# container rather than <html>, which Streamlit owns, and the few components
# that must stay left-to-right — numbers, charts, the map — are exempted.
RTL = """
<style>
.stApp, .stApp *{ font-family:'Cairo','Inter',sans-serif !important; }
.tw-mono, .tw-metric b, .tw-stat b, .tw-facts span, .tw-eyebrow,
.tw-footer__meta, [data-testid="stThumbValue"],
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"]{
  font-family:'JetBrains Mono',ui-monospace,monospace !important;
}
.stApp{ direction:rtl; }
.block-container{ direction:rtl; text-align:right; }
.tw-lede, .tw-sub, .tw-feat p, .tw-insight p, .tw-why, .tw-meta__row,
.tw-footer__note, .stMarkdown p, .stMarkdown li{ text-align:right; }
.tw-why{ border-left:none; border-right:2px solid var(--sky);
  padding:.1rem .8rem .1rem 0; }
.tw-eyebrow{ flex-direction:row-reverse; }
.tw-eyebrow::before{ background:linear-gradient(270deg,var(--sky),transparent); }
.tw-meta__row{ flex-direction:row-reverse; }
.tw-insight{ flex-direction:row-reverse; }
.tw-dest__place{ left:92px; right:16px; text-align:right; }
.tw-dest__match{ left:12px; right:auto; }
.tw-header{ flex-direction:row-reverse; }
.tw-brand{ flex-direction:row-reverse; }
.tw-facts{ direction:rtl; }

/* these read wrong mirrored, so they keep their own direction */
.tw-map, .js-plotly-plot, [data-testid="stDataFrame"],
[data-testid="stSlider"], [data-testid="stSlider"] *{ direction:ltr; }
[data-testid="stSlider"] label, .stSelectbox label, .stRadio label,
.stCheckbox label, .stNumberInput label, .stMultiSelect label{
  direction:rtl; text-align:right; display:block;
}
div[role="radiogroup"]{ flex-direction:row-reverse; }
</style>
"""


def stylesheet(rtl: bool = False) -> str:
    """The stylesheet for the active direction."""
    return THEME_BASE + (RTL if rtl else "")


# kept so existing imports of theme.CSS continue to resolve
THEME_CSS = THEME_BASE

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


def header() -> str:
    """The site header: mark, wordmark and a one-line positioning statement.

    The navigation itself is the tab strip Streamlit renders directly beneath
    this, styled to read as one continuous header band.
    """
    return f"""
<div class="tw-header">
<div class="tw-brand">
<svg viewBox="0 0 24 24" fill="none" stroke="url(#bg1)" stroke-width="1.8"
stroke-linecap="round" stroke-linejoin="round">
<defs><linearGradient id="bg1" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/>
</linearGradient></defs>
<path d="M2 13.4 21 4l-4.6 16.2-4.1-6.1z"/><path d="m12.3 14.1-3.4 6.9-1-5"/>
</svg>
<b>TripWise <span class="tw-grad">AI</span></b>
</div>
<div class="tw-header__tag">{t("hero.tagline")}</div>
</div>"""


def footer() -> str:
    """Closes the page the way a product site would."""
    return f"""
<div class="tw-footer">
<div class="tw-footer__brand">TripWise <span class="tw-grad">AI</span></div>
<div class="tw-footer__note">{_e(t("foot.note"))}</div>
<div class="tw-footer__meta">&copy; 2026 TripWise AI &nbsp;&middot;&nbsp;
hello@tripwise.ai</div>
</div>"""


STAT_ICONS = ["pin", "globe", "layers", "compass"]


def stat_grid(stats: list[tuple[str, str]]) -> str:
    """The headline figures.

    Each number counts up from zero using an animated custom property, which
    keeps the effect in CSS — no script, so it works inside st.markdown. The
    digits are emitted as a counter, and the raw value stays in the markup for
    assistive technology and for any engine that cannot animate the property.
    """
    rules, cells = [], []
    for i, (value, label) in enumerate(stats):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        target = int(digits) if digits else 0
        icon_name = STAT_ICONS[i % len(STAT_ICONS)]
        delay = 0.1 + i * 0.12

        rules.append(
            f"@property --v{i}{{syntax:'<integer>';initial-value:0;inherits:false}}"
            f"@keyframes cnt{i}{{to{{--v{i}:{target}}}}}"
            f".sv{i}{{counter-reset:v{i} var(--v{i});"
            f"animation:cnt{i} 1.9s cubic-bezier(.16,.8,.3,1) {delay:.2f}s forwards}}"
            f".sv{i}::after{{content:counter(v{i})}}"
        )
        cells.append(
            f'<div class="tw-stat tw-rise" style="animation-delay:{delay:.2f}s">'
            f'<div class="tw-stat__icon">{icon(icon_name)}</div>'
            f'<b class="sv{i}" aria-label="{_e(value)}"></b>'
            f'<span>{_e(label)}</span>'
            f'<i class="tw-stat__rule"></i>'
            f"</div>"
        )
    return (f"<style>{''.join(rules)}</style>"
            f'<div class="tw-stats">{"".join(cells)}</div>')


def hero() -> str:
    """The landing headline, with the aircraft drifting alongside it."""
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
<span class="tw-badge tw-rise d1">{_e(t("hero.badge"))}</span>
<h1 class="tw-display tw-rise d2">{_e(t("hero.title_1"))}<br/><span class="tw-grad">TripWise AI</span></h1>
<p class="tw-lede tw-rise d3">{_e(t("hero.lede"))}</p>
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

    rows = [("plane", t("card.airport"), d["airport"] or t("card.noairport")),
            ("calendar", t("card.season"), d["season"]),
            ("sun", t("card.weather"), d["climate"])]
    if d.get("hotel"):
        rows.append(("tag", t("card.stay"), d["hotel"]))
    if d.get("attractions"):
        rows.append(("pin", t("card.nearby"), ", ".join(d["attractions"])))
    rows.append(("bulb", t("card.tip"), d["tip"]))

    meta = "".join(
        f'<div class="tw-meta__row">{icon(ic)}<div><b>{_e(label)}</b> &mdash; {_e(val)}</div></div>'
        for ic, label, val in rows
    )

    return f"""
<div class="tw-card tw-rise" id="dest-{_e(d.get('slug', ''))}">
<div class="tw-dest__art">
{d['art']}
<div class="tw-dest__scrim"></div>
<div class="tw-dest__match">{_e(d['match'])}% {_e(t("card.match"))}</div>
<div class="tw-dest__place"><h3>{_e(d['city'])}</h3><span>{_e(d['country'])}</span></div>
</div>
<div class="tw-dest__body">
<div class="tw-facts">
<div><span>{_e(t("card.budget"))}</span><b>{_e(d['tier'])}</b></div>
<div><span>{_e(t("card.perday"))}</span><b>${_e(d['daily'])}</b></div>
<div><span>{_e(t("card.climate"))}</span><b>{_e(d['temp'])}&deg;</b></div>
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
# WORLD OUTLINE — Land outline for the destination map.
# ==========================================================================

WORLD_PATH = (
    "M0.0,66.1 16.9,73.8 16.6,76.6 18.9,77.7 18.1,74.5 27.1,75.1 33.7,79.3 30.4,81.2 24.9,81.6 24"
    ".8,85.9 23.5,86.8 13.4,83.9 12.6,82.0 5.5,81.8 3.7,80.3 4.4,78.7 0.4,79.7 1.9,81.8 0.0,83.6 "
    "1200.0,83.7 1195.7,85.6 1191.4,85.3 1197.9,92.4 1197.4,95.4 1191.2,94.4 1178.9,98.3 1169.0,1"
    "04.1 1167.8,106.1 1163.0,103.0 1154.3,106.5 1152.8,104.8 1149.6,106.7 1145.1,106.1 1140.1,11"
    "3.3 1140.2,115.0 1144.0,116.0 1143.5,122.4 1140.4,122.6 1139.0,126.3 1140.4,128.2 1134.6,130"
    ".4 1133.4,135.4 1128.4,136.5 1127.4,141.0 1122.6,145.1 1118.1,125.8 1119.7,119.7 1122.5,117."
    "1 1122.7,115.1 1127.9,114.1 1139.6,104.0 1145.6,100.5 1148.2,94.3 1144.2,94.7 1142.2,98.3 11"
    "33.7,103.1 1131.0,97.7 1122.4,99.2 1114.1,106.6 1116.8,109.3 1104.2,110.9 1104.5,107.7 1099."
    "3,107.0 1095.2,109.2 1085.0,108.4 1074.0,109.8 1050.4,128.7 1055.7,129.3 1057.3,132.0 1060.5"
    ",133.0 1062.7,130.8 1066.3,131.1 1071.2,135.9 1071.3,139.7 1068.7,144.1 1066.9,156.3 1060.7,"
    "165.7 1049.6,178.5 1045.1,181.1 1043.0,181.2 1040.9,179.0 1033.2,184.9 1032.3,189.6 1025.1,1"
    "94.5 1024.6,196.9 1027.8,199.6 1031.5,207.6 1031.6,212.7 1030.3,215.1 1021.6,218.1 1021.9,21"
    "2.4 1020.4,207.9 1022.9,207.1 1020.6,203.4 1019.0,202.5 1017.6,203.7 1015.7,201.8 1017.4,199"
    ".3 1017.7,195.4 1014.2,193.8 1003.5,198.3 1005.3,196.3 1004.6,194.6 1007.2,191.6 1005.5,189."
    "3 996.7,196.8 993.5,197.0 991.8,199.0 993.5,202.0 996.3,202.7 996.4,204.7 999.0,206.0 1002.7"
    ",202.8 1007.9,204.7 1008.4,207.0 1003.7,208.2 997.2,215.9 1000.8,218.3 1006.4,230.0 1006.3,2"
    "33.3 1004.2,234.5 1007.0,238.2 1005.6,245.2 1003.8,245.6 995.5,261.4 986.3,269.2 982.5,269.7"
    " 980.5,271.6 979.4,270.2 977.5,272.4 969.3,275.3 968.1,279.9 966.3,280.2 966.2,275.3 961.7,2"
    "73.9 955.7,278.4 952.9,282.5 952.2,285.6 957.9,295.9 962.9,302.2 964.4,310.3 964.0,318.1 950"
    ".5,331.5 949.3,328.7 950.3,325.7 947.8,323.3 945.0,322.6 941.9,315.8 939.0,313.8 936.1,313.8"
    " 936.6,310.4 933.7,310.4 930.7,328.7 932.9,328.9 934.9,336.7 943.2,345.1 944.6,348.0 945.0,3"
    "57.1 947.4,363.7 945.1,364.0 938.0,357.2 934.0,346.0 933.6,340.9 928.3,332.5 927.8,335.1 929"
    ".2,319.1 923.9,294.9 917.9,300.3 914.0,298.8 915.1,293.4 914.4,289.3 911.8,284.2 912.2,282.6"
    " 910.3,282.1 907.9,278.5 904.7,269.3 901.7,269.1 900.9,273.3 896.8,272.4 896.3,274.0 889.9,2"
    "74.8 890.1,278.2 888.3,280.7 883.5,283.7 874.0,294.5 874.0,296.6 867.8,299.5 866.2,323.8 864"
    ".5,324.0 863.0,327.4 864.0,328.8 860.9,330.1 858.5,334.3 855.3,330.2 849.5,313.3 848.1,305.1"
    " 845.1,299.1 842.1,275.5 837.2,278.1 834.9,277.6 830.6,272.2 832.1,270.6 831.2,268.9 824.8,2"
    "64.1 821.2,257.6 805.0,259.1 791.3,256.2 789.9,250.8 788.3,250.0 782.4,252.9 771.7,246.8 767"
    ".0,236.8 765.3,237.5 763.1,236.0 761.9,237.8 759.9,237.6 762.7,247.6 767.2,252.0 767.0,255.3"
    " 769.4,260.5 770.0,255.0 772.0,255.9 771.3,261.1 772.7,263.7 780.0,263.3 787.9,253.3 788.0,2"
    "59.8 789.5,262.8 795.8,265.7 799.4,271.3 795.0,279.5 792.8,280.3 792.3,286.1 788.7,287.7 787"
    ".6,290.8 784.2,291.8 784.2,293.6 774.6,297.3 773.9,300.8 765.2,304.7 762.3,307.8 752.1,310.9"
    " 750.0,313.5 744.9,313.8 742.0,302.5 742.2,295.6 736.5,283.7 730.5,275.7 730.2,270.1 728.3,2"
    "65.2 725.0,262.6 717.1,246.0 715.4,246.0 716.4,239.6 713.1,247.8 708.1,238.1 713.7,254.4 719"
    ".0,264.2 718.4,267.8 722.9,272.6 724.9,287.5 728.0,290.2 730.9,299.3 744.4,314.9 744.3,316.7"
    " 742.4,317.8 747.1,323.4 748.7,323.4 770.4,316.5 770.2,322.6 764.8,339.4 759.1,350.8 755.2,3"
    "56.8 743.8,368.1 738.6,376.8 734.2,380.7 730.7,389.9 729.3,397.8 731.5,399.4 730.6,406.7 734"
    ".9,416.7 735.9,434.0 731.5,442.9 724.7,446.7 716.0,456.4 715.7,459.5 718.5,466.5 718.2,475.4"
    " 708.6,482.5 709.7,484.6 707.3,495.8 694.1,513.5 685.9,518.6 675.2,518.3 665.4,522.5 660.8,5"
    "18.3 659.8,512.8 660.8,512.0 660.7,508.6 650.7,488.5 647.5,466.6 639.3,448.8 639.3,438.8 641"
    ".7,428.9 645.5,422.3 645.6,416.5 642.9,409.7 644.1,407.0 639.7,391.5 629.3,374.2 632.6,355.8"
    " 631.4,352.9 629.8,352.2 628.3,348.4 619.7,350.6 614.4,341.8 606.2,342.4 593.5,348.6 584.5,3"
    "46.6 574.9,350.3 570.0,348.1 558.6,337.4 550.5,321.5 544.6,315.8 544.3,309.6 541.3,304.6 545"
    ".1,298.4 546.2,289.7 545.7,281.0 543.1,277.0 543.4,273.1 545.8,269.6 546.7,265.0 549.7,261.5"
    " 551.9,253.9 557.9,246.1 561.0,245.6 568.1,237.7 567.3,232.3 569.0,226.2 571.1,223.2 577.0,2"
    "19.4 580.2,212.1 582.7,212.1 584.7,214.0 592.8,214.7 604.9,208.4 617.7,207.9 620.9,206.2 628"
    ".1,206.9 631.7,205.1 634.0,205.7 633.9,207.9 636.8,206.3 637.0,207.1 635.3,209.3 636.5,212.4"
    " 636.0,216.2 633.8,218.4 634.5,220.8 636.2,220.9 638.3,223.7 650.8,227.5 652.4,231.4 663.6,2"
    "36.3 666.8,233.1 666.1,229.7 667.1,227.6 671.8,224.9 676.3,225.8 677.5,227.8 683.1,229.1 683"
    ".9,230.5 688.3,230.5 696.4,233.6 703.3,230.6 705.6,231.2 706.5,233.3 707.3,231.9 712.6,233.2"
    " 715.2,230.6 720.0,217.0 720.5,211.8 719.3,209.9 720.5,208.2 715.7,207.6 713.4,210.1 708.4,2"
    "10.6 705.7,208.2 702.1,208.1 701.3,209.9 699.0,210.4 695.8,208.1 692.1,208.2 687.7,201.4 689"
    ".3,197.9 687.2,195.8 690.9,191.6 696.1,191.5 697.5,188.1 703.8,188.7 711.7,184.6 717.2,184.5"
    " 727.8,189.3 734.6,189.0 738.5,186.7 739.0,184.8 738.2,181.8 722.3,170.4 724.7,169.7 727.4,1"
    "66.0 725.6,164.3 730.5,162.5 730.4,161.5 716.5,165.9 716.7,168.6 721.8,169.4 721.1,171.0 712"
    ".9,174.3 711.1,173.4 711.8,171.3 708.2,170.1 712.0,167.7 711.0,166.7 705.8,165.6 705.6,164.0"
    " 702.5,164.5 698.8,171.3 696.1,171.9 695.2,177.2 692.3,182.1 693.7,186.3 696.6,187.8 696.0,1"
    "88.8 692.1,189.1 687.9,192.8 686.9,189.9 683.1,189.3 679.1,190.5 681.4,192.9 679.7,193.6 677"
    ".8,193.6 676.1,191.4 675.4,192.3 677.8,197.0 676.6,198.0 680.1,201.3 680.1,203.8 677.0,202.6"
    " 678.0,204.9 675.9,205.3 677.2,209.2 675.0,209.3 672.2,207.3 670.4,200.9 664.7,192.4 665.1,1"
    "85.9 653.4,178.1 650.6,174.8 649.7,171.2 647.5,170.5 646.5,172.4 645.5,170.9 646.5,168.9 643"
    ".8,168.2 641.1,169.8 642.0,175.5 650.5,184.9 653.1,184.9 653.9,185.8 653.0,186.7 661.6,192.7"
    " 661.0,194.3 656.2,191.5 654.8,194.4 657.2,196.0 656.8,198.3 655.4,198.6 653.7,202.3 652.3,2"
    "02.7 653.7,198.0 651.4,193.3 645.4,188.2 640.4,186.0 635.0,180.6 634.0,176.2 629.6,174.3 621"
    ".8,179.7 615.2,178.5 610.3,180.0 610.1,185.2 607.0,188.1 602.7,189.0 599.1,196.5 600.4,199.0"
    " 597.7,203.8 595.2,204.7 592.8,208.1 585.4,208.1 582.1,211.3 580.4,210.9 578.3,206.9 570.3,2"
    "07.2 570.5,201.1 568.2,199.0 570.8,190.1 570.1,182.1 568.7,180.2 573.4,177.0 593.7,178.4 595"
    ".4,175.8 596.0,167.0 590.1,160.2 585.0,158.5 584.7,155.3 589.0,154.3 594.6,155.5 593.5,150.5"
    " 596.7,152.4 604.5,148.9 605.5,145.3 612.8,142.4 615.7,135.9 623.7,133.3 627.1,134.0 629.3,1"
    "31.8 627.1,125.2 627.0,120.7 628.5,118.2 635.3,115.5 634.2,119.2 636.4,121.1 632.2,125.4 633"
    ".1,129.3 636.5,130.3 636.5,131.9 641.7,129.8 647.1,133.0 658.7,128.2 665.5,130.0 666.3,128.1"
    " 670.9,126.7 670.3,119.7 671.9,116.9 675.1,115.4 677.7,118.7 680.4,118.6 681.4,112.6 680.2,1"
    "13.2 678.1,111.6 677.8,109.1 686.2,107.2 693.3,107.8 697.1,105.4 693.6,103.3 676.2,106.2 671"
    ".1,102.4 671.8,98.0 670.2,94.1 671.8,91.5 684.7,83.1 684.3,81.2 679.7,79.1 673.9,80.4 670.7,"
    "83.4 671.2,86.1 659.5,93.4 657.1,99.6 662.6,105.2 659.6,110.1 656.1,111.2 652.9,122.7 648.9,"
    "122.2 647.0,125.7 643.1,125.9 636.8,110.6 634.5,107.9 627.9,113.0 623.5,114.0 618.9,111.7 61"
    "6.6,96.9 619.7,94.0 628.5,90.3 635.1,85.8 649.2,71.2 663.9,62.4 671.3,60.4 676.7,60.7 681.8,"
    "57.0 693.9,56.3 704.3,59.6 700.0,60.7 703.7,63.5 707.1,62.0 712.6,64.6 721.7,65.7 734.3,70.6"
    " 736.9,72.7 737.1,75.7 733.4,78.0 727.9,79.1 713.1,75.8 710.6,76.4 716.0,79.6 716.5,86.1 723"
    ".4,88.6 723.8,86.5 721.7,84.5 723.9,82.9 732.0,85.7 734.8,84.6 732.5,81.4 740.3,77.1 746.5,7"
    "8.8 748.4,75.8 745.7,73.2 747.3,70.6 744.8,67.8 754.2,69.3 756.1,71.7 751.9,72.3 751.9,74.7 "
    "754.5,76.2 759.6,75.3 760.5,72.5 779.1,66.6 781.6,66.8 778.3,69.5 782.4,69.9 784.8,68.4 791."
    "1,68.3 796.0,66.5 799.8,69.1 803.6,66.2 800.1,63.7 801.8,62.2 811.7,63.5 828.4,69.9 830.6,67"
    ".6 827.1,64.4 823.1,64.0 824.2,61.9 822.3,57.0 828.5,53.1 830.6,49.1 833.1,48.2 842.0,49.4 8"
    "42.6,51.8 839.5,55.4 841.6,56.8 842.6,59.8 841.9,65.9 845.6,68.6 837.6,77.7 841.4,78.4 846.4"
    ",75.7 850.2,71.4 848.2,68.9 849.8,66.0 846.1,65.6 845.3,63.2 848.0,58.8 843.7,55.2 849.6,52."
    "2 848.9,49.1 850.5,49.0 852.3,51.4 851.0,55.7 854.5,56.5 853.0,53.3 858.6,51.6 865.5,51.4 87"
    "1.7,53.9 868.7,50.2 868.4,45.5 889.4,44.3 886.7,42.0 890.6,39.1 909.7,36.2 910.8,35.0 919.5,"
    "34.6 922.3,35.6 929.7,33.2 935.9,33.3 940.0,29.5 947.8,27.7 953.6,29.1 949.0,30.2 956.6,30.9"
    " 957.5,33.1 970.3,32.1 980.4,35.8 979.6,38.1 964.7,43.2 973.7,44.9 976.7,44.1 978.4,46.9 985"
    ".2,45.1 995.9,45.8 996.7,47.8 1010.7,48.5 1010.9,45.1 1023.3,45.9 1028.6,48.2 1030.2,51.0 10"
    "28.2,52.9 1032.4,56.3 1037.6,58.1 1040.8,53.5 1046.2,55.5 1051.9,54.3 1058.3,55.6 1060.8,54."
    "4 1066.2,55.0 1063.8,50.9 1068.2,49.0 1098.3,51.9 1101.2,54.5 1109.9,57.9 1130.0,57.8 1132.8"
    ",59.6 1132.4,62.8 1136.5,64.0 1159.5,63.4 1165.3,67.3 1169.4,65.9 1166.7,63.1 1168.2,61.1 11"
    "85.7,62.1 1195.3,64.2 0.0,66.1ZM298.2,63.8 298.2,68.3 302.6,64.8 306.6,67.6 305.6,70.9 308.8"
    ",73.9 314.7,66.9 314.9,62.1 324.6,63.1 329.1,65.2 329.3,67.4 326.8,69.8 329.1,72.1 328.7,74."
    "3 322.2,77.3 317.5,78.0 314.1,76.7 308.9,84.5 305.1,87.5 300.3,87.8 297.7,89.7 297.4,92.5 29"
    "3.6,93.1 289.5,96.6 285.9,101.6 284.4,110.2 289.3,110.9 292.3,118.3 297.0,117.5 316.6,126.2 "
    "325.8,126.9 326.2,135.1 328.7,140.0 333.6,144.2 336.2,142.8 338.0,138.2 336.3,131.3 333.9,12"
    "9.0 339.2,126.9 344.9,120.8 342.3,114.1 338.3,110.8 342.2,106.2 339.6,95.3 342.0,94.3 351.1,"
    "95.9 353.9,94.8 362.1,100.5 368.0,100.9 369.0,110.1 372.1,110.8 374.5,113.4 379.3,110.9 384."
    "7,104.1 395.3,118.9 394.0,121.6 401.4,126.6 408.9,129.2 410.2,132.9 414.1,135.1 414.4,140.1 "
    "409.6,143.3 404.1,144.8 399.9,148.4 378.7,148.5 375.9,151.7 371.6,153.6 363.0,163.5 365.8,16"
    "2.8 371.2,157.0 378.2,153.3 383.2,152.9 386.1,155.0 382.9,158.0 385.1,166.0 389.4,168.2 394."
    "9,167.6 398.3,162.7 398.5,165.9 400.7,167.4 385.8,174.7 382.1,177.9 379.6,177.6 379.5,173.8 "
    "385.3,170.2 376.2,170.9 376.8,172.3 366.3,177.3 364.4,180.1 363.9,183.2 365.0,185.5 366.4,18"
    "5.6 366.1,184.0 367.0,185.0 366.8,186.3 354.3,189.4 360.2,189.4 353.5,190.2 352.7,194.8 350."
    "3,198.1 348.2,195.7 349.8,200.5 346.9,205.7 347.6,202.5 345.9,200.9 345.5,197.2 345.6,201.9 "
    "343.5,201.2 345.7,202.6 347.6,213.0 345.5,216.3 336.5,222.1 328.9,231.1 329.0,237.3 333.1,25"
    "1.2 332.1,258.5 329.4,258.5 327.6,255.6 323.8,246.7 323.6,241.4 319.7,237.0 316.3,239.0 312."
    "0,235.7 302.7,236.1 301.3,236.7 302.6,240.6 299.5,241.3 297.1,241.2 294.6,238.9 287.2,238.7 "
    "278.0,244.9 275.4,249.0 276.2,255.6 273.8,270.7 279.0,284.4 285.2,289.6 293.2,287.1 297.4,28"
    "4.6 299.1,277.0 304.9,274.8 309.8,274.6 310.5,277.7 307.9,283.0 307.2,289.1 305.7,288.0 305."
    "5,296.7 303.6,299.5 306.3,300.4 316.7,299.0 322.0,302.2 322.7,306.4 320.6,320.5 326.0,329.8 "
    "328.5,330.7 334.8,327.1 343.9,331.4 347.8,327.8 348.4,322.7 350.3,320.6 355.3,320.0 360.8,31"
    "4.7 362.9,316.1 362.1,318.6 360.2,319.1 361.2,323.4 359.8,326.0 361.0,329.5 362.5,329.2 363."
    "2,326.0 362.0,321.1 366.1,319.3 365.7,317.3 366.9,315.9 368.1,319.0 370.4,319.0 372.7,322.9 "
    "379.2,322.5 383.7,325.0 385.6,322.6 393.7,322.2 390.9,323.5 392.0,325.6 394.7,325.9 397.2,32"
    "8.1 397.8,331.6 403.0,334.2 405.1,339.3 409.5,343.1 413.5,344.0 416.6,342.9 420.1,344.0 423."
    "7,345.6 427.8,351.1 428.9,350.9 431.6,361.0 433.4,361.7 433.5,364.8 431.0,368.4 432.0,369.7 "
    "437.9,370.4 438.1,374.8 440.6,371.9 450.3,376.2 451.9,378.8 451.4,381.2 455.3,379.8 461.8,38"
    "2.2 466.7,382.0 475.9,390.6 481.3,392.0 484.2,401.7 482.9,408.9 471.1,426.8 469.1,447.9 463."
    "5,465.8 460.8,467.7 460.0,470.4 451.2,472.0 441.2,478.8 438.4,483.1 437.0,495.4 420.6,520.6 "
    "416.9,523.1 412.6,522.6 407.3,520.9 405.2,518.5 405.0,520.8 409.3,524.5 408.8,527.6 410.9,52"
    "9.5 410.7,531.6 407.5,537.3 402.6,539.6 392.2,540.1 392.8,548.2 390.8,549.8 387.4,550.4 384."
    "2,548.8 382.9,549.9 383.4,554.3 385.7,555.6 387.5,554.2 388.5,556.5 382.7,560.6 381.4,567.4 "
    "378.3,567.4 375.7,569.7 374.7,573.0 378.0,576.2 381.2,577.1 380.0,581.0 376.1,583.5 373.9,58"
    "8.6 369.5,592.4 370.6,597.0 372.8,599.6 368.5,599.3 363.9,602.0 363.3,606.1 358.1,604.7 350."
    "2,599.2 348.0,583.4 349.4,579.2 352.9,575.8 347.8,574.5 351.0,570.6 352.2,563.3 355.9,564.8 "
    "357.6,555.7 355.4,554.6 354.3,560.0 352.2,559.4 354.4,545.0 355.9,542.0 354.7,532.7 356.1,53"
    "2.6 361.9,511.9 361.7,496.3 363.7,490.9 366.1,456.2 365.4,450.0 362.1,447.5 361.8,445.7 346."
    "6,433.8 345.8,428.9 334.1,401.0 329.2,396.3 330.2,394.4 328.6,390.2 334.1,381.0 333.4,379.1 "
    "332.1,381.2 330.1,379.2 330.2,374.0 331.4,373.3 333.0,366.0 337.1,363.3 338.6,357.8 340.2,35"
    "7.5 342.9,352.4 341.7,351.4 341.7,339.9 339.3,336.3 339.4,332.8 336.3,329.8 334.8,330.1 331."
    "7,333.8 333.3,336.2 330.4,337.6 329.8,335.0 328.3,335.5 327.6,333.7 323.9,332.9 323.8,333.9 "
    "321.6,332.2 321.2,329.6 316.7,325.0 316.3,327.3 314.5,325.7 314.3,320.6 307.8,312.6 308.9,31"
    "2.3 308.4,310.9 305.1,311.5 295.9,308.1 288.8,300.7 284.4,298.1 278.1,300.5 263.9,293.9 260."
    "3,290.6 255.0,288.9 248.4,281.6 247.6,279.5 248.7,279.1 249.1,275.2 246.6,269.2 238.7,258.7 "
    "235.8,256.9 235.7,253.1 225.9,242.0 222.8,232.3 220.4,230.5 217.4,229.5 217.8,236.7 227.9,25"
    "2.1 231.1,262.5 232.8,262.7 235.3,266.6 233.2,269.0 232.3,266.3 226.1,260.6 225.7,255.0 216."
    "5,247.5 218.1,247.4 219.5,243.8 214.9,239.4 209.0,224.1 204.9,219.7 197.9,217.2 185.3,192.1 "
    "186.0,184.7 184.9,181.3 187.0,169.2 184.4,157.5 184.8,156.6 189.6,158.1 191.4,162.3 192.2,16"
    "1.1 190.5,153.9 181.2,147.7 175.2,145.8 173.4,142.0 173.8,139.3 169.6,137.4 169.0,133.8 165."
    "0,130.6 164.9,128.4 160.1,125.3 159.2,121.5 154.9,117.9 153.1,113.8 144.6,113.4 133.8,107.6 "
    "124.8,105.2 120.1,105.5 109.6,101.6 105.9,102.6 106.6,105.6 94.3,109.2 93.8,106.7 95.3,102.3"
    " 98.8,101.0 97.9,99.9 86.6,108.4 89.0,110.5 85.9,113.7 79.0,116.9 78.1,118.8 72.9,121.1 71.9"
    ",123.1 50.7,130.1 50.2,129.4 60.6,123.6 64.8,123.1 74.3,116.2 76.5,110.3 72.7,111.6 71.6,110"
    ".9 69.8,112.5 67.6,110.2 66.7,111.8 65.5,109.6 60.1,111.4 60.4,107.1 58.3,105.6 53.9,106.4 4"
    "8.8,103.3 48.8,100.8 46.3,98.9 51.5,91.7 56.4,92.1 59.1,90.0 64.1,89.0 63.5,87.0 61.6,86.2 6"
    "4.1,84.5 57.5,86.5 50.1,86.0 45.3,84.9 39.6,80.6 51.8,76.6 54.5,76.6 54.0,78.8 61.1,78.6 48."
    "7,70.2 44.1,68.8 46.0,66.5 51.9,66.3 56.1,64.3 56.9,62.2 60.3,60.1 72.9,57.9 78.1,55.6 83.1,"
    "56.5 85.5,58.5 92.6,57.9 92.4,58.9 97.5,59.7 116.9,61.6 121.4,60.9 145.0,66.4 151.9,63.2 156"
    ".9,63.7 167.4,60.7 169.6,62.5 172.1,61.5 172.9,59.4 180.8,63.8 185.2,60.9 185.7,64.2 191.1,6"
    "2.2 195.1,62.4 208.0,65.9 215.8,66.4 220.3,68.6 215.7,70.8 221.7,71.7 233.5,70.4 237.1,73.1 "
    "240.7,70.8 237.3,69.0 239.4,67.5 246.2,66.8 252.2,70.3 255.9,69.9 261.8,71.9 271.9,71.3 271."
    "5,68.6 274.4,67.8 279.6,69.3 279.6,73.5 281.7,70.0 284.4,70.1 285.9,65.7 278.4,61.2 278.7,56"
    ".3 282.6,53.1 290.4,55.8 294.9,60.7 292.0,62.9 298.2,63.8ZM1078.5,429.9 1079.7,433.3 1081.9,"
    "431.7 1084.6,435.3 1088.0,452.7 1096.2,459.0 1098.9,467.6 1100.3,466.6 1101.6,468.5 1102.4,4"
    "67.9 1103.0,472.5 1109.5,480.5 1110.5,484.0 1110.3,489.2 1111.9,493.0 1110.3,505.3 1108.2,51"
    "2.5 1105.7,514.7 1101.1,526.2 1100.0,533.9 1098.1,535.5 1094.4,535.6 1087.7,541.0 1082.9,538"
    ".3 1083.4,536.0 1078.7,540.0 1068.8,536.5 1066.6,533.8 1065.3,528.3 1063.6,526.5 1060.4,526."
    "0 1061.5,523.8 1060.7,520.6 1059.1,523.6 1056.1,524.4 1059.6,517.3 1059.4,514.0 1054.6,519.3"
    " 1053.3,522.8 1050.7,521.0 1050.8,518.6 1047.0,513.8 1047.6,512.8 1037.8,507.9 1031.8,508.3 "
    "1023.7,511.3 1020.5,511.0 1014.1,514.3 1012.2,518.4 999.6,518.8 993.4,523.5 988.7,523.4 983."
    "4,519.7 983.5,517.2 985.7,515.6 985.6,508.4 983.5,498.9 977.8,484.2 979.3,486.1 978.1,482.0 "
    "980.8,485.0 978.0,476.6 980.5,465.0 980.7,468.4 982.2,465.4 989.0,460.4 991.5,460.6 997.5,45"
    "7.1 1002.9,455.9 1007.5,449.4 1007.7,445.2 1010.0,441.5 1011.4,445.3 1012.9,444.4 1011.7,442"
    ".3 1012.7,440.2 1014.2,441.2 1014.6,437.8 1019.0,431.9 1020.4,432.4 1023.6,430.1 1027.9,434."
    "7 1032.1,435.2 1031.4,432.8 1035.4,424.5 1037.4,422.9 1041.9,422.6 1041.9,420.4 1039.4,418.9"
    " 1041.2,418.3 1051.0,423.2 1055.0,421.5 1056.5,423.7 1053.2,428.0 1051.7,435.3 1067.4,447.2 "
    "1069.6,445.7 1072.3,435.5 1071.7,429.6 1073.8,417.9 1075.1,416.3 1076.0,418.4 1078.5,429.9ZM"
    "509.7,2.1 530.5,5.6 524.4,7.3 493.7,7.9 495.3,8.7 507.1,8.2 517.2,9.7 523.7,8.4 526.4,10.0 5"
    "22.8,12.5 547.4,9.2 557.4,10.0 559.3,11.9 543.8,16.1 533.2,16.8 540.9,17.0 534.3,23.1 534.4,"
    "28.0 538.4,30.8 527.7,32.4 533.9,34.7 534.7,38.5 531.1,38.9 535.4,42.7 528.0,43.0 531.9,44.8"
    " 530.8,46.3 521.4,47.0 525.6,50.0 525.7,52.0 519.1,50.1 517.4,51.3 526.2,55.1 527.5,58.6 521"
    ".5,59.5 514.9,55.3 516.0,58.2 512.1,60.6 525.5,61.0 507.5,68.3 494.1,69.8 486.0,76.2 478.8,7"
    "9.2 467.3,81.5 464.4,84.2 462.7,90.2 457.3,93.7 458.6,97.2 455.4,105.1 450.7,105.4 445.8,101"
    ".8 439.1,101.8 427.9,89.6 425.7,82.8 421.1,78.7 422.3,75.5 420.1,73.9 423.4,68.8 428.4,67.1 "
    "430.4,61.9 421.8,64.7 417.7,63.3 418.8,58.0 428.7,59.1 420.0,54.8 416.7,55.4 413.9,54.3 417."
    "6,50.2 408.9,40.8 404.7,39.1 404.7,37.3 395.8,34.7 371.7,34.9 362.0,30.7 377.5,29.1 355.7,26"
    ".2 356.1,24.5 381.0,20.2 382.3,18.7 373.3,17.1 376.2,15.3 392.5,11.8 391.2,9.8 409.3,8.0 419"
    ".5,7.9 423.2,9.3 432.0,6.9 451.6,10.3 443.7,7.9 444.1,6.0 455.3,3.4 467.0,3.6 471.3,2.0 483."
    "0,1.6 509.7,2.1ZM0.0,442.2 1197.9,443.2 1195.8,444.2 1195.3,442.5 1198.0,441.4 0.7,439.8 0.0"
    ",442.2ZM311.5,47.7 314.1,50.4 317.2,46.9 325.6,45.1 331.3,49.6 330.8,52.5 340.6,49.5 352.6,5"
    "3.8 353.0,55.7 359.2,54.7 362.7,57.5 373.6,61.0 376.8,65.1 370.6,67.2 383.8,71.0 388.6,75.1 "
    "393.8,75.4 392.8,78.4 386.9,83.6 377.6,77.4 373.3,78.0 372.9,80.5 382.3,86.3 384.4,90.6 383."
    "3,93.8 370.7,89.1 379.4,97.0 363.3,92.7 359.2,90.6 360.4,89.4 350.5,85.0 350.6,86.2 341.0,86"
    ".9 338.2,85.4 340.3,82.2 353.5,81.5 352.4,80.0 357.8,73.5 356.9,71.6 350.5,67.9 343.8,66.4 3"
    "45.9,65.3 342.4,62.6 336.8,60.8 335.0,62.1 329.0,62.7 304.4,59.8 301.6,58.2 305.1,56.2 300.4"
    ",56.2 299.3,51.7 301.9,47.8 305.3,46.0 313.9,44.8 311.5,47.7ZM371.7,3.9 387.7,4.8 393.8,6.0 "
    "393.7,7.2 374.5,11.0 381.7,11.0 368.4,14.9 362.7,18.5 343.6,20.6 348.2,21.1 345.9,21.9 348.7"
    ",24.1 334.1,29.9 334.6,30.9 340.3,30.7 340.4,31.8 331.5,34.4 301.7,33.1 301.3,31.0 307.4,30."
    "0 305.8,26.8 316.7,28.4 312.2,25.6 306.8,24.8 315.4,22.0 316.3,20.5 311.6,18.7 310.2,16.5 32"
    "2.0,17.2 327.2,15.5 308.0,15.3 302.1,13.8 295.4,10.8 294.7,9.3 315.0,5.9 322.7,7.4 325.3,5.0"
    " 335.6,3.8 371.7,3.9ZM1047.1,374.4 1048.1,381.5 1051.5,384.2 1054.3,379.5 1058.1,376.9 1061."
    "1,376.8 1081.9,386.3 1086.1,390.8 1086.6,393.4 1092.2,396.1 1093.0,398.4 1089.9,398.9 1090.6"
    ",401.8 1095.8,409.4 1097.7,409.3 1097.6,411.2 1102.7,414.6 1102.3,415.9 1093.0,413.9 1086.8,"
    "404.8 1082.5,402.9 1077.6,405.6 1078.0,408.9 1075.4,410.4 1070.1,409.5 1067.1,405.8 1063.8,4"
    "05.0 1062.9,406.2 1058.7,406.3 1060.1,402.8 1062.2,401.5 1059.8,393.1 1050.6,389.0 1045.5,38"
    "4.9 1043.3,387.4 1042.5,383.9 1040.0,381.8 1045.9,380.3 1045.7,379.1 1040.8,379.1 1039.5,376"
    ".5 1036.5,375.7 1035.1,373.5 1041.3,371.0 1046.6,372.8 1047.1,374.4ZM992.9,361.3 996.7,365.4"
    " 992.7,365.9 991.7,372.9 988.5,375.9 987.2,387.0 986.7,385.4 982.9,387.4 981.6,384.7 977.5,3"
    "83.1 973.6,384.7 972.3,382.5 967.4,382.3 966.9,376.4 963.6,371.4 963.6,363.5 965.5,360.5 968"
    ".0,362.0 970.6,361.2 971.2,357.5 976.7,355.7 989.1,338.9 990.4,338.9 992.3,343.0 997.3,345.6"
    " 997.0,347.3 994.8,347.5 995.4,349.7 992.9,351.2 991.0,355.1 993.5,359.3 992.9,361.3ZM219.4,"
    "47.8 217.8,49.9 225.2,48.6 229.8,50.8 233.6,48.5 236.6,50.0 239.4,54.3 241.0,52.5 238.7,48.0"
    " 244.9,48.0 248.7,49.8 251.8,57.2 263.4,61.4 263.0,63.4 257.6,63.7 259.7,65.4 258.6,67.0 246"
    ".8,65.2 222.3,68.0 220.5,65.9 213.0,65.2 208.9,61.7 225.3,59.9 207.0,59.2 205.2,57.6 213.0,5"
    "5.8 202.0,54.7 207.1,49.7 216.0,47.0 219.4,47.8ZM766.9,429.0 767.9,438.4 767.3,439.7 766.2,4"
    "37.1 765.6,438.4 765.9,443.6 758.5,473.9 757.0,479.0 751.4,481.9 746.8,479.2 744.5,469.5 744"
    ".8,463.2 746.3,462.4 748.2,454.8 746.5,445.9 748.2,440.7 754.4,438.8 759.0,433.5 759.6,429.4"
    " 761.0,430.0 764.0,422.3 766.9,429.0ZM0.0,54.9 1200.0,57.9 1196.3,58.1 1195.8,56.7 0.0,54.9Z"
    "M763.7,187.8 765.4,191.0 768.0,192.3 765.2,192.7 762.9,198.7 764.0,204.1 769.5,207.2 774.2,2"
    "08.0 779.4,206.8 779.6,198.1 777.0,196.6 777.9,193.6 775.7,193.3 776.4,189.6 779.5,190.7 782"
    ".5,189.3 779.1,184.1 776.4,185.3 776.0,188.5 775.0,181.2 771.1,179.7 767.7,173.2 770.9,173.6"
    " 771.1,170.4 776.8,170.3 776.8,163.3 770.6,162.5 763.7,165.3 762.2,167.9 758.9,168.7 755.6,1"
    "73.2 758.6,177.4 758.3,180.3 763.7,187.8ZM952.7,395.1 949.0,395.2 941.9,387.9 930.9,368.6 92"
    "8.7,361.3 917.9,347.5 917.6,345.3 924.9,346.3 935.5,360.1 938.9,360.2 946.1,368.9 944.8,372."
    "5 947.9,374.1 949.6,379.7 952.1,380.0 953.7,382.8 952.7,395.1ZM590.0,111.5 586.4,116.3 593.5"
    ",115.7 592.6,119.3 589.6,123.2 593.0,123.5 596.3,129.2 598.6,129.9 601.6,136.6 605.6,137.5 6"
    "05.2,140.3 603.5,141.6 604.8,143.8 601.8,146.1 591.7,147.3 590.1,146.4 587.9,148.5 584.9,148"
    ".0 582.5,149.7 580.7,148.8 585.6,144.2 588.6,143.2 583.4,142.5 582.4,140.7 585.9,139.4 584.1"
    ",137.0 584.7,134.1 589.7,134.5 590.2,132.0 587.9,129.2 583.9,128.4 583.1,127.2 584.3,125.3 5"
    "83.2,124.1 581.4,126.1 581.2,121.9 579.5,119.7 583.3,111.6 590.0,111.5ZM1069.9,206.0 1069.2,"
    "211.8 1067.5,214.9 1063.2,216.9 1057.4,217.2 1052.6,222.2 1050.4,220.5 1050.3,217.2 1036.6,2"
    "20.4 1040.0,223.6 1037.8,231.1 1035.6,232.9 1034.0,231.2 1034.8,227.2 1031.4,223.0 1034.5,22"
    "1.6 1042.1,213.6 1048.7,212.2 1052.3,213.1 1055.7,205.3 1058.0,207.4 1064.8,201.3 1066.8,195"
    ".9 1066.3,191.0 1067.7,188.2 1071.2,187.4 1073.1,193.5 1072.9,197.1 1069.9,201.5 1069.9,206."
    "0ZM791.8,58.4 778.9,58.2 778.0,56.3 772.0,55.1 771.5,52.7 774.9,51.8 774.8,49.4 781.4,45.6 7"
    "78.4,45.1 786.3,41.2 785.4,39.2 803.9,34.1 820.7,31.6 827.2,31.0 829.5,32.8 805.3,38.4 794.9"
    ",42.6 784.7,51.1 785.4,54.8 791.8,58.4ZM551.6,77.1 550.9,80.0 554.6,83.0 550.3,86.3 537.8,90"
    ".2 524.1,88.1 527.4,86.2 520.1,84.0 526.0,83.2 525.9,81.9 518.9,80.9 521.2,78.0 526.2,77.3 5"
    "31.4,80.3 536.5,77.9 540.7,79.2 546.1,76.8 551.6,77.1ZM198.5,55.4 189.7,57.6 187.9,55.7 180."
    "2,53.3 186.9,45.4 183.6,42.7 194.9,42.0 208.1,43.2 215.0,46.3 202.6,50.5 198.5,53.6 198.5,55"
    ".4ZM309.9,19.1 313.9,20.5 303.2,25.1 297.3,25.4 290.4,24.9 286.8,23.1 286.9,21.5 289.5,20.3 "
    "283.4,20.3 277.6,16.9 282.3,13.6 285.7,13.3 284.2,12.3 292.0,12.1 296.2,14.4 307.3,16.2 309."
    "9,19.1ZM284.4,30.3 288.1,31.8 294.7,31.8 297.5,33.2 296.8,34.9 302.7,36.9 329.6,36.4 333.1,3"
    "8.1 333.9,39.9 326.8,42.0 306.2,42.2 291.9,40.3 290.4,35.7 287.0,33.8 276.3,31.9 277.5,30.1 "
    "284.4,30.3ZM660.8,18.9 671.8,22.2 663.4,23.9 661.6,27.1 658.7,28.0 657.1,31.6 653.0,31.8 645"
    ".9,29.1 648.9,27.5 637.4,22.6 634.8,19.1 643.9,17.5 645.7,19.1 650.5,19.0 651.7,17.5 656.6,1"
    "7.4 660.8,18.9ZM1176.7,549.3 1177.5,551.1 1179.9,549.3 1180.8,551.2 1180.8,553.0 1175.7,560."
    "1 1176.9,562.2 1171.5,563.9 1168.7,571.2 1164.4,574.5 1155.6,572.6 1155.0,571.0 1156.8,567.7"
    " 1168.4,558.6 1173.7,549.5 1176.0,547.4 1176.7,549.3ZM1017.5,363.1 1014.8,367.5 1012.3,368.3"
    " 1009.1,367.5 1000.6,368.3 1000.1,371.6 1003.1,375.6 1004.9,373.6 1011.1,372.1 1010.9,374.1 "
    "1009.4,373.5 1005.0,377.7 1008.2,383.4 1007.6,384.9 1010.6,390.0 1010.5,392.8 1008.8,394.1 1"
    "007.5,392.6 1009.1,389.0 1005.8,390.7 1005.0,389.5 1005.4,387.8 1003.0,385.2 1003.2,380.9 10"
    "01.0,382.2 1001.4,393.7 999.3,394.3 997.9,393.0 998.3,384.7 996.9,384.7 995.9,381.7 1000.1,3"
    "66.9 1003.0,363.6 1009.8,365.5 1013.6,365.3 1016.9,362.1 1017.5,363.1ZM412.9,146.5 410.7,150"
    ".3 412.9,148.8 415.1,149.8 413.9,151.3 421.7,152.8 420.7,156.0 423.0,155.3 424.5,160.3 423.1"
    ",164.2 419.4,163.5 420.1,159.9 419.2,159.4 415.3,163.2 413.3,163.0 415.7,161.0 412.5,159.9 4"
    "02.4,160.0 401.9,158.7 404.0,157.2 402.6,156.0 405.4,153.3 408.8,146.3 413.8,142.3 415.3,142"
    ".5 412.9,146.5ZM239.3,34.3 240.6,35.9 247.1,35.3 247.6,37.5 245.6,39.6 220.9,42.2 220.4,40.8"
    " 227.4,38.9 207.6,38.6 215.3,33.1 224.7,34.6 230.6,37.2 236.4,37.5 231.7,33.3 234.7,31.7 238"
    ".2,32.2 239.3,34.3ZM933.1,22.5 925.9,23.1 916.6,21.8 911.0,20.1 908.5,17.0 903.9,16.1 919.8,"
    "12.1 934.0,18.6 933.1,22.5ZM1182.0,528.3 1184.5,533.0 1184.5,530.0 1186.0,531.2 1186.5,534.5"
    " 1189.2,535.9 1195.1,535.1 1193.2,541.6 1190.7,541.5 1186.7,550.9 1184.1,552.7 1182.2,550.9 "
    "1184.1,547.3 1183.0,544.8 1179.4,543.1 1179.5,541.5 1181.9,540.0 1182.3,533.7 1175.4,521.2 1"
    "176.7,520.9 1181.1,524.4 1182.0,528.3ZM374.2,606.1 383.2,609.9 381.7,612.1 378.5,612.3 376.8"
    ",610.8 372.8,613.9 369.2,613.4 359.1,609.0 351.1,601.7 363.0,607.1 365.8,602.1 368.8,600.3 3"
    "71.2,600.8 374.2,606.1ZM962.1,399.2 968.5,399.6 969.2,397.8 975.4,399.9 976.6,402.8 981.6,40"
    "3.6 985.7,406.2 981.9,407.8 978.2,406.1 960.9,403.5 954.8,401.7 954.3,399.8 951.2,399.5 953."
    "5,395.3 957.5,395.5 961.6,397.6 962.1,399.2ZM1079.7,175.1 1082.0,176.1 1084.4,174.2 1085.1,1"
    "79.1 1080.2,180.3 1077.3,184.7 1072.0,181.7 1070.2,186.5 1066.5,186.6 1066.1,182.2 1067.7,17"
    "8.8 1071.3,178.6 1073.2,169.1 1079.7,175.1ZM1078.8,146.2 1082.2,154.0 1077.3,152.6 1075.2,15"
    "8.9 1078.5,163.4 1078.4,166.5 1075.8,163.8 1073.6,167.2 1073.9,145.3 1072.0,141.0 1072.3,135"
    ".0 1075.4,133.0 1074.0,130.9 1075.5,130.3 1078.8,146.2ZM265.5,44.7 275.4,45.0 276.3,46.3 273"
    ".2,48.4 278.2,50.3 277.6,54.3 272.1,56.0 268.9,55.6 258.3,50.5 258.4,49.1 265.2,49.7 261.5,4"
    "6.8 265.5,44.7ZM1004.4,288.0 1006.5,289.2 1007.5,288.1 1008.4,294.2 1007.5,297.9 1005.5,299."
    "3 1005.8,306.4 1009.0,306.3 1013.2,308.8 1013.6,314.2 1009.8,309.8 1008.9,311.4 1006.8,308.7"
    " 1003.8,309.4 1002.1,308.4 1003.3,305.5 1002.3,304.5 1001.9,306.1 1000.2,303.5 999.6,297.4 1"
    "001.0,298.9 1002.4,288.0 1004.4,288.0ZM577.4,139.6 571.5,142.2 566.7,141.5 569.4,136.9 567.7"
    ",132.4 574.8,126.9 577.6,126.8 581.1,129.5 579.3,132.5 579.9,135.6 577.4,139.6ZM334.4,269.3 "
    "335.7,270.9 338.8,270.4 344.9,276.1 348.0,276.9 347.8,278.2 352.7,280.2 352.3,281.2 340.8,28"
    "2.1 343.0,279.6 339.5,278.2 337.6,274.4 327.3,271.8 326.1,270.9 327.4,269.8 324.1,269.6 319."
    "8,273.0 316.8,273.1 320.7,269.2 325.8,267.4 334.4,269.3ZM1083.6,37.1 1081.0,40.4 1063.2,41.3"
    " 1056.6,38.4 1058.4,35.4 1062.8,34.6 1071.6,34.8 1083.6,37.1ZM1021.3,332.4 1021.8,337.8 1020"
    ".7,341.8 1019.4,337.3 1017.9,339.5 1018.9,342.8 1018.0,344.8 1014.1,342.3 1013.1,339.1 1014."
    "1,337.0 1012.0,334.9 1007.0,339.0 1006.4,337.7 1007.7,334.0 1011.6,331.1 1012.8,333.1 1015.3"
    ",331.9 1015.9,330.0 1018.2,329.8 1018.0,326.4 1020.7,328.5 1021.3,332.4ZM684.8,15.8 691.4,17"
    ".3 686.4,19.7 676.7,20.2 666.9,19.5 666.3,18.3 661.5,18.2 657.9,16.2 668.2,15.0 673.0,16.0 6"
    "76.4,14.7 684.8,15.8ZM316.1,80.7 316.7,82.6 318.5,81.9 327.9,85.9 328.2,88.0 330.6,87.7 333."
    "0,89.2 330.0,90.5 324.8,89.5 323.0,87.5 314.9,92.1 313.8,89.5 309.3,90.0 312.2,87.8 313.7,80"
    ".3 316.1,80.7ZM1084.7,548.7 1087.9,550.2 1094.3,549.1 1094.5,554.3 1093.0,559.4 1091.9,558.2"
    " 1089.6,561.2 1086.8,560.9 1082.4,550.4 1082.5,548.3 1084.7,548.7ZM289.3,49.4 285.8,52.7 282"
    ".0,52.5 279.9,48.6 281.7,44.6 285.0,43.4 298.3,44.6 293.3,48.5 289.3,49.4ZM358.1,282.0 366.8"
    ",283.0 372.3,287.5 371.0,289.3 366.8,288.3 364.9,289.4 364.4,288.3 362.0,292.0 361.0,290.0 3"
    "58.8,289.3 353.6,290.1 351.8,288.7 352.1,287.3 357.7,288.3 358.9,287.3 357.4,283.7 355.3,283"
    ".0 356.0,281.8 358.1,282.0ZM271.7,32.0 274.2,34.0 272.8,39.6 267.3,40.0 263.7,39.3 263.8,36."
    "8 258.3,37.1 258.1,33.7 271.7,32.0ZM212.7,27.9 212.2,31.3 209.6,32.8 195.0,35.6 190.5,34.7 2"
    "03.0,28.5 212.7,27.9ZM870.7,342.1 867.8,343.1 866.2,339.6 865.7,333.3 867.2,326.2 869.5,328."
    "6 872.6,336.3 872.1,340.9 870.7,342.1ZM950.3,25.0 931.5,26.7 937.5,21.0 940.3,20.5 951.2,23."
    "2 950.3,25.0ZM266.5,25.0 267.8,26.8 249.4,24.7 252.6,23.4 248.6,22.3 248.4,20.7 263.9,22.9 2"
    "66.5,25.0ZM188.3,156.1 186.6,156.7 181.2,154.7 176.6,150.3 173.1,149.5 172.1,146.1 180.8,148"
    ".2 188.3,156.1ZM345.5,47.9 345.8,49.1 335.0,49.5 330.4,46.9 330.6,45.3 339.8,45.5 345.5,47.9"
    "ZM651.7,201.3 650.3,208.3 641.4,204.0 641.9,201.7 651.7,201.3ZM967.8,287.2 964.9,289.3 962.2"
    ",288.0 962.1,284.2 963.7,282.2 969.3,281.1 970.0,282.8 967.8,287.2ZM281.2,65.5 279.1,67.0 27"
    "1.9,66.2 267.3,64.2 272.6,60.9 281.2,65.5ZM1106.6,393.5 1100.8,397.1 1094.4,394.6 1094.7,393"
    ".3 1099.5,393.6 1100.5,391.4 1100.8,393.7 1102.7,393.4 1105.5,390.3 1105.1,387.7 1107.8,388."
    "3 1106.6,393.5ZM630.7,188.2 632.7,191.3 632.2,197.1 629.4,198.3 628.1,197.1 627.2,189.3 630."
    "7,188.2ZM1003.9,269.1 1002.5,272.8 1000.4,265.8 1005.0,258.1 1006.5,259.4 1003.9,269.1ZM770."
    "5,15.2 758.6,17.5 755.0,16.5 756.9,15.1 749.5,15.0 761.1,14.1 761.7,15.3 766.8,13.6 771.7,14"
    ".5 770.5,15.2ZM682.4,27.0 675.0,28.8 669.1,27.8 671.4,26.7 669.4,25.3 676.3,24.4 677.6,26.0 "
    "682.4,27.0ZM404.8,594.1 407.5,596.0 406.5,597.6 402.0,598.9 400.5,597.4 397.7,599.3 396.0,59"
    "7.4 400.0,594.7 402.8,595.8 404.8,594.1ZM1014.8,413.9 1011.5,414.4 1013.3,410.2 1017.0,407.4"
    " 1024.5,406.3 1017.0,410.7 1014.8,413.9ZM1048.8,219.2 1049.2,220.7 1047.3,223.4 1046.0,222.0"
    " 1043.4,225.6 1041.2,224.3 1043.1,219.6 1045.0,220.1 1046.4,218.3 1048.8,219.2ZM288.0,39.7 2"
    "86.1,41.4 277.3,39.9 279.0,37.9 283.8,36.7 288.0,39.7ZM1029.0,364.4 1028.8,368.2 1027.1,367."
    "8 1026.6,370.5 1027.9,372.8 1027.0,373.3 1024.7,364.9 1025.3,361.4 1026.4,359.8 1026.7,362.2"
    " 1028.7,362.6 1029.0,364.4ZM347.1,74.1 343.4,74.3 342.5,72.2 344.0,69.7 347.0,69.1 349.6,70."
    "3 349.3,72.8 347.1,74.1ZM1018.3,315.9 1019.3,320.8 1016.7,319.6 1017.6,323.8 1016.0,324.8 10"
    "14.3,318.8 1016.3,319.2 1016.3,317.5 1014.2,314.1 1017.4,314.2 1018.3,315.9ZM1013.3,324.2 10"
    "10.0,329.7 1007.9,326.6 1009.8,321.5 1011.7,321.3 1011.1,324.2 1013.6,320.0 1013.3,324.2ZM28"
    "0.6,26.1 275.6,27.0 272.9,26.0 271.2,22.5 281.5,24.5 280.6,26.1ZM1102.4,39.2 1098.6,40.9 108"
    "7.1,38.8 1087.9,37.4 1102.4,39.2ZM1157.1,466.8 1155.8,467.9 1151.6,464.7 1146.8,457.8 1150.1"
    ",459.3 1157.1,466.8ZM90.0,118.2 86.7,119.9 84.4,116.7 89.2,114.5 91.5,114.8 92.9,116.1 90.0,"
    "118.2ZM157.6,131.7 160.8,131.4 159.8,136.4 162.7,139.9 161.4,139.9 156.5,134.5 156.1,131.2 1"
    "57.6,131.7ZM232.7,27.7 226.5,29.0 221.6,27.6 229.1,25.7 233.8,26.4 232.7,27.7ZM995.0,328.4 9"
    "90.6,332.6 998.4,319.4 999.0,323.0 995.0,328.4ZM834.3,587.9 829.1,588.2 829.8,583.2 835.1,58"
    "5.1 834.3,587.9ZM1009.7,405.0 1009.2,407.4 1004.2,408.6 999.7,408.1 999.7,406.5 1002.4,405.6"
    " 1004.5,406.9 1009.7,405.0ZM642.3,124.8 640.3,128.4 636.8,125.9 636.3,124.1 641.2,122.6 642."
    "3,124.8ZM1034.9,383.0 1036.1,386.3 1033.3,384.5 1026.3,384.3 1027.1,381.9 1034.9,383.0ZM993."
    "0,405.0 996.3,405.8 997.1,407.6 989.1,409.1 990.3,406.5 992.1,406.5 993.0,405.0ZM1006.3,317."
    "1 1010.4,318.4 1010.3,320.3 1006.7,323.5 1006.3,317.1ZM1078.7,47.4 1066.2,46.7 1073.5,44.6 1"
    "078.7,47.4ZM251.7,46.5 248.7,49.4 243.5,46.3 249.1,45.6 251.7,46.5ZM715.3,212.5 713.0,214.4 "
    "713.3,215.6 709.9,217.3 708.3,216.8 707.5,215.0 715.3,212.5ZM1110.5,389.1 1109.4,390.3 1108."
    "0,386.0 1102.2,381.4 1103.1,380.4 1107.5,383.6 1110.5,389.1ZM679.0,212.4 680.8,213.8 687.6,2"
    "14.1 687.2,215.4 682.4,215.8 678.4,214.2 679.0,212.4ZM1119.6,399.4 1118.7,399.8 1115.8,395.3"
    " 1115.5,391.5 1120.1,398.1 1119.6,399.4ZM27.6,88.9 37.7,91.0 34.9,92.4 31.1,90.7 28.2,90.9 2"
    "7.6,88.9ZM387.8,164.7 393.3,165.1 390.4,167.2 386.2,165.4 385.4,163.9 386.6,162.5 387.8,164."
    "7ZM343.7,290.8 340.8,290.8 338.9,289.2 339.3,288.2 343.7,288.5 346.0,290.7 343.7,290.8ZM1194"
    ".6,445.6 1195.7,446.9 1195.2,449.2 1191.3,449.2 1192.2,445.8 1194.6,445.6ZM631.9,184.0 630.8"
    ",187.4 628.5,183.6 631.3,180.2 631.9,184.0ZM0.0,54.9 8.1,56.0 0.0,57.9 0.0,54.9ZM327.0,93.6 "
    "323.1,96.0 320.0,94.7 322.5,92.7 327.0,93.6ZM0.0,650.0 400.4,650.0 407.3,647.6 409.3,648.7 4"
    "08.0,650.0 339.9,650.0 1194.3,650.0 0.0,650.0ZM81.5,285.4 80.2,285.6 80.5,280.2 84.0,283.6 8"
    "1.5,285.4ZM1005.1,311.9 1004.2,315.7 1001.1,310.1 1003.9,310.3 1005.1,311.9ZM381.4,289.2 380"
    ".5,290.3 376.0,290.4 375.9,288.6 379.1,287.9 381.4,289.2ZM394.0,153.4 388.0,152.1 384.9,150."
    "1 390.5,150.8 394.0,153.4ZM396.9,324.9 393.5,325.0 394.4,322.0 397.0,321.6 396.9,324.9ZM287."
    "2,28.5 279.4,28.3 278.5,27.1 285.3,27.2 287.2,28.5ZM48.1,105.9 46.0,106.6 41.8,104.6 47.7,10"
    "4.2 48.1,105.9ZM341.5,264.9 338.6,261.3 339.4,258.5 340.4,258.7 341.5,264.9ZM1024.2,384.6 10"
    "22.9,386.0 1020.6,385.2 1020.0,383.3 1023.3,383.1 1024.2,384.6ZM234.5,23.7 224.9,24.6 228.3,"
    "22.6 234.5,23.7ZM1002.4,414.4 996.6,411.4 999.7,410.5 1002.4,414.4Z"
)

# ==========================================================================
# STATIC MAP — The destination map (background layer for the interactive map).
# ==========================================================================

MAP_W, MAP_H = 1200.0, 620.0
LAT_TOP, LAT_BOT = 84.0, -57.0


def project_flat(lat: float, lon: float) -> tuple[float, float]:
    """Equirectangular projection matching the baked outline."""
    x = (lon + 180.0) / 360.0 * MAP_W
    y = (LAT_TOP - lat) / (LAT_TOP - LAT_BOT) * MAP_H
    return x, y


# Bounding boxes used to zoom the viewBox.
FOCUS: dict[str, tuple[float, float, float, float]] = {
    # key: (lat_north, lat_south, lon_west, lon_east)
    "region_africa": (38.0, -36.0, -20.0, 53.0),
    "region_asia": (56.0, -11.0, 40.0, 148.0),
    "region_europe": (71.0, 34.0, -12.0, 45.0),
    "region_middle_east": (43.0, 11.0, 24.0, 66.0),
    "region_north_america": (72.0, 7.0, -170.0, -52.0),
    "region_oceania": (0.0, -48.0, 110.0, 180.0),
    "region_south_america": (14.0, -56.0, -82.0, -34.0),
}


def _viewbox(focus: str | None) -> str:
    """The visible window, widened slightly so pins never touch the edge."""
    box = FOCUS.get(focus or "")
    if not box:
        return f"0 0 {MAP_W:.0f} {MAP_H:.0f}"
    north, south, west, east = box
    x0, y0 = project_flat(north, west)
    x1, y1 = project_flat(south, east)
    pad_x, pad_y = (x1 - x0) * 0.06, (y1 - y0) * 0.08
    return (f"{x0 - pad_x:.0f} {y0 - pad_y:.0f} "
            f"{(x1 - x0) + pad_x * 2:.0f} {(y1 - y0) + pad_y * 2:.0f}")


def _graticule() -> str:
    """Faint meridians and parallels — reads as a chart, not a picture."""
    lines = []
    for lon in range(-150, 181, 30):
        x, _ = project_flat(0, lon)
        lines.append(f'<line x1="{x:.0f}" y1="0" x2="{x:.0f}" y2="{MAP_H:.0f}"/>')
    for lat in range(-40, 81, 20):
        _, y = project_flat(lat, 0)
        lines.append(f'<line x1="0" y1="{y:.0f}" x2="{MAP_W:.0f}" y2="{y:.0f}"/>')
    return (f'<g stroke="#7FB4E6" stroke-width="1" opacity=".22">{"".join(lines)}</g>')


def _route(points: list[tuple[float, float]]) -> str:
    """A soft arc through the ranked destinations, best match first."""
    if len(points) < 2:
        return ""
    d = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        ox, oy = -dy / length, dx / length
        if oy > 0:
            ox, oy = -ox, -oy
        bow = min(length * 0.22, 90.0)
        d += f" Q{mx + ox * bow:.1f},{my + oy * bow:.1f} {x2:.1f},{y2:.1f}"
    return d


def world_map(points: list[dict], height: int = 420, routed: bool = False,
              focus: str | None = None) -> str:
    """Static fallback map (no pan/zoom/JS) for destinations.

    Used only if the browser cannot run the interactive results map component.
    """
    uid = hashlib.sha256(
        "|".join(str(p.get("label", "")) for p in points).encode("utf-8")
    ).hexdigest()[:8]

    placed = []
    for i, p in enumerate(points):
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        x, y = project_flat(lat, lon)
        placed.append((x, y, str(p.get("label", "")), int(p.get("rank", i))))

    markers = []
    for n, (x, y, label, rank) in enumerate(placed):
        top = rank == 0
        r = 7.0 if top else 5.0
        delay = (n % 6) * 0.35
        markers.append(
            f'<g class="tw-map__pin" style="animation-delay:{delay:.2f}s">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r * 3.2:.1f}" '
            f'fill="url(#halo{uid})" class="tw-map__halo"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 3:.1f}" fill="#FFFFFF" opacity=".9"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="url(#pin{uid})"/>'
            + (f'<text x="{x:.1f}" y="{y - r - 11:.1f}" text-anchor="middle" '
               f'class="tw-map__label{"" if top else " tw-map__label--hover"}">'
               f'{escape(label)}</text>' if label else "")
            + f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 12:.1f}" fill="transparent"/>'
            + "</g>"
        )

    route = ""
    if routed and len(placed) > 1:
        d = _route([(x, y) for x, y, _, _ in placed])
        route = (
            f'<path d="{d}" fill="none" stroke="url(#route{uid})" stroke-width="7" '
            f'opacity=".22" filter="url(#soft{uid})"/>'
            f'<path class="tw-map__route" d="{d}" fill="none" '
            f'stroke="url(#route{uid})" stroke-width="2.4" stroke-linecap="round" '
            f'stroke-dasharray="6 9"/>'
        )

    return f"""
<div class="tw-map" style="--map-h:{height}px">
<svg viewBox="{_viewbox(focus)}" xmlns="http://www.w3.org/2000/svg"
preserveAspectRatio="xMidYMid slice">
<defs>
<linearGradient id="sea{uid}" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="#E7F3FD"/><stop offset="55%" stop-color="#CFE7FA"/>
<stop offset="100%" stop-color="#B9DAF6"/>
</linearGradient>
<linearGradient id="land{uid}" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="#8FC3EC"/><stop offset="100%" stop-color="#5E9FD4"/>
</linearGradient>
<radialGradient id="halo{uid}">
<stop offset="0%" stop-color="#0EA5E9" stop-opacity=".5"/>
<stop offset="100%" stop-color="#0EA5E9" stop-opacity="0"/>
</radialGradient>
<radialGradient id="pin{uid}">
<stop offset="0%" stop-color="#38BDF8"/><stop offset="100%" stop-color="#2563EB"/>
</radialGradient>
<linearGradient id="route{uid}" x1="0" y1="0" x2="1" y2="0">
<stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/>
</linearGradient>
<filter id="soft{uid}" x="-20%" y="-20%" width="140%" height="140%">
<feGaussianBlur stdDeviation="7"/>
</filter>
<filter id="drop{uid}" x="-10%" y="-10%" width="120%" height="120%">
<feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#1E3A8A" flood-opacity=".28"/>
</filter>
</defs>
<rect width="{MAP_W:.0f}" height="{MAP_H:.0f}" fill="url(#sea{uid})"/>
{_graticule()}
<path d="{WORLD_PATH}" fill="url(#land{uid})" fill-rule="evenodd"
filter="url(#drop{uid})" opacity=".95"/>
{route}
{''.join(markers)}
</svg>
</div>"""


def normalise_slug(text: str | None) -> str:
    """City-name slug shared between destination cards and the results map pins."""
    return normalise(text)


def results_interactive_map(entries: list[dict], height: int = 460) -> str:
    """Interactive results map: pan/zoom, rich tooltip, click-to-scroll.

    `entries` are dicts with lat, lon, city, country, match, cost, temp, slug,
    rank (0 = best match). Rendered as a standalone HTML document meant for
    components.html, since st.markdown cannot run JavaScript.
    """
    uid = hashlib.sha256(
        "|".join(str(e.get("slug", "")) for e in entries).encode("utf-8")
    ).hexdigest()[:8]

    pins = []
    for e in entries:
        try:
            lat, lon = float(e["lat"]), float(e["lon"])
        except (TypeError, ValueError, KeyError):
            continue
        x, y = project_flat(lat, lon)
        pins.append({
            "x": round(x, 1), "y": round(y, 1),
            "city": str(e.get("city", "")), "country": str(e.get("country", "")),
            "match": e.get("match"), "cost": e.get("cost"), "temp": e.get("temp"),
            "slug": str(e.get("slug", "")), "rank": int(e.get("rank", 0)),
        })

    ranked = sorted(pins, key=lambda p: p["rank"])
    route_d = ""
    if len(ranked) > 1:
        d = f"M{ranked[0]['x']},{ranked[0]['y']}"
        for a, b in zip(ranked, ranked[1:]):
            mx, my = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            length = max((dx * dx + dy * dy) ** 0.5, 1.0)
            ox, oy = -dy / length, dx / length
            if oy > 0:
                ox, oy = -ox, -oy
            bow = min(length * 0.22, 90.0)
            d += f" Q{mx + ox * bow:.1f},{my + oy * bow:.1f} {b['x']:.1f},{b['y']:.1f}"
        route_d = d

    pins_json = json.dumps(pins, ensure_ascii=False)
    route_markup = ""
    if route_d:
        route_markup = (
            f'<path class="route-glow" d="{route_d}" fill="none" '
            f'stroke="url(#route{uid})" stroke-width="7"/>'
            f'<path class="route-line" d="{route_d}" fill="none" '
            f'stroke="url(#route{uid})" stroke-width="2.4" stroke-linecap="round"/>'
        )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@700;800&family=Inter:wght@400;500;600&family=Cairo:wght@500;600;700&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{height:100%;overflow:hidden;font-family:'Inter','Cairo',sans-serif}}
#wrap{{
  position:relative;height:100%;border-radius:22px;overflow:hidden;
  border:1px solid rgba(255,255,255,.75);
  box-shadow:0 1px 2px rgba(9,16,32,.04),0 10px 30px rgba(37,99,235,.1);
  background:linear-gradient(180deg,#E7F3FD 0%,#CFE7FA 55%,#B9DAF6 100%);
  cursor:grab;
}}
#wrap.drag{{cursor:grabbing}}
svg{{width:100%;height:100%;display:block;touch-action:none}}
.land{{opacity:.95}}
.grat{{stroke:#7FB4E6;stroke-width:.7;opacity:.22;fill:none}}
.route-glow{{opacity:.22}}
.route-line{{stroke-dasharray:6 9;animation:flow 22s linear infinite}}
@keyframes flow{{to{{stroke-dashoffset:-300}}}}
.pin-g{{cursor:pointer}}
.pin-halo{{animation:pulse 3s ease-out infinite;transform-origin:center}}
@keyframes pulse{{0%{{opacity:.55;transform:scale(.55)}}70%,100%{{opacity:0;transform:scale(1.35)}}}}
.pin-dot{{transition:transform .18s ease}}
.pin-g:hover .pin-dot{{transform:scale(1.28)}}
.pin-label{{font-family:'Inter',sans-serif;font-size:13px;font-weight:700;
  fill:#0F172A;paint-order:stroke;stroke:#FFFFFF;stroke-width:4px;stroke-linejoin:round}}

#tip{{
  position:absolute;pointer-events:none;opacity:0;transform:translate(-50%,-135%);
  background:rgba(255,255,255,.97);border:1px solid rgba(255,255,255,.9);
  border-radius:12px;padding:.55rem .7rem;box-shadow:0 10px 26px rgba(30,58,138,.24);
  transition:opacity .15s ease;min-width:150px;
}}
#tip.show{{opacity:1}}
#tip .city{{font-family:'Outfit',sans-serif;font-weight:800;font-size:.88rem;color:#0F172A}}
#tip .country{{font-size:.68rem;color:#64748B;margin-bottom:.3rem}}
#tip .row{{display:flex;gap:.5rem;margin-top:.2rem}}
#tip .kpi b{{color:#2563EB;font-weight:700;font-size:.78rem}}
#tip .kpi span{{display:block;font-size:.56rem;letter-spacing:.06em;
  text-transform:uppercase;color:#64748B}}
#tip .hint{{margin-top:.35rem;font-size:.62rem;color:#0EA5E9;font-weight:600}}

#ctl{{position:absolute;top:12px;right:12px;display:flex;flex-direction:column;gap:.32rem}}
#ctl button{{
  width:32px;height:32px;border-radius:10px;border:1px solid rgba(255,255,255,.85);
  background:rgba(255,255,255,.92);color:#2563EB;font-size:1rem;font-weight:700;
  cursor:pointer;box-shadow:0 6px 16px rgba(30,58,138,.16);
  transition:transform .18s ease,background .18s ease;
}}
#ctl button:hover{{transform:translateY(-1px);background:#fff}}
#hint{{
  position:absolute;top:12px;left:12px;font-size:.68rem;color:#3F5069;
  background:rgba(255,255,255,.82);border:1px solid rgba(255,255,255,.85);
  border-radius:99px;padding:.28rem .65rem;box-shadow:0 6px 16px rgba(30,58,138,.12);
}}
@media (max-width:480px){{#hint{{display:none}}}}
</style></head><body>
<div id="wrap">
<svg id="map" viewBox="0 0 {MAP_W:.0f} {MAP_H:.0f}" preserveAspectRatio="xMidYMid slice">
<defs>
<linearGradient id="landg{uid}" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="#8FC3EC"/><stop offset="100%" stop-color="#5E9FD4"/>
</linearGradient>
<radialGradient id="halo{uid}"><stop offset="0%" stop-color="#0EA5E9" stop-opacity=".5"/>
<stop offset="100%" stop-color="#0EA5E9" stop-opacity="0"/></radialGradient>
<radialGradient id="pin{uid}"><stop offset="0%" stop-color="#38BDF8"/>
<stop offset="100%" stop-color="#2563EB"/></radialGradient>
<linearGradient id="route{uid}" x1="0" y1="0" x2="1" y2="0">
<stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/></linearGradient>
</defs>
<g id="cam">
<g class="grat">{_graticule()}</g>
<path class="land" d="{WORLD_PATH}" fill="url(#landg{uid})"/>
{route_markup}
<g id="pins"></g>
</g>
</svg>
<div id="hint">اسحب للتحريك · مرّر للتكبير · اضغط دبوس للوصول لكاردها</div>
<div id="ctl"><button id="zin">+</button><button id="zout">&minus;</button><button id="reset">&#8634;</button></div>
<div id="tip"></div>
</div>
<script>
const PINS = {pins_json};
const wrap = document.getElementById('wrap');
const cam = document.getElementById('cam');
const tip = document.getElementById('tip');
const pinsG = document.getElementById('pins');

PINS.forEach((p, i) => {{
  const top = p.rank === 0;
  const r = top ? 7.2 : 5.2;
  const g = document.createElementNS('http://www.w3.org/2000/svg','g');
  g.setAttribute('class','pin-g');
  g.dataset.slug = p.slug;
  g.style.animationDelay = (i % 6) * 0.3 + 's';
  g.innerHTML = `
    <circle cx="${{p.x}}" cy="${{p.y}}" r="${{r*3.2}}" fill="url(#halo{uid})" class="pin-halo"/>
    <circle cx="${{p.x}}" cy="${{p.y}}" r="${{r+3}}" fill="#fff" opacity=".9"/>
    <circle class="pin-dot" cx="${{p.x}}" cy="${{p.y}}" r="${{r}}" fill="url(#pin{uid})"/>
    ${{top ? `<text x="${{p.x}}" y="${{p.y - r - 11}}" text-anchor="middle" class="pin-label">${{p.city}}</text>` : ''}}
    <circle cx="${{p.x}}" cy="${{p.y}}" r="${{r+13}}" fill="transparent"/>
  `;
  pinsG.appendChild(g);

  g.addEventListener('mousemove', e => {{
    const rect = wrap.getBoundingClientRect();
    const kpis = [];
    if (p.match != null) kpis.push(`<div class="kpi"><b>${{Math.round(p.match)}}%</b><span>تطابق</span></div>`);
    if (p.cost != null) kpis.push(`<div class="kpi"><b>$${{p.cost}}</b><span>لليوم</span></div>`);
    if (p.temp != null) kpis.push(`<div class="kpi"><b>${{Math.round(p.temp)}}°</b><span>الحرارة</span></div>`);
    tip.innerHTML = `<div class="city">${{p.city}}</div><div class="country">${{p.country}}</div>
      <div class="row">${{kpis.join('')}}</div><div class="hint">اضغط للانتقال للكارد ↓</div>`;
    tip.style.left = (e.clientX - rect.left) + 'px';
    tip.style.top = (e.clientY - rect.top) + 'px';
    tip.classList.add('show');
  }});
  g.addEventListener('mouseleave', () => tip.classList.remove('show'));

  g.addEventListener('click', () => {{
    try {{
      const doc = window.parent.document;
      const target = doc.getElementById('dest-' + p.slug);
      if (target) {{
        target.scrollIntoView({{behavior:'smooth', block:'center'}});
        const prev = target.style.boxShadow;
        target.style.transition = 'box-shadow .3s ease';
        target.style.boxShadow = '0 0 0 3px #0EA5E9, 0 26px 60px rgba(37,99,235,.35)';
        setTimeout(() => {{ target.style.boxShadow = prev; }}, 1400);
      }}
    }} catch (err) {{ /* cross-origin fallback: no-op */ }}
  }});
}});

let scale = 1, tx = 0, ty = 0;
const apply = () => cam.setAttribute('transform', `translate(${{tx}},${{ty}}) scale(${{scale}})`);

function zoomAt(factor, cx, cy) {{
  const next = Math.min(8, Math.max(1, scale * factor));
  const k = next / scale;
  tx = cx - (cx - tx) * k;
  ty = cy - (cy - ty) * k;
  scale = next;
  clampPan();
  apply();
}}
function clampPan() {{
  const r = wrap.getBoundingClientRect();
  const limX = r.width * (scale - 1), limY = r.height * (scale - 1);
  tx = Math.min(0, Math.max(-limX, tx));
  ty = Math.min(0, Math.max(-limY, ty));
}}
wrap.addEventListener('wheel', e => {{
  e.preventDefault();
  const r = wrap.getBoundingClientRect();
  zoomAt(e.deltaY < 0 ? 1.18 : 1/1.18, e.clientX - r.left, e.clientY - r.top);
}}, {{passive:false}});

let dragging = false, sx = 0, sy = 0;
wrap.addEventListener('pointerdown', e => {{
  dragging = true; sx = e.clientX - tx; sy = e.clientY - ty;
  wrap.classList.add('drag'); wrap.setPointerCapture(e.pointerId);
}});
wrap.addEventListener('pointermove', e => {{
  if (dragging) {{ tx = e.clientX - sx; ty = e.clientY - sy; clampPan(); apply(); }}
}});
wrap.addEventListener('pointerup', () => {{ dragging = false; wrap.classList.remove('drag'); }});

document.getElementById('zin').onclick = () => {{
  const r = wrap.getBoundingClientRect(); zoomAt(1.35, r.width/2, r.height/2);
}};
document.getElementById('zout').onclick = () => {{
  const r = wrap.getBoundingClientRect(); zoomAt(1/1.35, r.width/2, r.height/2);
}};
document.getElementById('reset').onclick = () => {{ scale = 1; tx = 0; ty = 0; apply(); }};
apply();
</script></body></html>"""

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
    ("cA", 0.007, 5, 11, -1.7, 1.10, 0.75, "mDeep", 34, 12),
    ("cB", 0.011, 5, 23, -2.0, 1.24, 0.85, "mDeep", 27, 18),
    ("cC", 0.017, 6, 5, -2.5, 1.45, 1.00, "mDeck", 21, 26),
    ("cD", 0.024, 6, 37, -2.7, 1.56, 0.92, "mDeck", 16, 32),
    ("cE", 0.034, 5, 29, -3.0, 1.72, 0.90, "mLow", 12, 40),
    ("cF", 0.048, 4, 53, -3.2, 1.86, 0.65, "mLow", 9, 48),
    ("cG", 0.013, 4, 41, -2.6, 1.30, 0.34, "mHigh", 40, 10),
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
    gid = f"glass{i}"
    delay = 1.0 + i * 0.12

    clouds = []
    for n, (cid, freq, octv, seed, slope, off, op, mask, dur, dist) in enumerate(CLOUD_LAYERS):
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

      <g class="tw-shade" style="animation-delay:{delay}s">
        <rect x="{GX - 4}" y="{GY - 2}" width="{GW + 8}" height="{GH + 4}" fill="url(#shade)"/>
        <rect x="{GX - 4}" y="{GY + GH - 40}" width="{GW + 8}" height="42" fill="url(#lipG)"/>
        <rect x="{GX + GW / 2 - 30}" y="{GY + GH - 26}" width="60" height="9" rx="4.5"
              fill="url(#gripG)"/>
      </g>

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
  pointer-events:none !important;
}}
.tw-splash{{
  position:fixed; inset:0; z-index:99998; display:grid; place-items:center;
  background:linear-gradient(180deg,#DCEFFC 0%,#C9E6FA 45%,#BFE0F8 100%);
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
  filter:drop-shadow(0 30px 64px rgba(66,80,96,.22));
}}
.tw-splash .tw-shade{{
  transform:translateY(0);
  animation:twShadeUp 2.2s cubic-bezier(.72,0,.18,1) forwards;
}}
@keyframes twShadeUp{{ to{{ transform:translateY(-{GH + 60}px); }} }}
.tw-splash .tw-drift{{
  animation-name:twDrift;
  animation-timing-function:cubic-bezier(.42,0,.58,1);
  animation-iteration-count:infinite; animation-direction:alternate;
}}
@keyframes twDrift{{
  from{{ transform:translate(calc(var(--dx) * -1), 0); }}
  50%{{ transform:translate(0, -2px); }}
  to{{ transform:translate(var(--dx), 0); }}
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
    """Strip leading indentation from every line."""
    return "\n".join(line.lstrip() for line in markup.splitlines() if line.strip())

# ==========================================================================
# APPLICATION — TripWise AI — an AI travel platform.
# ==========================================================================

VIEW_KEYS = ["home", "planner", "explore"]
NAV_KEY = "nav_view"
PENDING_KEY = "nav_pending"
LANG_KEY = "lang"


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def go_to(view: str) -> None:
    """Request a section change."""
    st.session_state[PENDING_KEY] = view
    st.rerun()


def apply_pending_nav() -> None:
    """Honour a queued section change. Must run before the nav radio is built."""
    pending = st.session_state.pop(PENDING_KEY, None)
    if pending in VIEW_KEYS:
        st.session_state[NAV_KEY] = pending


def language_bar() -> None:
    """Header row: the wordmark, and the language control on the far side."""
    brand, picker = st.columns([3, 1], vertical_alignment="center")
    with brand:
        html(header())
    with picker:
        st.radio(t("nav.language"), list(LANGUAGES),
                 key=LANG_KEY, horizontal=True, label_visibility="collapsed",
                 format_func=lambda c: LANGUAGES[c])


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
    slug = normalise(str(row.get("city", "x")))

    style = trip_style(row)
    profile = row.get("profile")
    if profile and profile not in style:
        style = [profile] + style[:2]

    hotel = row.get("HotelName")
    if hotel is not None and pd.isna(hotel):
        hotel = None

    return {
        "slug": slug,
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
    """Report what the catalogue can and cannot support."""
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
    """The landing page."""
    html(hero())

    cta, _ = st.columns([1, 2.2])
    with cta:
        if st.button(t("hero.cta"), key="hero_cta", type="primary",
                     use_container_width=True):
            go_to("planner")

    html(stat_grid(catalogue_stats(df)))
    data_notice(report, airports, df)

    for col, (ic, title, body) in zip(st.columns(3, gap="medium"), [
        ("compass", t("home.c1.title"), t("home.c1.body")),
        ("sparkle", t("home.c2.title"), t("home.c2.body")),
        ("wallet", t("home.c3.title"), t("home.c3.body")),
    ]):
        with col:
            html(feature(ic, title, body))


def planner_form() -> tuple[dict, bool]:
    """Collect preferences. Every control is a native widget.

    Budget uses st.selectbox — the same dropdown control as the region picker
    in the "Where and how long" group — so every dropdown in the planner shares
    one visual language instead of budget being a slider and region a select.
    """
    with st.form("planner", border=False):
        st.markdown(f"**{t('plan.group.tastes')}**")
        answers: dict = {}
        cols = st.columns(3, gap="medium")
        for i, (key, label, hint) in enumerate(TASTES):
            with cols[i % 3]:
                answers[key] = st.slider(t(f"taste.{key}"), TASTE_MIN,
                                         TASTE_MAX, 3,
                                         help=t(f"hint.{key}"), key=f"t_{key}")

        st.markdown(f"**{t('plan.group.budget')}**")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            answers["budget"] = st.selectbox(
                t("plan.budget"), [1, 2, 3], index=1,
                format_func=lambda v: t(f"budget.{v}"))
        with c2:
            answers["temp"] = st.slider(t("plan.temp"), -5, 40, 22)
        with c3:
            answers["stars"] = st.slider(t("plan.stars"), STARS_MIN,
                                         STARS_MAX, 4.0, 0.5)

        st.markdown(f"**{t('plan.group.where')}**")
        c4, c5, c6 = st.columns(3, gap="medium")
        with c4:
            options = [None] + REGION_COLS
            answers["region"] = st.selectbox(
                t("plan.region"), options,
                format_func=lambda c: t("plan.anywhere") if c is None else t(f"region.{c}"))
        with c5:
            answers["nights"] = st.slider(t("plan.nights"), NIGHTS_MIN, 30, 7)
            answers["trip_length"] = "short" if answers["nights"] <= 4 else "week"
        with c6:
            answers["travellers"] = st.number_input(
                t("plan.travellers"), TRAVELLERS_MIN, TRAVELLERS_MAX, 2)

        answers["needs_airport"] = st.checkbox(
            t("plan.airport"), value=True)

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(t("plan.submit"), type="primary")

    return answers, submitted


def tab_planner(df: pd.DataFrame, airports: AirportIndex) -> None:
    html(heading(t("plan.eyebrow"), t("plan.title"), t("plan.sub")))

    raw, submitted = planner_form()
    if not submitted:
        st.caption(t("plan.hint"))
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
        st.error(t("res.error"), icon="⚠️")
        return
    if result.empty:
        st.warning(t("res.none"), icon="ℹ️")
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
        (f"{frame['match'].max():.0f}%", t("res.metric.match")),
        (f"${min(costs) * nights * people:,}", t("res.metric.from")),
        (f"{len(result.profiles)}", t("res.metric.profiles")),
        (f"{frame['temp_avg_yearly'].mean():.0f}°", t("res.metric.climate")),
    ]))

    html(heading(t("res.eyebrow"), t("res.title"),
                    t("res.sub", n=f"{result.considered:,}", p=people, k=nights)))

    rows = list(frame.iterrows())
    for start in range(0, len(rows), 3):
        for col, (_, row) in zip(st.columns(3, gap="medium"), rows[start:start + 3]):
            with col, guard(f"card for {row.get('city')}"):
                html(destination(card_payload(row, prefs, airports)))

    readings: list = []
    with guard("insights"):
        readings = travel_insights(frame, prefs, answers, result.profiles)
    if readings:
        html(heading(t("ins.eyebrow"), t("ins.title")))
        for start in range(0, len(readings), 2):
            for col, (title, body) in zip(st.columns(2, gap="medium"),
                                          readings[start:start + 2]):
                with col:
                    html(insight(title, body))

    # Interactive results map: pan/zoom, rich tooltip, click-to-scroll to the
    # matching card below. Falls back to the static markdown map if the
    # component API is ever unavailable.
    with guard("result map"):
        html(heading(t("map.eyebrow"), t("map.title")))
        map_entries = [
            {
                "lat": r["latitude"], "lon": r["longitude"],
                "city": r["city"], "country": r.get("country", ""),
                "match": float(r.get("match", 0)),
                "cost": daily_cost(r), "temp": float(r.get("temp_avg_yearly", 0)),
                "slug": normalise(str(r["city"])), "rank": i,
            }
            for i, (_, r) in enumerate(frame.iterrows())
        ]
        try:
            components.html(results_interactive_map(map_entries), height=460,
                            scrolling=False)
        except (AttributeError, TypeError):
            html(world_map([
                {"lat": r["latitude"], "lon": r["longitude"], "label": r["city"], "rank": i}
                for i, (_, r) in enumerate(frame.iterrows())
            ], height=400, routed=True))

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

    html(heading(t("chart.eyebrow"), t("chart.title")))
    axis = dict(showgrid=True, gridcolor="rgba(30,58,138,.1)", zeroline=False,
                tickfont=dict(color="#3F5069", size=12),
                title_font=dict(color="#3F5069", size=12))
    layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="Inter, sans-serif", color="#0F172A", size=13),
                  margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False,
                  hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#0EA5E9",
                                  font=dict(color="#0F172A")))
    scale = ["#BAE6FD", "#0EA5E9", "#2563EB"]

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(f"**{t('chart.match')}**")
        fig = px.bar(frame.sort_values("match"), x="match", y="city",
                     orientation="h", color="match", color_continuous_scale=scale)
        fig.update_layout(**layout, xaxis=dict(**axis, title=t("chart.matchpct")),
                          yaxis=dict(**axis, title=""), coloraxis_showscale=False)
        fig.update_traces(hovertemplate="%{y}: %{x:.0f}%<extra></extra>")
        chart(fig)
    with right:
        costs = frame.assign(daily=[daily_cost(r) for _, r in frame.iterrows()])
        costs["total"] = costs["daily"] * answers["nights"] * answers["travellers"]
        st.markdown(f"**{t('chart.cost')}** — "
                    + t("chart.cost_sub", p=answers["travellers"], k=answers["nights"]))
        fig = px.bar(costs.sort_values("total"), x="total", y="city",
                     orientation="h", color="total", color_continuous_scale=scale)
        fig.update_layout(**layout, xaxis=dict(**axis, title=t("chart.usd")),
                          yaxis=dict(**axis, title=""), coloraxis_showscale=False)
        fig.update_traces(hovertemplate="%{y}: $%{x:,.0f}<extra></extra>")
        chart(fig)


def tab_explore(df: pd.DataFrame, airports: AirportIndex) -> None:
    """Browse the catalogue with filters and a data table.

    The interactive world map that used to live here has been removed by
    request; this tab is filters + table only.
    """
    html(heading(t("exp.eyebrow"), t("exp.title")))

    c1, c2, c3 = st.columns([1.4, 1, 1.2], gap="medium")
    with c1:
        regions = st.multiselect(
            t("exp.region"), REGION_COLS,
            format_func=lambda c: t(f"region.{c}"), placeholder=t("exp.allregions"))
    with c2:
        tiers = st.multiselect(t("exp.budget"), [1, 2, 3],
                               format_func=lambda v: t(f"budget.{v}"),
                               placeholder=t("exp.anybudget"))
    with c3:
        lo, hi = st.slider(t("exp.temp"), -10, 40, (-10, 40))

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

    st.caption(t("exp.count", a=f"{len(view):,}", b=f"{len(df):,}"))
    if view.empty:
        st.info(t("exp.empty"))
        return

    with st.expander(t("exp.table")):
        cols = ["city", "country", "temp_avg_yearly", "budget_level_encoded"]
        cols += [c for c in ("name", "iata") if c in view.columns]
        cols += TASTE_KEYS
        table = view[[c for c in cols if c in view.columns]].copy()
        table["budget_level_encoded"] = (table["budget_level_encoded"].astype(int)
                                         .map(lambda v: t(f"budget.{v}")))
        table = table.rename(columns={
            "city": t("exp.col.city"), "country": t("exp.col.country"),
            "temp_avg_yearly": t("exp.col.temp"),
            "budget_level_encoded": t("exp.col.budget"),
            "name": t("exp.col.airport"), "iata": t("exp.col.code")})
        st.dataframe(table, hide_index=True)


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #


def main() -> None:
    # language first: the stylesheet and every label below depend on it
    apply_pending_nav()
    set_language(st.session_state.get(LANG_KEY, DEFAULT))
    html(stylesheet(rtl=is_rtl()))

    if not st.session_state.get("seen_splash", False):
        st.session_state["seen_splash"] = True
        html(build_splash())

    language_bar()

    df = report = airports = None
    with guard("startup") as failed:
        df, report = load_catalogue()
        airports = airport_index(fingerprint(df))
    if failed or df is None:
        st.error("TripWise could not start: the destination catalogue failed to "
                 "load. Check that tripwise_data.csv sits beside app.py.",
                 icon="⚠️")
        return

    st.radio(t("nav.section"), VIEW_KEYS, key=NAV_KEY, horizontal=True,
             label_visibility="collapsed",
             format_func=lambda k: t(f"nav.{k}"))

    view = st.session_state.get(NAV_KEY, VIEW_KEYS[0])
    if view == "planner":
        tab_planner(df, airports)
    elif view == "explore":
        tab_explore(df, airports)
    else:
        tab_home(df, report, airports)

    html(footer())


if __name__ == "__main__":
    main()
