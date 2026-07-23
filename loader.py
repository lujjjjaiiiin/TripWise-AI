"""Flatten the package into a single deployable app.py.

Streamlit Cloud deployments frequently lose the package folder during upload,
so a one-file build removes that whole class of failure. This produces it from
the same sources, keeping the modular tree as the thing that is maintained.
"""

from __future__ import annotations

import ast
import pathlib
import re

# order matters: a module may only depend on ones already emitted
MODULES = [
    ("CONFIGURATION", "tripwise/config.py"),
    ("ERROR HANDLING", "tripwise/errors.py"),
    ("AIRPORT RESOLUTION", "tripwise/airports.py"),
    ("CATALOGUE", "tripwise/loader.py"),
    ("MODELS", "tripwise/models.py"),
    ("RECOMMENDER", "tripwise/recommender.py"),
    ("TRAVEL INTELLIGENCE", "tripwise/insights.py"),
    ("VALIDATION", "tripwise/validation.py"),
    ("DESIGN SYSTEM", "tripwise/theme.py"),
    ("DESTINATION ARTWORK", "tripwise/art.py"),
    ("HTML COMPONENTS", "tripwise/ui.py"),
    ("SPLASH", "tripwise/splash.py"),
]

# stdlib and third-party imports the flat file provides once, up top
HOISTED = {
    "os", "sys", "re", "math", "hashlib", "unicodedata", "functools", "logging",
    "dataclasses", "contextlib", "typing", "html", "numpy", "pandas",
    "streamlit", "sklearn",
}


def _import_line_span(node: ast.AST) -> range:
    return range(node.lineno - 1, (node.end_lineno or node.lineno))


def _is_droppable(node: ast.AST) -> bool:
    """True for intra-package imports and the ones hoisted into the header."""
    if isinstance(node, ast.ImportFrom):
        if node.module == "__future__":            # hoisted; must lead the file
            return True
        if node.level and node.level > 0:          # from .x import y
            return True
        root = (node.module or "").split(".")[0]
        return root == "tripwise" or root in HOISTED
    if isinstance(node, ast.Import):
        return all(a.name.split(".")[0] in HOISTED or a.name.split(".")[0] == "tripwise"
                   for a in node.names)
    return False


HEADER = '''"""
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
'''


def strip_module(path: str) -> tuple[str, str]:
    """Return (docstring first line, body with imports and docstring removed).

    Uses the AST to locate imports precisely. An earlier regex version also
    matched the closing parenthesis of ordinary multi-line expressions and
    silently corrupted the output.
    """
    src = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    lines = src.splitlines()

    drop: set[int] = set()
    doc = ""

    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        doc = body[0].value.value.strip().splitlines()[0]
        drop.update(_import_line_span(body[0]))

    for node in body:
        if _is_droppable(node):
            drop.update(_import_line_span(node))

    kept = [ln for i, ln in enumerate(lines) if i not in drop]
    return doc, "\n".join(kept).strip("\n")


def qualify(app_src: str) -> str:
    """Rewrite package-qualified names to the flat namespace."""
    app_src = app_src.replace("theme.CSS", "THEME_CSS")
    app_src = app_src.replace("loader.CatalogueReport", "CatalogueReport")
    app_src = app_src.replace("loader.load()", "load_catalogue_file()")
    app_src = app_src.replace("loader.stats", "catalogue_stats")
    app_src = app_src.replace("models.ModelBundle", "ModelBundle")
    app_src = app_src.replace("models.fit", "fit_models")
    app_src = app_src.replace("recommender.Recommendation", "Recommendation")
    app_src = app_src.replace("recommender.build_preference_vector", "build_preference_vector")
    app_src = app_src.replace("recommender.recommend", "recommend")
    app_src = app_src.replace("insights.insights", "travel_insights")
    app_src = app_src.replace("validation.", "")
    app_src = app_src.replace("insights.", "")
    app_src = app_src.replace("config.", "")
    app_src = app_src.replace("ui.", "")
    app_src = app_src.replace("build_splash()", "build_splash()")
    return app_src


def build() -> str:
    parts = [HEADER]

    for label, path in MODULES:
        doc, body = strip_module(path)

        if path.endswith("art.py"):
            body = re.sub(r"^W, H = 400, 240$", "ART_W, ART_H = 400, 240", body, flags=re.M)
            body = re.sub(r"\bW\b(?!\w)", "ART_W", body)
            body = re.sub(r"\bH\b(?!\w)", "ART_H", body)
            body = body.replace("ART_ART_", "ART_")
        if path.endswith("splash.py"):
            body = re.sub(r"^W, H = 1200, 620$", "SPL_W, SPL_H = 1200, 620", body, flags=re.M)
            body = re.sub(r"\bW\b(?!\w)", "SPL_W", body)
            body = re.sub(r"\bH\b(?!\w)", "SPL_H", body)
            body = body.replace("SPL_SPL_", "SPL_")
            body = body.replace("def build()", "def build_splash()")
        if path.endswith("theme.py"):
            body = body.replace('CSS = """', 'THEME_CSS = """', 1)
        if path.endswith("loader.py"):
            body = body.replace("def load(", "def load_catalogue_file(")
            body = body.replace("def stats(", "def catalogue_stats(")
        if path.endswith("models.py"):
            body = body.replace("def fit(", "def fit_models(")
        if path.endswith("insights.py"):
            body = body.replace("def insights(", "def travel_insights(")

        bar = "=" * 74
        parts.append(f"\n\n# {bar}\n# {label} — {doc}\n# {bar}\n\n{body}")

    doc, app_body = strip_module("app.py")
    app_body = re.sub(r"st\.set_page_config\(.*?\n\)\n", "", app_body, flags=re.S)
    app_body = "\n".join(
        line for line in app_body.splitlines()
        if "sys.path.insert" not in line
        and not line.strip().startswith("# `streamlit run` does not guarantee")
        and not line.strip().startswith("# so put it there")
    )
    app_body = qualify(app_body)

    bar = "=" * 74
    parts.append(f"\n\n# {bar}\n# APPLICATION — {doc}\n# {bar}\n\n{app_body}\n")
    return "".join(parts)


if __name__ == "__main__":
    out = build()
    pathlib.Path("app_single.py").write_text(out)
    print(f"built app_single.py: {len(out):,} chars, {out.count(chr(10)):,} lines")
