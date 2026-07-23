"""The recommendation pipeline.

Three stages, each doing one thing:

1. the traveller's answers become a point in feature space
2. every destination is scored against it by cosine similarity
3. the ranking is diversified so one destination profile cannot fill the
   shortlist

Stage 3 is where K-Means earns its place. Cosine similarity alone tends to
return six variations of whichever place matched best, because near-identical
destinations sit at near-identical angles. Capping results per cluster keeps the
strongest match from each distinct profile and gives a shortlist worth choosing
between.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT_TOP_N, FEATURE_COLS, MAX_PER_CLUSTER, REGION_COLS, TASTE_KEYS
from .errors import log, to_float
from .models import ModelBundle


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
