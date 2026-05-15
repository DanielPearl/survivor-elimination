"""Inference wrapper for the trained elimination model.

Loads ``data/processed/artifacts/model.joblib`` and exposes a
``predict_eliminated_proba`` function the live scorer calls per
contestant-episode row. Falls back to a uniform-over-active prior if
the artifact is missing so the dashboard never renders a blank table.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from ..features.build_features import FEATURE_COLUMNS, build_features
from ..utils.config import load_config, resolve_path

log = logging.getLogger("survivor.models.predict")


@lru_cache(maxsize=1)
def _load_artifact() -> Dict[str, Any] | None:
    cfg = load_config()
    p = resolve_path(cfg["paths"]["artifacts_dir"]) / "model.joblib"
    if not p.exists():
        log.warning("no model artifact at %s — using uniform prior", p)
        return None
    try:
        return joblib.load(p)
    except Exception as exc:  # noqa: BLE001
        log.warning("model artifact at %s failed to load: %s", p, exc)
        return None


def predict_eliminated_proba(df: pd.DataFrame) -> np.ndarray:
    """Return P(eliminated_this_episode) for each row in `df`.

    `df` must carry the columns the historical panel does (plus the
    optional Reddit columns). Rows for already-eliminated contestants
    should not be in `df` — the live scorer filters them out.
    """
    art = _load_artifact()
    if art is None:
        n = len(df)
        # Uniform-over-active prior: 1 / remaining (rough boot-rate)
        if "remaining" in df.columns:
            r = pd.to_numeric(df["remaining"], errors="coerce").fillna(8).values
            return np.clip(1.0 / np.maximum(r, 2), 0.02, 0.5)
        return np.full(n, 1.0 / max(1, n))
    X = build_features(df)
    model = art["calibrated_gbt"] if art.get("best") == "calibrated_gbt" else art["logistic"]
    if art.get("best") == "logistic":
        scaler = art["scaler"]
        X_arr = scaler.transform(X)
        return model.predict_proba(X_arr)[:, 1]
    return model.predict_proba(X)[:, 1]


def reset_model_cache() -> None:
    """Drop the cached artifact so the next predict call reloads it.
    Called by the trainer after a fresh fit so the live scorer sees
    the new model without restarting the process."""
    _load_artifact.cache_clear()
