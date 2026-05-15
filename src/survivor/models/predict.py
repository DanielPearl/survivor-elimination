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

    The artifact layout supports two generations:
      - Newer: ``families`` dict carrying every CV-tuned family +
        the stacker; ``best`` is one of the keys.
      - Older: bare ``logistic`` / ``calibrated_gbt`` at the top
        level (legacy back-compat path).
    """
    art = _load_artifact()
    if art is None:
        n = len(df)
        if "remaining" in df.columns:
            r = pd.to_numeric(df["remaining"], errors="coerce").fillna(8).values
            return np.clip(1.0 / np.maximum(r, 2), 0.02, 0.5)
        return np.full(n, 1.0 / max(1, n))
    X = build_features(df)
    best = art.get("best") or "logistic"
    raw: np.ndarray
    families = art.get("families") or {}
    if best in families:
        # New artifact layout.
        fam = families[best]
        model = fam["model"]
        scaler = fam.get("scaler")
        X_arr = scaler.transform(X) if scaler is not None else X.values
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(X_arr)[:, 1]
        else:
            d = model.decision_function(X_arr)
            raw = 1.0 / (1.0 + np.exp(-d))
    else:
        # Legacy back-compat path — top-level logistic / calibrated_gbt.
        if best == "calibrated_gbt":
            raw = art["calibrated_gbt"].predict_proba(X)[:, 1]
        else:
            scaler = art["scaler"]
            X_arr = scaler.transform(X)
            raw = art["logistic"].predict_proba(X_arr)[:, 1]
    if not art.get("per_episode_normalize"):
        return raw
    from .train import normalize_per_episode
    return normalize_per_episode(df, raw)


def get_decision_threshold() -> float:
    """The threshold the trainer locked in for BUY-eligibility on the
    blended model's normalised probabilities. Live exporter reads
    this so its BUY YES verdict matches the trainer's reported F1
    operating point."""
    art = _load_artifact()
    if art is None:
        return 0.5
    return float(art.get("threshold") or 0.5)


def reset_model_cache() -> None:
    """Drop the cached artifact so the next predict call reloads it.
    Called by the trainer after a fresh fit so the live scorer sees
    the new model without restarting the process."""
    _load_artifact.cache_clear()
