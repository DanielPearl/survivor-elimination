"""Train the elimination-prediction model.

Pipeline (mirrors the tennis bot's pattern):

  1. Load historical panel via data.historical.load_historical_panel.
  2. Hold out the most recent N seasons for evaluation.
  3. Fit two models on the training portion:
       - LogisticRegression on standardised features  (interpretable)
       - HistGradientBoostingClassifier               (predictive)
     Calibrate the GBT on a tail slice with sigmoid calibration so its
     scores are usable as probabilities (the live scorer uses these
     directly when computing edge vs Kalshi).
  4. Score the held-out seasons with both models. Pick the lower-Brier
     model as the "blended" output (in practice the calibrated GBT
     wins). Persist both alongside the metrics + coefficients.
  5. Write artifacts: model.joblib, metrics.json, model_coefficients.json,
     feature_importance.csv, holdout_predictions.csv. The dashboard
     re-uses the same readers it has for the tennis bot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                              log_loss, precision_score, recall_score,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler

from ..data.historical import load_historical_panel
from ..features.build_features import FEATURE_COLUMNS, build_features, label_column
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("survivor.models.train")


@dataclass
class ComponentMetrics:
    accuracy: float
    brier: float
    log_loss: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    threshold: float = 0.5


def _eval_predictions(y_true: np.ndarray, y_prob: np.ndarray,
                       threshold: float = 0.5) -> ComponentMetrics:
    """Compute the standard classifier metrics at the supplied threshold.

    Boots are ~9% of rows in this panel, so the default 0.5 threshold
    produces a degenerate all-zeros classifier (P/R/F1 all 0, accuracy
    97%-ish but useless). The caller tunes a per-model threshold on
    the training set (see ``_optimal_f1_threshold``) and passes it
    here for both train and test evaluation.
    """
    y_pred = (y_prob >= threshold).astype(int)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")
    return ComponentMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        brier=brier_score_loss(y_true, y_prob),
        log_loss=log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)),
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        roc_auc=roc,
        threshold=float(threshold),
    )


def _optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                            min_recall: float = 0.30) -> float:
    """Sweep thresholds in [0.05, 0.95] and pick the one that maximises
    F1 on the training probabilities, subject to recall ≥ ``min_recall``.

    The recall floor stops the optimizer from degenerating into an
    "only fire on near-certainties" classifier — that would still give
    F1 ~ 0 on the test set because we'd rarely flip a contestant.
    """
    best_threshold = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec < min_recall:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
    if best_f1 < 0:
        # No threshold met the recall floor — fall back to the
        # threshold that maximises F1 without the floor.
        for t in np.linspace(0.05, 0.95, 91):
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)
    return best_threshold


def _calibrate_gbt(model: HistGradientBoostingClassifier,
                    X: pd.DataFrame, y: np.ndarray,
                    holdout_frac: float = 0.2):
    """Sigmoid calibration on a holdout tail. Same approach as the
    tennis bot — tries the sklearn 1.6+ FrozenEstimator API and falls
    back to ``cv='prefit'`` for older installs."""
    n = len(X)
    cut = int(n * (1 - holdout_frac))
    X_fit, X_cal = X.iloc[:cut], X.iloc[cut:]
    y_fit, y_cal = y[:cut], y[cut:]
    model.fit(X_fit, y_fit)
    if len(np.unique(y_cal)) < 2:
        # Calibration needs both classes in the holdout. If the tail
        # is all 0s (which can happen on the small training panel),
        # fall back to the uncalibrated model.
        return model
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except Exception:
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def _split_by_seasons(panel: pd.DataFrame, holdout_seasons: int
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(panel["season"].unique())
    if holdout_seasons >= len(seasons):
        raise ValueError("holdout_seasons cannot exceed training seasons")
    test_seasons = set(seasons[-holdout_seasons:])
    train = panel[~panel["season"].isin(test_seasons)].copy()
    test = panel[panel["season"].isin(test_seasons)].copy()
    return train, test


def train_and_persist() -> Dict[str, Any]:
    cfg = load_config()
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg["training"]

    log.info("loading historical panel…")
    panel = load_historical_panel()
    n_seasons = panel["season"].nunique()
    log.info("panel rows=%d, seasons=%d, boots=%d",
              len(panel), n_seasons, int(panel["eliminated"].sum()))

    train_df, test_df = _split_by_seasons(panel, int(train_cfg["test_holdout_seasons"]))
    log.info("train seasons=%s, test seasons=%s",
              sorted(train_df["season"].unique()),
              sorted(test_df["season"].unique()))

    X_train = build_features(train_df)
    y_train = label_column(train_df)
    X_test = build_features(test_df)
    y_test = label_column(test_df)

    # ── Logistic baseline (interpretable + handles class imbalance via
    # the standard sklearn weighting). Standardise features so the
    # coefficient magnitudes are comparable across columns. ──────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=int(train_cfg["random_state"]),
    )
    lr.fit(X_train_s, y_train)
    lr_train_prob = lr.predict_proba(X_train_s)[:, 1]
    lr_test_prob = lr.predict_proba(X_test_s)[:, 1]
    lr_threshold = _optimal_f1_threshold(y_train, lr_train_prob)
    lr_train_metrics = _eval_predictions(y_train, lr_train_prob, lr_threshold)
    lr_metrics = _eval_predictions(y_test, lr_test_prob, lr_threshold)
    log.info("logistic threshold=%.2f train P/R/F1=%.2f/%.2f/%.2f test P/R/F1=%.2f/%.2f/%.2f auc=%.3f",
              lr_threshold,
              lr_train_metrics.precision, lr_train_metrics.recall, lr_train_metrics.f1,
              lr_metrics.precision, lr_metrics.recall, lr_metrics.f1,
              lr_metrics.roc_auc)

    # ── HistGradientBoosting (calibrated). ───────────────────────────
    hgb = HistGradientBoostingClassifier(
        max_depth=int(train_cfg["hgb_max_depth"]),
        learning_rate=float(train_cfg["hgb_learning_rate"]),
        max_iter=int(train_cfg["hgb_max_iter"]),
        l2_regularization=float(train_cfg["hgb_l2_regularization"]),
        class_weight="balanced",
        random_state=int(train_cfg["random_state"]),
    )
    gbt = _calibrate_gbt(hgb, X_train, y_train,
                          holdout_frac=float(train_cfg["calibration_holdout_fraction"]))
    gbt_train_prob = gbt.predict_proba(X_train)[:, 1]
    gbt_test_prob = gbt.predict_proba(X_test)[:, 1]
    gbt_threshold = _optimal_f1_threshold(y_train, gbt_train_prob)
    gbt_train_metrics = _eval_predictions(y_train, gbt_train_prob, gbt_threshold)
    gbt_metrics = _eval_predictions(y_test, gbt_test_prob, gbt_threshold)
    log.info("GBT threshold=%.2f train P/R/F1=%.2f/%.2f/%.2f test P/R/F1=%.2f/%.2f/%.2f auc=%.3f",
              gbt_threshold,
              gbt_train_metrics.precision, gbt_train_metrics.recall, gbt_train_metrics.f1,
              gbt_metrics.precision, gbt_metrics.recall, gbt_metrics.f1,
              gbt_metrics.roc_auc)

    # ── Blend: pick the lower-Brier model. ───────────────────────────
    if not np.isnan(gbt_metrics.brier) and gbt_metrics.brier <= lr_metrics.brier:
        blended_prob = gbt_test_prob
        blended_metrics = gbt_metrics
        blended_train_metrics = gbt_train_metrics
        best_name = "calibrated_gbt"
        best_threshold = gbt_threshold
    else:
        blended_prob = lr_test_prob
        blended_metrics = lr_metrics
        blended_train_metrics = lr_train_metrics
        best_name = "logistic"
        best_threshold = lr_threshold

    # ── Persist artifacts. ───────────────────────────────────────────
    model_path = artifacts_dir / "model.joblib"
    joblib.dump({
        "feature_columns": FEATURE_COLUMNS,
        "scaler": scaler,
        "logistic": lr,
        "calibrated_gbt": gbt,
        "best": best_name,
        "threshold": best_threshold,
    }, model_path)
    log.info("wrote %s", model_path)

    # metrics.json — same shape the tennis bot uses so the dashboard
    # adapter can lift it straight into the cross-bot card grid.
    metrics_payload: Dict[str, Any] = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "train_seasons": sorted(train_df["season"].unique().tolist()),
        "test_seasons": sorted(test_df["season"].unique().tolist()),
        # Holdout-only metrics for back-compat with the existing
        # dashboard readers (Brier / ROC AUC / etc. were already test-set).
        "logistic": asdict(lr_metrics),
        "calibrated_gbt": asdict(gbt_metrics),
        "blended": asdict(blended_metrics),
        # Train-set metrics at the same threshold — surfaces train/test
        # drift on the dashboard card alongside the holdout numbers.
        "logistic_train": asdict(lr_train_metrics),
        "calibrated_gbt_train": asdict(gbt_train_metrics),
        "blended_train": asdict(blended_train_metrics),
        "best_model": best_name,
        "threshold": float(best_threshold),
        "feature_count": len(FEATURE_COLUMNS),
        # Class-balance stats — useful context for reading F1 / recall.
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
    }
    metrics_path = resolve_path(cfg["paths"]["metrics_json"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    log.info("wrote %s", metrics_path)

    # model_coefficients.json — feeds the dashboard "Model coefficients"
    # table on the watchlist page.
    coefs_payload = {
        "logistic": {
            "features": FEATURE_COLUMNS,
            "coefficients": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
    }
    coefs_path = resolve_path(cfg["paths"]["coefficients_json"])
    with coefs_path.open("w", encoding="utf-8") as f:
        json.dump(coefs_payload, f, indent=2)
    log.info("wrote %s", coefs_path)

    # feature_importance.csv (logistic |coef| ranking — the GBT's gain-
    # based importance is harder to read for a small panel like this).
    fi_path = resolve_path(cfg["paths"]["feature_importance_csv"])
    fi_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": np.abs(lr.coef_[0]),
        "source": [_feature_source(f) for f in FEATURE_COLUMNS],
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(fi_path, index=False)
    log.info("wrote %s", fi_path)

    # holdout_predictions.csv — feeds the ROC / calibration / confusion
    # SVG widgets the dashboard already renders for tennis.
    hp_path = resolve_path(cfg["paths"]["holdout_predictions_csv"])
    hp_df = pd.DataFrame({
        "season": test_df["season"].values,
        "episode": test_df["episode"].values,
        "contestant": test_df["contestant"].values,
        "y_true": y_test,
        "y_prob": blended_prob,
    })
    hp_df.to_csv(hp_path, index=False)
    log.info("wrote %s", hp_path)

    return metrics_payload


def _feature_source(name: str) -> str:
    """Tag a feature with where its signal comes from. Used to colour
    the dashboard's Feature importance bars by source."""
    if name.startswith("reddit_"):
        return "reddit"
    if name in {"episode", "remaining", "tribe_size", "starting_tribe_size",
                "merged", "swap_phase", "season", "episode_share_remaining",
                "pre_merge_phase", "is_finale"}:
        return "structural"
    if name in {"immunity_won", "tribe_immunity", "has_idol",
                 "in_main_alliance", "prior_perf_score"}:
        return "in_game"
    return "edit"
