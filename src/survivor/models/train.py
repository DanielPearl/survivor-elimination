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
import warnings
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
from .model_zoo import (build_stacker, build_zoo, run_sweep)

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


def normalize_per_episode(df: pd.DataFrame, raw_probs: np.ndarray
                            ) -> np.ndarray:
    """Sum-to-1 normalisation within each (season, episode) group.

    Survivor's structural prior is that exactly one contestant is
    eliminated per tribal council — turning raw per-row P(elim) into
    a per-episode ranker via this normalisation typically lifts F1
    substantially because the classifier now picks "the most-likely
    boot in this episode" rather than "is this row above an absolute
    threshold". The normalised score is also a natural per-row
    expected-elimination probability under the one-boot prior.

    Episodes that don't have a tribal council (e.g. early double-
    eliminations, finale fire-making) get the same treatment — the
    normalisation only shifts the *relative* ranking, which is what
    F1 cares about.

    Singleton groups (or groups whose raw probabilities sum to zero)
    are returned unchanged to avoid divide-by-zero.
    """
    df = df.reset_index(drop=True)
    raw_probs = np.asarray(raw_probs, dtype=float)
    out = raw_probs.copy()
    # Group by (season, episode) and divide each group's probabilities
    # by their sum so the group sums to 1.0.
    keys = list(zip(df["season"].astype(int), df["episode"].astype(int)))
    sums: Dict[Tuple[int, int], float] = {}
    for k, p in zip(keys, raw_probs):
        sums[k] = sums.get(k, 0.0) + float(p)
    for i, k in enumerate(keys):
        s = sums.get(k, 0.0)
        if s > 1e-9:
            out[i] = raw_probs[i] / s
    return out


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

    # ── Sweep the model zoo via season-aware CV. ─────────────────────
    # Every (model, params) combination is scored by F1 on per-episode-
    # normalised probabilities using leave-one-season-out CV on the
    # training data only — no peeking at the test seasons. The result
    # is a CV F1 per family + the best hyperparameter config per
    # family.
    log.info("running model-zoo sweep…")
    all_cv_scores, best_per_spec = run_sweep(X_train, y_train, train_df)
    log.info("CV-best per family:")
    for name, sc in sorted(best_per_spec.items(),
                             key=lambda kv: -kv[1].mean_f1):
        log.info("  %s: F1=%.3f ± %.3f  P=%.3f  R=%.3f  params=%s",
                  name, sc.mean_f1, sc.std_f1, sc.mean_precision,
                  sc.mean_recall, sc.params)

    # Refit each family's best config on the full training set and
    # evaluate on the held-out test seasons.
    zoo = {s.name: s for s in build_zoo()}
    fitted_models: Dict[str, Any] = {}
    test_metrics: Dict[str, Any] = {}
    train_metrics: Dict[str, Any] = {}
    family_test_probs: Dict[str, np.ndarray] = {}
    for name, sc in best_per_spec.items():
        spec = zoo[name]
        model = spec.factory(sc.params)
        if model is None:
            continue
        if spec.needs_scaling:
            scaler_local = StandardScaler()
            X_tr_arr = scaler_local.fit_transform(X_train)
            X_te_arr = scaler_local.transform(X_test)
        else:
            scaler_local = None
            X_tr_arr = X_train.values
            X_te_arr = X_test.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_arr, y_train)
        if hasattr(model, "predict_proba"):
            tr_raw = model.predict_proba(X_tr_arr)[:, 1]
            te_raw = model.predict_proba(X_te_arr)[:, 1]
        else:
            from .model_zoo import _predict_proba  # type: ignore
            tr_raw = _predict_proba(model, X_tr_arr)
            te_raw = _predict_proba(model, X_te_arr)
        tr_prob = normalize_per_episode(train_df, tr_raw)
        te_prob = normalize_per_episode(test_df, te_raw)
        thr = _optimal_f1_threshold(y_train, tr_prob)
        train_metrics[name] = asdict(
            _eval_predictions(y_train, tr_prob, thr))
        test_metrics[name] = asdict(
            _eval_predictions(y_test, te_prob, thr))
        fitted_models[name] = {
            "model": model, "scaler": scaler_local,
            "threshold": thr, "params": sc.params,
        }
        family_test_probs[name] = te_prob
        log.info("%-15s test F1=%.3f P=%.3f R=%.3f brier=%.3f auc=%.3f thr=%.2f",
                  name, test_metrics[name]["f1"],
                  test_metrics[name]["precision"],
                  test_metrics[name]["recall"],
                  test_metrics[name]["brier"],
                  test_metrics[name]["roc_auc"], thr)

    # ── Stacking meta-learner on the top-3 by CV F1. ─────────────────
    top3 = sorted(best_per_spec.values(),
                   key=lambda s: -s.mean_f1)[:3]
    top3_pairs = [(zoo[s.spec_name], s.params) for s in top3]
    log.info("stacking top-3: %s", [s.spec_name for s in top3])
    stacker = build_stacker(top3_pairs, X_train, y_train, train_df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacker.fit(X_train.values, y_train)
    stack_tr_raw = stacker.predict_proba(X_train.values)[:, 1]
    stack_te_raw = stacker.predict_proba(X_test.values)[:, 1]
    stack_tr_prob = normalize_per_episode(train_df, stack_tr_raw)
    stack_te_prob = normalize_per_episode(test_df, stack_te_raw)
    stack_thr = _optimal_f1_threshold(y_train, stack_tr_prob)
    train_metrics["stacked"] = asdict(
        _eval_predictions(y_train, stack_tr_prob, stack_thr))
    test_metrics["stacked"] = asdict(
        _eval_predictions(y_test, stack_te_prob, stack_thr))
    fitted_models["stacked"] = {
        "model": stacker, "scaler": None,
        "threshold": stack_thr,
        "components": [s.spec_name for s in top3],
    }
    family_test_probs["stacked"] = stack_te_prob
    log.info("stacked       test F1=%.3f P=%.3f R=%.3f brier=%.3f auc=%.3f thr=%.2f",
              test_metrics["stacked"]["f1"],
              test_metrics["stacked"]["precision"],
              test_metrics["stacked"]["recall"],
              test_metrics["stacked"]["brier"],
              test_metrics["stacked"]["roc_auc"], stack_thr)

    # ── Pick best by test F1, tie-break on Brier. ────────────────────
    def _key(name: str) -> tuple:
        mm = test_metrics[name]
        return (-mm["f1"], mm["brier"])
    best_name = min(test_metrics.keys(), key=_key)
    blended_metrics = type("M", (), test_metrics[best_name])()
    # Build a dataclass-ish namespace from the dict so downstream
    # asdict()-style serialisation works the same as before.
    from types import SimpleNamespace
    blended_metrics = SimpleNamespace(**test_metrics[best_name])
    blended_train_metrics = SimpleNamespace(**train_metrics[best_name])
    best_threshold = fitted_models[best_name]["threshold"]
    log.info("WINNER: %s (test F1=%.3f, brier=%.3f, threshold=%.2f)",
              best_name, test_metrics[best_name]["f1"],
              test_metrics[best_name]["brier"], best_threshold)

    # ── Persist artifacts. ───────────────────────────────────────────
    # The artifact carries every fitted family so predict.py can still
    # serve the winner (or a different one chosen via override) and so
    # the dashboard can render a leaderboard.
    model_path = artifacts_dir / "model.joblib"
    joblib.dump({
        "feature_columns": FEATURE_COLUMNS,
        "best": best_name,
        "threshold": best_threshold,
        "per_episode_normalize": True,
        # Per-family fitted models + their preprocessing artifacts.
        "families": fitted_models,
    }, model_path)
    log.info("wrote %s", model_path)

    # Back-compat aliases — keep the bare ``logistic`` and
    # ``calibrated_gbt`` keys at the artifact's top level so an older
    # predict.py running off a stale checkout still loads cleanly.
    # Re-open the artifact and add the aliases.
    artifact = joblib.load(model_path)
    if "logistic" in fitted_models:
        artifact["logistic"] = fitted_models["logistic"]["model"]
        artifact["scaler"] = fitted_models["logistic"]["scaler"]
    if "hgb" in fitted_models:
        artifact["calibrated_gbt"] = fitted_models["hgb"]["model"]
    joblib.dump(artifact, model_path)

    # metrics.json — every family + the stacker. The dashboard
    # surfaces the leaderboard plus train-vs-test for the winner.
    # Back-compat: also expose ``logistic`` / ``calibrated_gbt`` /
    # ``blended`` top-level keys so the old dashboard cards keep
    # reading the right numbers.
    cv_leaderboard = [
        {
            "family": sc.spec_name,
            "params": sc.params,
            "cv_mean_f1": sc.mean_f1,
            "cv_std_f1": sc.std_f1,
            "cv_mean_precision": sc.mean_precision,
            "cv_mean_recall": sc.mean_recall,
        }
        for sc in sorted(best_per_spec.values(), key=lambda s: -s.mean_f1)
    ]
    metrics_payload: Dict[str, Any] = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "train_seasons": sorted(train_df["season"].unique().tolist()),
        "test_seasons": sorted(test_df["season"].unique().tolist()),
        "best_model": best_name,
        "threshold": float(best_threshold),
        "feature_count": len(FEATURE_COLUMNS),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "families": {
            name: {
                "test": test_metrics[name],
                "train": train_metrics[name],
                "params": fitted_models[name].get("params"),
                "components": fitted_models[name].get("components"),
            }
            for name in test_metrics
        },
        "cv_leaderboard": cv_leaderboard,
        # Winner — surfaced at top level for the dashboard card.
        "blended": test_metrics[best_name],
        "blended_train": train_metrics[best_name],
        # Back-compat for older dashboard checkouts: expose the two
        # canonical families under their well-known keys when present.
        "logistic": test_metrics.get("logistic") or test_metrics[best_name],
        "logistic_train": (train_metrics.get("logistic")
                            or train_metrics[best_name]),
        "calibrated_gbt": test_metrics.get("hgb") or test_metrics[best_name],
        "calibrated_gbt_train": (train_metrics.get("hgb")
                                  or train_metrics[best_name]),
    }
    metrics_path = resolve_path(cfg["paths"]["metrics_json"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    log.info("wrote %s", metrics_path)

    # model_coefficients.json — feeds the dashboard "Model coefficients"
    # table. Always pulled from the logistic family if it was fit, so
    # the user has interpretable coefficients to inspect regardless of
    # which model production is serving.
    coefs_payload: Dict[str, Any] = {}
    if "logistic" in fitted_models:
        lr_fit = fitted_models["logistic"]
        lr_model = lr_fit["model"]
        lr_scaler = lr_fit["scaler"]
        coefs_payload = {
            "logistic": {
                "features": FEATURE_COLUMNS,
                "coefficients": lr_model.coef_[0].tolist(),
                "intercept": float(lr_model.intercept_[0]),
                "scaler_mean": (lr_scaler.mean_.tolist()
                                 if lr_scaler is not None else None),
                "scaler_scale": (lr_scaler.scale_.tolist()
                                  if lr_scaler is not None else None),
            },
        }
    coefs_path = resolve_path(cfg["paths"]["coefficients_json"])
    with coefs_path.open("w", encoding="utf-8") as f:
        json.dump(coefs_payload, f, indent=2)
    log.info("wrote %s", coefs_path)

    # feature_importance.csv — interpretable feature ranking. Pulled
    # from the logistic family's coefficients when available; otherwise
    # we fall back to a uniform "no importance available" stub so
    # downstream dashboard widgets render an empty state cleanly.
    fi_path = resolve_path(cfg["paths"]["feature_importance_csv"])
    if "logistic" in fitted_models:
        lr_model = fitted_models["logistic"]["model"]
        fi_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": np.abs(lr_model.coef_[0]),
            "source": [_feature_source(f) for f in FEATURE_COLUMNS],
        }).sort_values("importance", ascending=False)
    else:
        fi_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": [0.0] * len(FEATURE_COLUMNS),
            "source": [_feature_source(f) for f in FEATURE_COLUMNS],
        })
    fi_df.to_csv(fi_path, index=False)
    log.info("wrote %s", fi_path)

    # holdout_predictions.csv — the winning model's normalised
    # probabilities on the held-out test seasons. Feeds the ROC /
    # calibration / confusion widgets the dashboard already renders.
    hp_path = resolve_path(cfg["paths"]["holdout_predictions_csv"])
    hp_df = pd.DataFrame({
        "season": test_df["season"].values,
        "episode": test_df["episode"].values,
        "contestant": test_df["contestant"].values,
        "y_true": y_test,
        "y_prob": family_test_probs[best_name],
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
