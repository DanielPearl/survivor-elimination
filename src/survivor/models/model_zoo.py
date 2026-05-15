"""Model zoo + season-aware CV for the elimination model.

Defines a collection of candidate classifiers with small
hyperparameter grids, plus a season-aware cross-validation routine
that scores each (model, params) combination by F1 on per-episode-
normalised probabilities — the same evaluation surface the trainer
applies on the held-out test seasons.

Models in the zoo (out-of-the-box configs, lightly tuned):

  logistic         L2-regularised logistic regression
  random_forest    sklearn RandomForestClassifier
  hgb              sklearn HistGradientBoostingClassifier
  xgboost          XGBClassifier (skipped if xgboost not installed)
  lightgbm         LGBMClassifier (skipped if lightgbm not installed)
  naive_bayes      GaussianNB baseline
  linear_svm       calibrated LinearSVC

Stacker — sklearn StackingClassifier with a logistic meta-learner
on the top-3 base models (ranked by CV F1).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (HistGradientBoostingClassifier,
                                RandomForestClassifier,
                                StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

log = logging.getLogger("survivor.models.zoo")


# --------------------------------------------------------------------------- #
# Optional deps                                                                #
# --------------------------------------------------------------------------- #

def _try_xgb():
    # Catch any exception, not just ImportError — xgboost raises
    # XGBoostError when libomp can't be loaded (common on macOS dev
    # machines that haven't run ``brew install libomp``). The droplet
    # has libgomp; xgboost loads cleanly there.
    try:
        from xgboost import XGBClassifier  # noqa: F401
        return XGBClassifier
    except Exception:  # noqa: BLE001
        return None


def _try_lgb():
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
        return LGBMClassifier
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# ModelSpec + factories                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class ModelSpec:
    name: str
    factory: Callable[[Dict[str, Any]], BaseEstimator]
    needs_scaling: bool
    param_grid: List[Dict[str, Any]]


def _logistic_factory(params):
    p = {"penalty": "l2", "solver": "liblinear"}
    p.update(params)
    return LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=17, **p,
    )


def _rf_factory(params):
    return RandomForestClassifier(
        class_weight="balanced", random_state=17, n_jobs=-1, **params,
    )


def _hgb_factory(params):
    return HistGradientBoostingClassifier(
        class_weight="balanced", random_state=17, **params,
    )


def _nb_factory(_params):
    return GaussianNB()


def _linear_svm_factory(params):
    base = LinearSVC(class_weight="balanced", random_state=17,
                     max_iter=4000, dual=False, **params)
    return CalibratedClassifierCV(base, cv=3, method="sigmoid")


def _xgb_factory(params):
    XGB = _try_xgb()
    if XGB is None:
        return None
    return XGB(
        random_state=17, n_jobs=-1, eval_metric="logloss",
        scale_pos_weight=10.0, verbosity=0, **params,
    )


def _lgb_factory(params):
    LGB = _try_lgb()
    if LGB is None:
        return None
    return LGB(
        random_state=17, n_jobs=-1, class_weight="balanced",
        verbosity=-1, **params,
    )


def build_zoo() -> List[ModelSpec]:
    """Return the full model zoo. xgboost / lightgbm specs are omitted
    when those packages aren't installed."""
    zoo: List[ModelSpec] = [
        ModelSpec("logistic", _logistic_factory, True, [
            {"C": 0.1, "penalty": "l2"},
            {"C": 1.0, "penalty": "l2"},
            {"C": 10.0, "penalty": "l2"},
            {"C": 1.0, "penalty": "l1"},
        ]),
        ModelSpec("random_forest", _rf_factory, False, [
            {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 5},
            {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 5},
            {"n_estimators": 400, "max_depth": 5, "min_samples_leaf": 5},
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 20},
        ]),
        ModelSpec("hgb", _hgb_factory, False, [
            {"learning_rate": 0.03, "max_depth": 3, "max_iter": 300,
             "l2_regularization": 1.0},
            {"learning_rate": 0.05, "max_depth": 4, "max_iter": 200,
             "l2_regularization": 1.0},
            {"learning_rate": 0.05, "max_depth": 5, "max_iter": 200,
             "l2_regularization": 0.5},
            {"learning_rate": 0.10, "max_depth": 3, "max_iter": 200,
             "l2_regularization": 1.0},
        ]),
        ModelSpec("naive_bayes", _nb_factory, True, [{}]),
        ModelSpec("linear_svm", _linear_svm_factory, True, [
            {"C": 0.1},
            {"C": 1.0},
        ]),
    ]
    if _try_xgb() is not None:
        zoo.append(ModelSpec("xgboost", _xgb_factory, False, [
            {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 200,
             "subsample": 0.7},
            {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 200,
             "subsample": 0.7},
            {"learning_rate": 0.10, "max_depth": 3, "n_estimators": 200,
             "subsample": 1.0},
            {"learning_rate": 0.10, "max_depth": 5, "n_estimators": 400,
             "subsample": 0.7},
        ]))
    if _try_lgb() is not None:
        zoo.append(ModelSpec("lightgbm", _lgb_factory, False, [
            {"learning_rate": 0.05, "num_leaves": 15, "n_estimators": 200},
            {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 200},
            {"learning_rate": 0.10, "num_leaves": 15, "n_estimators": 200},
            {"learning_rate": 0.10, "num_leaves": 31, "n_estimators": 400},
        ]))
    return zoo


# --------------------------------------------------------------------------- #
# Per-episode normalisation (array form — same idiom as train.py)             #
# --------------------------------------------------------------------------- #

def normalize_per_episode_arr(season: np.ndarray, episode: np.ndarray,
                                raw: np.ndarray) -> np.ndarray:
    """Sum-to-1 normalisation within (season, episode) groups.
    Mirrors ``train.normalize_per_episode`` but takes array inputs so
    CV folds don't need to round-trip through a DataFrame.
    """
    season = np.asarray(season)
    episode = np.asarray(episode)
    raw = np.asarray(raw, dtype=float)
    out = raw.copy()
    keys = list(zip(season.tolist(), episode.tolist()))
    sums: Dict[Tuple[int, int], float] = {}
    for k, p in zip(keys, raw):
        sums[k] = sums.get(k, 0.0) + float(p)
    for i, k in enumerate(keys):
        s = sums.get(k, 0.0)
        if s > 1e-9:
            out[i] = raw[i] / s
    return out


def _optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                            min_recall: float = 0.30) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        if recall_score(y_true, y_pred, zero_division=0) < min_recall:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    if best_f1 < 0:
        for t in np.linspace(0.05, 0.95, 91):
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
    return best_t


# --------------------------------------------------------------------------- #
# Probability accessor                                                        #
# --------------------------------------------------------------------------- #

def _predict_proba(model, X) -> np.ndarray:
    """Return positive-class probabilities for any sklearn-compatible
    classifier. Falls back to a sigmoid over ``decision_function`` when
    the estimator doesn't expose ``predict_proba`` directly."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))
    raise AttributeError(f"{type(model).__name__} has no proba interface")


# --------------------------------------------------------------------------- #
# Season-aware CV evaluator                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class CVScore:
    spec_name: str
    params: Dict[str, Any]
    mean_f1: float
    std_f1: float
    mean_precision: float
    mean_recall: float
    fold_f1s: List[float]


def evaluate_spec(spec: ModelSpec, params: Dict[str, Any],
                    X: pd.DataFrame, y: np.ndarray,
                    train_df: pd.DataFrame,
                    n_splits: int = 5) -> CVScore | None:
    """Season-aware K-fold CV for one (model, params) combination.

    Each fold leaves one training-season out for validation. F1 is
    computed on per-episode-normalised probabilities at a per-fold
    F1-optimal threshold (≥ 30% recall floor) — identical scoring
    surface to the final test-set evaluation in train.py.
    """
    groups = train_df["season"].values
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    kf = GroupKFold(n_splits=splits)
    fold_f1s: List[float] = []
    fold_recalls: List[float] = []
    fold_precisions: List[float] = []
    for tr_idx, va_idx in kf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        df_va = train_df.iloc[va_idx]
        if spec.needs_scaling:
            scaler = StandardScaler()
            X_tr_arr = scaler.fit_transform(X_tr)
            X_va_arr = scaler.transform(X_va)
        else:
            X_tr_arr = X_tr.values
            X_va_arr = X_va.values
        model = spec.factory(params)
        if model is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_arr, y_tr)
            p_va_raw = _predict_proba(model, X_va_arr)
        p_va = normalize_per_episode_arr(
            df_va["season"].values, df_va["episode"].values, p_va_raw,
        )
        thr = _optimal_f1_threshold(y_va, p_va)
        y_pred = (p_va >= thr).astype(int)
        fold_f1s.append(f1_score(y_va, y_pred, zero_division=0))
        fold_recalls.append(recall_score(y_va, y_pred, zero_division=0))
        fold_precisions.append(precision_score(y_va, y_pred, zero_division=0))
    return CVScore(
        spec_name=spec.name, params=params,
        mean_f1=float(np.mean(fold_f1s)),
        std_f1=float(np.std(fold_f1s)),
        mean_precision=float(np.mean(fold_precisions)),
        mean_recall=float(np.mean(fold_recalls)),
        fold_f1s=fold_f1s,
    )


def run_sweep(X_train: pd.DataFrame, y_train: np.ndarray,
                train_df: pd.DataFrame,
                n_splits: int = 5
                ) -> Tuple[List[CVScore], Dict[str, CVScore]]:
    """Evaluate every (spec, params) in the zoo via season-aware CV.

    Returns ``(all_scores, best_per_spec)``: every CV result and the
    single best-by-mean-F1 config per model family.
    """
    zoo = build_zoo()
    all_scores: List[CVScore] = []
    best_per_spec: Dict[str, CVScore] = {}
    for spec in zoo:
        for params in spec.param_grid:
            sc = evaluate_spec(spec, params, X_train, y_train, train_df,
                                 n_splits=n_splits)
            if sc is None:
                continue
            all_scores.append(sc)
            cur = best_per_spec.get(spec.name)
            if cur is None or sc.mean_f1 > cur.mean_f1:
                best_per_spec[spec.name] = sc
            log.info("CV %s %s -> F1=%.3f±%.3f P=%.3f R=%.3f",
                      spec.name, params, sc.mean_f1, sc.std_f1,
                      sc.mean_precision, sc.mean_recall)
    return all_scores, best_per_spec


# --------------------------------------------------------------------------- #
# Stacker                                                                     #
# --------------------------------------------------------------------------- #

def _wrap_for_stack(spec: ModelSpec, params: Dict[str, Any]
                     ) -> BaseEstimator:
    """Return an estimator pipelined with a StandardScaler if the spec
    needs scaling. StackingClassifier uses cross_val_predict to feed
    the meta-learner, which needs each base learner to be self-
    contained re: preprocessing."""
    base = spec.factory(params)
    if spec.needs_scaling:
        base = Pipeline([("scaler", StandardScaler()), ("clf", base)])
    return base


def build_stacker(best_specs: List[Tuple[ModelSpec, Dict[str, Any]]],
                    X_train: pd.DataFrame, y_train: np.ndarray,
                    train_df: pd.DataFrame) -> StackingClassifier:
    """Stack the supplied base models with a logistic meta-learner.

    Uses GroupKFold (by season) so the meta-learner sees out-of-fold
    predictions that don't leak within-season patterns. Pre-computed
    splits are passed via the ``cv`` parameter.
    """
    estimators = [(spec.name, _wrap_for_stack(spec, params))
                   for spec, params in best_specs]
    n_groups = len(np.unique(train_df["season"]))
    splits = list(GroupKFold(n_splits=min(5, n_groups)).split(
        X_train, y_train, train_df["season"].values,
    ))
    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000,
                                              random_state=17),
        cv=splits,
        passthrough=False,
        n_jobs=1,
    )
    return stacker
