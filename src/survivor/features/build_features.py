"""Feature engineering for the per-contestant-per-episode panel.

Takes the historical_boots.csv schema (or the live state record) and
returns the matrix of features the model trains/predicts on. Keeping
the transform here (rather than inline in the trainer) means the live
scorer uses the exact same featurisation — no train/serve skew.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# Order matters: the trainer dumps coefficients in this order, the
# live scorer rebuilds the feature matrix in this order. Any addition
# bumps the model artifact version (handled in the trainer).
FEATURE_COLUMNS: List[str] = [
    # ── Structural / game-state ───────────────────────────────────
    "season",
    "episode",
    "remaining",
    "tribe_size",
    "starting_tribe_size",
    "merged",
    "swap_phase",
    "episode_share_remaining",      # episode / (episode + remaining)
    "pre_merge_phase",                # 1 if merged == 0
    "is_finale",                       # 1 if remaining <= 4

    # ── Per-contestant on-show signal ─────────────────────────────
    "immunity_won",
    "tribe_immunity",
    "has_idol",
    "in_main_alliance",
    "prior_votes_against",
    "times_targeted",
    "confessional_count",
    "visibility_score",
    "visibility_spike",
    "negative_edit_score",
    "strategic_isolation",
    "prior_perf_score",

    # ── Edgic / screen-time extensions ────────────────────────────
    # confessional_share = contestant's share of total cast
    # confessionals that episode. Captures relative-not-absolute
    # screen time — a 6-confessional ep where everyone else got 8
    # is a very different signal from a 6-confessional ep where
    # everyone else got 2.
    "confessional_share",
    "narrative_intensity",
    "swing_vote_potential",

    # ── Game-state advantages ────────────────────────────────────
    "advantages_held",
    "idols_played_this_ep",
    "vote_steals_active",
    "same_starting_tribe_remaining",
    "voting_minority_score",

    # ── Returnee dynamics ─────────────────────────────────────────
    "is_returnee",
    "season_returnee_count",
    "is_returnee_first_three_eps",

    # ── Reddit-derived (zero for historical rows, populated live) ─
    "reddit_mention_count",
    "reddit_boot_pick_count",
    "reddit_sentiment",
    "reddit_visibility_score",
    "reddit_target_share",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model's input matrix.

    Accepts either the training panel (must include all
    HISTORICAL_COLUMNS) or the live-state DataFrame produced by the
    Kalshi market scorer. Any missing Reddit column is filled with
    zero so the trained model is happy at inference time.
    """
    out = df.copy()
    # Reddit-derived columns are optional — fill missing with 0.
    for col in ("reddit_mention_count", "reddit_boot_pick_count",
                 "reddit_sentiment", "reddit_visibility_score",
                 "reddit_target_share"):
        if col not in out.columns:
            out[col] = 0.0

    # Derived structural columns.
    out["episode_share_remaining"] = (
        out["episode"].astype(float)
        / (out["episode"].astype(float) + out["remaining"].astype(float))
    )
    out["pre_merge_phase"] = (1 - out["merged"].astype(int)).astype(int)
    out["is_finale"] = (out["remaining"].astype(int) <= 4).astype(int)

    # Guard: numeric coercion + NaN -> 0 for the few derived ratios.
    for col in FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0
    return out[FEATURE_COLUMNS].astype(float)


def label_column(df: pd.DataFrame) -> np.ndarray:
    return df["eliminated"].astype(int).values
