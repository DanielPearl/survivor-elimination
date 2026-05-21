"""Historical boot-data loader.

The training panel is a CSV at ``data/raw/historical_boots.csv``,
materialised from the ``doehm/survivoR`` archive via
``scripts/build_historical_panel.py``. There is no synthetic fallback —
if the CSV is missing the trainer raises and the operator is expected
to regenerate it from the archive.

Each row is one (season, episode, contestant) observation from a past
US Survivor season. Schema:

    season              int      Survivor season number (e.g. 41)
    episode             int      episode number within the season (1-indexed)
    contestant          str      contestant display name
    tribe               str      tribe at the start of the episode (post-swap)
    eliminated          int      1 if voted out / medevac / quit this episode
    starting_tribe      str      original tribe at season start
    starting_tribe_size int      number of contestants on starting tribe at S=1
    tribe_size          int      contestants on the contestant's tribe at start of ep
    remaining           int      contestants left in the game at start of ep
    merged              int      1 if the merge has occurred at start of episode
    swap_phase          int      1 if a tribe swap has occurred but not yet merged
    immunity_won        int      1 if contestant held individual immunity at TC
    tribe_immunity      int      1 if contestant's tribe won tribal immunity (pre-merge)
    prior_votes_against int      cumulative votes against the contestant so far
    times_targeted      int      cumulative episodes where named as target in confessionals
    has_idol            int      1 if known to hold a hidden immunity idol
    in_main_alliance    int      1 if part of the dominant alliance going in
    confessional_count  int      confessionals the contestant had this episode
    visibility_score    float    rolling mean confessionals share (0-1) over last 3 eps
    visibility_spike    float    this episode's confessional share - rolling mean
    negative_edit_score float    edgic-style negative edit signal (0-1)
    strategic_isolation float    0=well-connected, 1=on the bottom (count of fewer allies)
    prior_perf_score    float    rolling mean of (challenge wins / votes received) so far

Sources & curation
------------------
The panel is built from the ``doehm/survivoR`` GitHub archive
(MIT-licensed, actively maintained). Confessional counts, vote history,
advantage movements, and challenge results are all real per-episode
records — no synthetic data. See
``src/survivor/data/survivor_archive.py`` for the construction.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.config import load_config, resolve_path

log = logging.getLogger("survivor.data.historical")


HISTORICAL_COLUMNS = [
    # ── Identity / structure ──────────────────────────────────────
    "season", "episode", "contestant", "tribe", "eliminated",
    "starting_tribe", "starting_tribe_size", "tribe_size", "remaining",
    "merged", "swap_phase",
    # ── On-show signal (pre-tribal) ───────────────────────────────
    "immunity_won", "tribe_immunity",
    "prior_votes_against", "times_targeted",
    "has_idol", "in_main_alliance",
    "confessional_count", "visibility_score", "visibility_spike",
    "negative_edit_score", "strategic_isolation", "prior_perf_score",
    # ── Game-state extensions (added in the screen-time + advantage
    # expansion). Defaults to 0 / False if a CSV row leaves them
    # blank so older panels keep working unchanged.
    "is_returnee",                  # contestant played a previous season
    "season_returnee_count",        # total returnees in the cast (cohort dynamics)
    "advantages_held",              # idol + non-idol advantages (extra vote, steal, …)
    "idols_played_this_ep",         # idol played at TC this ep (known pre-vote)
    "vote_steals_active",           # vote-steal-type advantages outstanding in the game
    "same_starting_tribe_remaining",# how many of contestant's OG tribe are still in
    "voting_minority_score",        # 0..1 — 1 = locked in the minority bloc
    "confessional_share",           # confessionals / total cast confessionals this ep
    "narrative_intensity",          # 0..1 — strength of the contestant's storyline
    "swing_vote_potential",         # 0..1 — likelihood of being a tipping vote
    "is_returnee_first_three_eps",  # returnees often safe in their first few episodes
]


# Optional columns — older CSV checkouts may not have these. The loader
# fills them with the per-column defaults below before training.
_OPTIONAL_DEFAULTS: dict[str, float] = {
    "is_returnee": 0.0,
    "season_returnee_count": 0.0,
    "advantages_held": 0.0,
    "idols_played_this_ep": 0.0,
    "vote_steals_active": 0.0,
    "same_starting_tribe_remaining": 0.0,
    "voting_minority_score": 0.5,
    "confessional_share": 0.0,
    "narrative_intensity": 0.5,
    "swing_vote_potential": 0.0,
    "is_returnee_first_three_eps": 0.0,
}


def load_historical_panel(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Read the historical boot panel into a DataFrame.

    The CSV is built from the ``doehm/survivoR`` archive (see
    ``survivor_archive.build_historical_panel``). If it's missing,
    raise — we no longer fall back to synthetic data.
    """
    cfg = load_config()
    if csv_path is None:
        csv_path = resolve_path(cfg["paths"]["historical_csv"])
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"historical panel not found at {p}. Run "
            "`python scripts/build_historical_panel.py` to materialise it "
            "from the doehm/survivoR archive."
        )
    df = pd.read_csv(p)
    # Required columns — fail loudly if these are missing.
    required = [c for c in HISTORICAL_COLUMNS if c not in _OPTIONAL_DEFAULTS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"historical csv missing columns: {missing}")
    # Optional columns — back-fill from defaults so older panels keep
    # training cleanly when new feature columns are added.
    for col, default in _OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    df = df[df["episode"].astype(int) >= 1].copy()
    df = df.sort_values(["season", "episode", "contestant"]).reset_index(drop=True)
    log.info("loaded historical panel: %d rows, %d seasons, %d boots",
              len(df), df["season"].nunique(),
              int(df["eliminated"].sum()))
    return df


def write_historical_panel(df: pd.DataFrame,
                            csv_path: str | Path | None = None) -> Path:
    cfg = load_config()
    if csv_path is None:
        csv_path = resolve_path(cfg["paths"]["historical_csv"])
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df[HISTORICAL_COLUMNS].to_csv(p, index=False)
    return p
