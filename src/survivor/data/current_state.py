"""Current-season state file.

The live scorer needs *some* idea of where each Kalshi-listed
contestant sits in the game: tribe, has-idol, votes-against,
visibility, etc. There's no API for that — the show is the source.

We persist the editable state in ``data/raw/current_state.json``. The
file ships in the repo, gets updated by hand as each new episode airs
(usually a couple of dozen lines per week), and the live scorer reads
it on every tick. The shape mirrors the historical-panel schema so
the same feature builder works on either side.

Schema
------

{
  "season": 50,
  "current_episode": 7,
  "merge_episode": 6,
  "swap_episode": 4,
  "as_of": "2026-05-14T20:00:00Z",
  "contestants": [
    {
      "name": "Sam",
      "tribe": "Merge",
      "starting_tribe": "Yanu",
      "starting_tribe_size": 6,
      "tribe_size": 11,
      "merged": 1,
      "swap_phase": 0,
      "immunity_won": 0,
      "tribe_immunity": 0,
      "prior_votes_against": 1,
      "times_targeted": 2,
      "has_idol": 0,
      "in_main_alliance": 1,
      "confessional_count": 6,
      "visibility_score": 0.42,
      "visibility_spike": 0.08,
      "negative_edit_score": 0.28,
      "strategic_isolation": 0.30,
      "prior_perf_score": 0.55
    },
    …
  ]
}

If the file is missing, the loader returns an empty stub — the scorer
then falls back to a "blank prior" (every Kalshi-listed contestant
gets defaults derived purely from remaining count + episode number).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.config import load_config, resolve_path

log = logging.getLogger("survivor.data.current_state")


# Defaults for every numeric column on the contestant record. When a
# user edits the JSON they only need to fill the columns they know;
# missing columns get the defaults below so the feature builder is
# never handed a NaN.
_CONTESTANT_DEFAULTS: Dict[str, Any] = {
    "tribe": "Merge",
    "starting_tribe": "Unknown",
    "starting_tribe_size": 6,
    "tribe_size": 0,           # filled from remaining count if 0
    "merged": 1,
    "swap_phase": 0,
    "immunity_won": 0,
    "tribe_immunity": 0,
    "prior_votes_against": 0,
    "times_targeted": 0,
    "has_idol": 0,
    "in_main_alliance": 0,
    "confessional_count": 5,
    "visibility_score": 0.40,
    "visibility_spike": 0.0,
    "negative_edit_score": 0.25,
    "strategic_isolation": 0.40,
    "prior_perf_score": 0.50,
}


def load_current_state() -> Dict[str, Any]:
    """Return the current-season state, or an empty stub when absent."""
    cfg = load_config()
    p = resolve_path(cfg["paths"]["current_state_json"])
    if not p.exists():
        return {
            "season": None,
            "current_episode": None,
            "merge_episode": None,
            "swap_episode": 4,
            "as_of": None,
            "contestants": [],
        }
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f) or {}
    data.setdefault("contestants", [])
    return data


def write_current_state(state: Dict[str, Any]) -> Path:
    cfg = load_config()
    p = resolve_path(cfg["paths"]["current_state_json"])
    p.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["as_of"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return p


def state_to_dataframe(state: Dict[str, Any]) -> pd.DataFrame:
    """Project the state document into a feature-builder-ready frame.

    Each contestant entry produces exactly one row with the current
    episode number / season / remaining-count fields populated from
    the state header. Missing per-contestant columns get the defaults
    from ``_CONTESTANT_DEFAULTS``.
    """
    contestants = state.get("contestants") or []
    if not contestants:
        return pd.DataFrame(columns=[
            "season", "episode", "contestant", "tribe", "eliminated",
            "starting_tribe", "starting_tribe_size", "tribe_size",
            "remaining", "merged", "swap_phase", "immunity_won",
            "tribe_immunity", "prior_votes_against", "times_targeted",
            "has_idol", "in_main_alliance", "confessional_count",
            "visibility_score", "visibility_spike", "negative_edit_score",
            "strategic_isolation", "prior_perf_score",
        ])
    season = int(state.get("season") or 0)
    episode = int(state.get("current_episode") or 1)
    remaining = len(contestants)
    rows: List[Dict[str, Any]] = []
    for c in contestants:
        c = dict(c)
        for k, v in _CONTESTANT_DEFAULTS.items():
            c.setdefault(k, v)
        if not c.get("tribe_size"):
            c["tribe_size"] = remaining
        rows.append({
            "season": season,
            "episode": episode,
            "contestant": c.get("name") or c.get("contestant"),
            "tribe": c["tribe"],
            "eliminated": 0,
            "starting_tribe": c["starting_tribe"],
            "starting_tribe_size": int(c["starting_tribe_size"]),
            "tribe_size": int(c["tribe_size"]),
            "remaining": remaining,
            "merged": int(c["merged"]),
            "swap_phase": int(c["swap_phase"]),
            "immunity_won": int(c["immunity_won"]),
            "tribe_immunity": int(c["tribe_immunity"]),
            "prior_votes_against": int(c["prior_votes_against"]),
            "times_targeted": int(c["times_targeted"]),
            "has_idol": int(c["has_idol"]),
            "in_main_alliance": int(c["in_main_alliance"]),
            "confessional_count": int(c["confessional_count"]),
            "visibility_score": float(c["visibility_score"]),
            "visibility_spike": float(c["visibility_spike"]),
            "negative_edit_score": float(c["negative_edit_score"]),
            "strategic_isolation": float(c["strategic_isolation"]),
            "prior_perf_score": float(c["prior_perf_score"]),
        })
    return pd.DataFrame(rows)


def synthesize_state_from_kalshi(kalshi_records: List[Dict[str, Any]]
                                   ) -> Dict[str, Any]:
    """Build a stub current_state from Kalshi market rows alone.

    Used as a fallback when no hand-edited state file is present.
    Every contestant gets the defaults; season/episode pulled from
    the first record. The model is still scored, but the only signal
    available is the market price + (when reachable) Reddit features.
    """
    if not kalshi_records:
        return {"season": None, "current_episode": None, "contestants": []}
    season = next((r["season"] for r in kalshi_records if r.get("season")), None)
    episode = next((r["episode"] for r in kalshi_records if r.get("episode")), None)
    names = []
    seen = set()
    for r in kalshi_records:
        c = r.get("contestant")
        if c and c not in seen:
            seen.add(c)
            names.append(c)
    return {
        "season": season,
        "current_episode": episode,
        "merge_episode": None,
        "swap_episode": 4,
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthesized": True,
        "contestants": [{"name": n, **_CONTESTANT_DEFAULTS} for n in names],
    }
