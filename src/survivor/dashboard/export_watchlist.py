"""Survivor watchlist exporter.

End-to-end pipeline run by the live scorer every tick:

  1. Fetch Kalshi survivor markets (kalshi.markets.fetch_survivor_markets)
  2. Normalise (one row per contestant per market).
  3. Build the current_state DataFrame and join Reddit-derived
     signals onto it (reddit.ingest.safe_pull).
  4. Score every row with the trained model.
  5. Compute edge / EV / verdict / validators.
  6. Write watchlist.json + watchlist.csv.

The dashboard reads watchlist.json directly via the survivor adapter
in trading_dashboard/survivor.py.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..data.current_state import (
    load_current_state, state_to_dataframe, synthesize_state_from_kalshi,
)
from ..kalshi.markets import fetch_survivor_markets, normalise_markets
from ..models.predict import predict_eliminated_proba
from ..reddit.ingest import safe_pull as reddit_safe_pull
from ..trading.ev import ev as ev_calc
from ..trading.validators import evaluate_row
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("survivor.dashboard.export")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _verdict(row: Dict[str, Any],
              buy_eligible: bool, blockers: List[str],
              min_edge: float, min_ev: float) -> str:
    """Standard vocabulary used by the dashboard adapter:
      BUY YES   — model probability >> market, +EV, no blockers
      BUY NO    — model probability << market, +EV on the NO side
      SKIP      — EV on both sides ≤ 0, or no quote
      WATCH     — positive EV but a validator gate blocked the trade
    """
    ev_y = row.get("ev_yes")
    ev_n = row.get("ev_no")
    edge = row.get("edge")
    if buy_eligible and edge is not None:
        if edge >= min_edge and ev_y is not None and ev_y >= min_ev:
            return "BUY YES"
        if -edge >= min_edge and ev_n is not None and ev_n >= min_ev:
            return "BUY NO"
    # Quoted but failing a gate, with positive EV anywhere -> WATCH.
    if (ev_y is not None and ev_y > 0) or (ev_n is not None and ev_n > 0):
        if blockers:
            return "WATCH"
    return "SKIP"


def _confidence_score(edge: float | None, market_prob: float | None) -> float:
    """A 0..1 confidence the model has on this row.

    Higher when:
      - |edge| is large
      - market price is away from the noisy 50% line
    Used as a single sortable column the user can scan to triage rows.
    """
    if edge is None or market_prob is None:
        return 0.0
    base = min(1.0, abs(edge) * 4.0)
    distance_from_coin = abs(market_prob - 0.5) * 2.0
    return round(0.6 * base + 0.4 * distance_from_coin, 3)


def build_watchlist(kalshi_records: List[Dict[str, Any]] | None = None
                     ) -> Dict[str, Any]:
    """Return the full watchlist payload (also written to disk).

    `kalshi_records` defaults to a live fetch — pass a fixture in for
    tests and offline smoke runs.
    """
    cfg = load_config()
    if kalshi_records is None:
        try:
            raw = fetch_survivor_markets()
            kalshi_records = normalise_markets(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("kalshi fetch failed: %s — returning empty watchlist", exc)
            kalshi_records = []

    # Per-episode "Will X be eliminated" markets only. Season-winner
    # markets (the bulk of what Kalshi currently lists under the
    # KXSURVIVOR series) are explicitly excluded — they answer a
    # different question than the model and would dilute the
    # watchlist. When no elimination markets are active, the bot's
    # output is an empty rows list and the dashboard hides the card
    # entirely (see survivor.is_available below).
    kalshi_records = [
        r for r in (kalshi_records or [])
        if (r.get("market_type") or "") == "elimination"
    ]

    state = load_current_state()
    if not state.get("contestants"):
        state = synthesize_state_from_kalshi(kalshi_records)

    # Make sure every contestant Kalshi has a market for is in the
    # state-derived scoring set. If the user hasn't edited
    # current_state.json yet (or this is a brand-new season), the
    # Kalshi-only contestants would otherwise miss the model
    # entirely. They get the same defaults as a synthesized state.
    state_df = state_to_dataframe(state)
    state_names = set(state_df["contestant"].astype(str))
    kalshi_names = []
    seen = set()
    for r in kalshi_records:
        c = r.get("contestant")
        if c and c not in seen:
            seen.add(c)
            kalshi_names.append(c)
    missing = [n for n in kalshi_names if n not in state_names]
    if missing:
        # Mint default-only state rows for each missing contestant
        # and concatenate so the model still scores them.
        stub = {
            **state,
            "contestants": [{"name": n} for n in missing],
            "synthesized": True,
        }
        # Re-use the canonical state-to-frame conversion so the
        # defaults are applied identically.
        from ..data.current_state import state_to_dataframe as _s2df
        more = _s2df(stub)
        if not more.empty:
            import pandas as pd
            state_df = pd.concat([state_df, more], ignore_index=True)

    # ── Reddit features ──────────────────────────────────────────────
    contestant_names = state_df["contestant"].astype(str).tolist()
    reddit_features = reddit_safe_pull(contestant_names) if contestant_names else {}
    for col in ("reddit_mention_count", "reddit_boot_pick_count",
                 "reddit_sentiment", "reddit_visibility_score",
                 "reddit_target_share"):
        state_df[col] = state_df["contestant"].map(
            lambda c, k=col: reddit_features.get(c, {}).get(k, 0.0)
        )

    # ── Model scoring ────────────────────────────────────────────────
    if len(state_df) > 0:
        probs = predict_eliminated_proba(state_df)
    else:
        probs = []
    state_df["model_prob_eliminated"] = probs

    # ── Join with Kalshi markets ─────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    slip = float(cfg["kalshi"].get("slippage_pct", 0.02))
    min_edge = float(cfg["trading"].get("min_edge", 0.06))
    min_ev = float(cfg["trading"].get("min_ev_per_contract", 0.02))
    seen_names: List[str] = [r.get("contestant") for r in kalshi_records]

    # Map contestant -> model probability for quick lookup.
    model_by_contestant = dict(zip(state_df["contestant"].astype(str),
                                     state_df["model_prob_eliminated"].astype(float)))
    # Derive each contestant's model-based P(wins season) for season-
    # winner markets. Approximation: each remaining episode the
    # contestant has the same boot probability (model_p_elim). After
    # `r-1` more boots they're a finalist; from there assume the win
    # is uniform among finalists (≈ 1/4 with the modern fire-making
    # final four). So:
    #     P(survives to finale) ≈ (1 - p_elim) ^ (active - 4)
    #     P(wins | survives)     ≈ 0.25
    # Clamp to [0.005, 0.4].
    active = max(1, len(state_df))
    win_season_by_contestant: Dict[str, float] = {}
    for c, p_elim in model_by_contestant.items():
        steps_to_finale = max(0, active - 4)
        p_survive = max(0.0, min(1.0, 1.0 - float(p_elim))) ** steps_to_finale
        p_win = p_survive * 0.25
        win_season_by_contestant[c] = max(0.005, min(0.4, p_win))

    for k in kalshi_records:
        name = k.get("contestant")
        market_type = k.get("market_type") or "unknown"
        mp_elim = (float(k["market_prob_eliminated"])
                    if k.get("market_prob_eliminated") is not None else None)
        mp_win = (float(k["market_prob_win_season"])
                   if k.get("market_prob_win_season") is not None else None)
        model_p_elim = (float(model_by_contestant.get(name))
                         if name in model_by_contestant else None)
        model_p_win = (float(win_season_by_contestant.get(name))
                        if name in win_season_by_contestant else None)

        # Decide which model probability we're comparing to which
        # Kalshi probability for *this* market.
        if market_type == "season_win":
            model_p = model_p_win
            mp = mp_win
        else:
            model_p = model_p_elim
            mp = mp_elim
        # Edge = model - market on the YES side of THIS market.
        edge = None
        ev_yes = ev_no = None
        if model_p is not None and mp is not None:
            edge = model_p - mp
            ev_yes = ev_calc(model_p, mp, slip).ev_per_contract
            ev_no = ev_calc(1.0 - model_p, 1.0 - mp, slip).ev_per_contract

        row: Dict[str, Any] = {
            "match_id": k.get("event_ticker") or k.get("ticker"),
            "ticker": k.get("ticker"),
            "series_ticker": k.get("series_ticker"),
            "season": k.get("season"),
            "episode": k.get("episode"),
            "contestant": name,
            "title": k.get("title"),
            "market_type": market_type,
            "model_prob_eliminated": (round(model_p_elim, 4)
                                       if model_p_elim is not None else None),
            "market_prob_eliminated": (round(mp_elim, 4)
                                        if mp_elim is not None else None),
            "model_prob_win_season": (round(model_p_win, 4)
                                       if model_p_win is not None else None),
            "market_prob_win_season": (round(mp_win, 4)
                                        if mp_win is not None else None),
            "model_prob": (round(model_p, 4) if model_p is not None else None),
            "market_prob": (round(mp, 4) if mp is not None else None),
            "edge": (round(edge, 4) if edge is not None else None),
            "ev_yes": (round(ev_yes, 4) if ev_yes is not None else None),
            "ev_no": (round(ev_no, 4) if ev_no is not None else None),
            "confidence_score": _confidence_score(edge, mp),
            "yes_ask_cents": k.get("yes_ask_cents"),
            "no_ask_cents": k.get("no_ask_cents"),
            "spread_cents": k.get("spread_cents"),
            "volume": k.get("volume"),
            "open_interest": k.get("open_interest"),
            "status": k.get("status"),
            "last_updated": k.get("last_updated"),
            "expected_expiration_time": k.get("expected_expiration_time"),
            "rules_primary": k.get("rules_primary"),
            "reddit_mention_count": int(reddit_features.get(name, {}).get(
                "reddit_mention_count", 0)) if name else 0,
            "reddit_boot_pick_count": int(reddit_features.get(name, {}).get(
                "reddit_boot_pick_count", 0)) if name else 0,
            "reddit_sentiment": round(float(reddit_features.get(name, {}).get(
                "reddit_sentiment", 0.0)), 3) if name else 0.0,
            "reddit_target_share": round(float(reddit_features.get(name, {}).get(
                "reddit_target_share", 0.0)), 3) if name else 0.0,
        }
        passes, blockers = evaluate_row(
            {**row, "model_prob_eliminated": model_p},
            seen_contestants=seen_names,
        )
        row["buy_eligible"] = bool(passes and (
            (edge is not None and edge >= min_edge and (ev_yes or 0) >= min_ev)
            or (edge is not None and -edge >= min_edge and (ev_no or 0) >= min_ev)
        ))
        row["buy_blockers"] = blockers
        if row["buy_eligible"]:
            row["buy_side"] = "YES" if (edge or 0) >= 0 else "NO"
        else:
            row["buy_side"] = None
        row["verdict"] = _verdict(row, row["buy_eligible"], blockers,
                                    min_edge, min_ev)
        # "gap" = unsigned edge in percentage points, shown in the table.
        row["gap_pp"] = (round(abs(edge) * 100, 1)
                          if edge is not None else None)
        rows.append(row)

    rows.sort(key=lambda r: (
        0 if r.get("buy_eligible") else 1,
        -(abs(r.get("edge") or 0)),
    ))

    payload = {
        "generated_at": _now_iso(),
        "season": state.get("season"),
        "current_episode": state.get("current_episode"),
        "merge_episode": state.get("merge_episode"),
        "active_contestants": len([c for c in state.get("contestants", [])]),
        "rows": rows,
        "synthesized_state": bool(state.get("synthesized")),
    }
    return payload


def export(kalshi_records: List[Dict[str, Any]] | None = None
            ) -> tuple[Path, Path]:
    """Write watchlist.json + watchlist.csv. Returns the two paths."""
    cfg = load_config()
    payload = build_watchlist(kalshi_records)

    json_path = resolve_path(cfg["paths"]["watchlist_json"])
    csv_path = resolve_path(cfg["paths"]["watchlist_csv"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    if payload["rows"]:
        cols = list(payload["rows"][0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in payload["rows"]:
                w.writerow({k: r.get(k) for k in cols})
    log.info("wrote watchlist: %s (%d rows)", json_path, len(payload["rows"]))
    return csv_path, json_path
