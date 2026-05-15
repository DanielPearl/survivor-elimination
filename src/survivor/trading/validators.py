"""Buy-gate validators for the Survivor watchlist.

Each validator returns ``(passes, reason)``. The watchlist exporter
chains them: if all pass and EV > min_ev_per_contract, the row is
marked BUY (with side derived from sign of model − market). Anything
failing surfaces as SKIP or WATCH.

Threshold defaults come from ``kalshi_sdk.validators.UNIFIED_VALIDATOR_DEFAULTS``
so the survivor bot matches every other Kalshi bot's cautious floor
(prob bounds 25-75c, max-entry 70c, spread cap 6c, min volume/OI 50).
Survivor-specific gates (contestant parsing, season/episode parsing,
duplicate detection, model-output presence) remain local because they
have no analog in the other bots.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from kalshi_sdk.validators import UNIFIED_VALIDATOR_DEFAULTS

from ..utils.config import load_config


def _to_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def market_open(row: Dict[str, Any]) -> Tuple[bool, str]:
    status = (row.get("status") or "").lower()
    if status in ("closed", "settled", "finalized", "cancelled"):
        return False, f"market not open (status={status})"
    return True, ""


def has_valid_price(row: Dict[str, Any], lo: int, hi: int
                     ) -> Tuple[bool, str]:
    ya = row.get("yes_ask_cents")
    if ya is None:
        return False, "no Kalshi quote"
    if ya < lo or ya > hi:
        return False, f"price {ya}c outside [{lo}, {hi}]"
    return True, ""


def max_entry_price_ok(row: Dict[str, Any], cap_cents: int) -> Tuple[bool, str]:
    """Hard cap on the price we'd pay. Same gate every other Kalshi bot
    runs — at >70c the loss-vs-gain ratio is 2.3:1+ and a single
    mispredicted elimination eats many wins."""
    ya = row.get("yes_ask_cents")
    if ya is None:
        return True, ""
    if ya > cap_cents:
        return False, f"entry too expensive ({ya}c > {cap_cents}c cap)"
    return True, ""


def parses_season_episode(row: Dict[str, Any]) -> Tuple[bool, str]:
    if row.get("season") is None:
        return False, "could not parse season"
    if row.get("episode") is None:
        return False, "could not parse episode"
    return True, ""


def parses_contestant(row: Dict[str, Any]) -> Tuple[bool, str]:
    if not row.get("contestant"):
        return False, "could not parse contestant"
    return True, ""


def market_fresh(row: Dict[str, Any], stale_seconds: int
                  ) -> Tuple[bool, str]:
    lu = _to_dt(row.get("last_updated"))
    if lu is None:
        return True, ""
    age = (datetime.now(timezone.utc) - lu).total_seconds()
    if age > stale_seconds:
        return False, f"market quote {int(age)}s old > {stale_seconds}s"
    return True, ""


def has_model_output(row: Dict[str, Any]) -> Tuple[bool, str]:
    if row.get("model_prob") is None and row.get("model_prob_eliminated") is None \
            and row.get("model_prob_win_season") is None:
        return False, "no model output for contestant"
    return True, ""


def spread_ok(row: Dict[str, Any], max_spread: int) -> Tuple[bool, str]:
    sp = row.get("spread_cents")
    if sp is None:
        return True, ""
    if sp > max_spread:
        return False, f"spread {sp:.0f}c > {max_spread}c"
    return True, ""


def closes_in_window(row: Dict[str, Any], min_min: int, max_min: int
                      ) -> Tuple[bool, str]:
    exp = _to_dt(row.get("expected_expiration_time"))
    if exp is None:
        return True, ""
    mtc = (exp - datetime.now(timezone.utc)).total_seconds() / 60.0
    if mtc < min_min:
        return False, f"closes in {mtc:.0f}m < {min_min}m"
    if mtc > max_min:
        return False, f"closes in {mtc:.0f}m > {max_min}m"
    return True, ""


def liquidity_ok(row: Dict[str, Any], min_vol: int, min_oi: int
                  ) -> Tuple[bool, str]:
    v = row.get("volume") or 0
    oi = row.get("open_interest") or 0
    if v < min_vol:
        return False, f"volume {int(v)} < {min_vol}"
    if oi < min_oi:
        return False, f"open interest {int(oi)} < {min_oi}"
    return True, ""


def evaluate_row(row: Dict[str, Any], seen_contestants: List[str]
                  ) -> Tuple[bool, List[str]]:
    """Run every gate, collect the blocker list. Caller decides what
    to do with the row based on whether ``passes`` is True.

    Validator thresholds fall back to ``UNIFIED_VALIDATOR_DEFAULTS`` so
    a missing config.yaml field doesn't accidentally widen a gate; the
    cautious-side floor is the same one every other Kalshi bot runs.
    """
    cfg = load_config()
    val = cfg.get("validators", {})
    lo, hi = val.get("prob_bounds_cents",
                     UNIFIED_VALIDATOR_DEFAULTS["prob_bounds_cents"])
    blockers: List[str] = []
    for fn in (market_open, parses_season_episode, parses_contestant,
               has_model_output):
        ok, reason = fn(row)
        if not ok:
            blockers.append(reason)
    ok, reason = has_valid_price(row, int(lo), int(hi))
    if not ok:
        blockers.append(reason)
    ok, reason = max_entry_price_ok(
        row,
        int(val.get("max_entry_price_cents",
                    UNIFIED_VALIDATOR_DEFAULTS["max_entry_price_cents"])),
    )
    if not ok:
        blockers.append(reason)
    ok, reason = spread_ok(
        row,
        int(val.get("max_spread_cents",
                    UNIFIED_VALIDATOR_DEFAULTS["max_spread_cents"])),
    )
    if not ok:
        blockers.append(reason)
    ok, reason = closes_in_window(
        row,
        int(val.get("min_minutes_to_close",
                    UNIFIED_VALIDATOR_DEFAULTS["min_minutes_to_close"])),
        int(val.get("max_minutes_to_close",
                    UNIFIED_VALIDATOR_DEFAULTS["max_minutes_to_close"])),
    )
    if not ok:
        blockers.append(reason)
    ok, reason = liquidity_ok(
        row,
        int(val.get("min_volume",
                    UNIFIED_VALIDATOR_DEFAULTS["min_volume"])),
        int(val.get("min_open_interest",
                    UNIFIED_VALIDATOR_DEFAULTS["min_open_interest"])),
    )
    if not ok:
        blockers.append(reason)
    ok, reason = market_fresh(row, int(val.get("stale_market_seconds", 1800)))
    if not ok:
        blockers.append(reason)
    name = row.get("contestant")
    if name and seen_contestants.count(name) > 1:
        blockers.append("duplicate contestant row")
    return (not blockers), blockers
