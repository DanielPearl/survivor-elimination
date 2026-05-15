"""Buy-gate validators for the Survivor watchlist.

Each validator returns ``(passes, reason)``. The watchlist exporter
chains them: if all pass and EV > min_ev_per_contract, the row is
marked BUY (with side derived from sign of model − market). Anything
failing surfaces as SKIP or WATCH (price + EV present but a structural
gate blocked the trade — e.g. stale market, thin book).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

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
        # If Kalshi never reported last_updated we don't fail the gate
        # — we just emit a WATCH downstream. Returns True here.
        return True, ""
    age = (datetime.now(timezone.utc) - lu).total_seconds()
    if age > stale_seconds:
        return False, f"market quote {int(age)}s old > {stale_seconds}s"
    return True, ""


def has_model_output(row: Dict[str, Any]) -> Tuple[bool, str]:
    if row.get("model_prob_eliminated") is None:
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
    to do with the row based on whether `passes` is True.

    Duplicate-contestant detection runs here: if the same contestant
    appears in `seen_contestants` already, we mark the row blocked so
    the dashboard surfaces it as a data-quality issue rather than
    silently double-scoring the same boot.
    """
    cfg = load_config()
    val = cfg.get("validators", {})
    lo, hi = val.get("prob_bounds_cents", [3, 97])
    blockers: List[str] = []
    for fn in (market_open, parses_season_episode, parses_contestant,
               has_model_output):
        ok, reason = fn(row)
        if not ok:
            blockers.append(reason)
    ok, reason = has_valid_price(row, int(lo), int(hi))
    if not ok:
        blockers.append(reason)
    ok, reason = spread_ok(row, int(val.get("max_spread_cents", 12)))
    if not ok:
        blockers.append(reason)
    ok, reason = closes_in_window(row,
                                   int(val.get("min_minutes_to_close", 30)),
                                   int(val.get("max_minutes_to_close", 20160)))
    if not ok:
        blockers.append(reason)
    ok, reason = liquidity_ok(row,
                               int(val.get("min_volume", 0)),
                               int(val.get("min_open_interest", 0)))
    if not ok:
        blockers.append(reason)
    ok, reason = market_fresh(row, int(val.get("stale_market_seconds", 1800)))
    if not ok:
        blockers.append(reason)
    name = row.get("contestant")
    if name and seen_contestants.count(name) > 1:
        blockers.append("duplicate contestant row")
    return (not blockers), blockers
