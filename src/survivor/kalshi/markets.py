"""Kalshi Survivor-elimination market fetcher.

Kalshi runs a per-episode "Who will be eliminated next?" market for
every active Survivor season. Each event spans one episode and carries
one YES contract per active contestant (plus often a "no elimination"
or "quit/medevac" wildcard). The series_ticker follows the pattern
``KXSURVIVOR{season_number}``.

This module:

  - Lists every open event matching the configured prefix
    (default KXSURVIVOR — picks up KXSURVIVOR50, KXSURVIVOR51, …).
  - Pulls all open markets per event and normalises the price /
    spread / volume / OI columns the same way the tennis bot does.
  - Parses the contestant name out of the market title (Kalshi titles
    follow "Will {Contestant} be eliminated …" patterns).
  - Identifies the latest active episode by the event with the
    soonest expected_expiration_time among open events.

The result is one record per (event, contestant) pair the live scorer
later joins to the season state to produce model inputs.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List

from ..utils.config import load_config

log = logging.getLogger("survivor.kalshi.markets")


# Kalshi's Survivor markets come in two shapes:
#   ELIMINATION  "Will Sam be eliminated in episode 7?"
#   SEASON_WIN   "Will Sam Phalen win Survivor Season 50?"
# We recognise each with its own regex and tag the market with a
# ``market_type`` field so the watchlist exporter can apply the right
# transform when computing model_prob vs kalshi_prob.
# Single name token: capitalised word, OR a 1-2 letter token (middle
# initials like "Q" or "J.M."). Joined into a multi-word name via
# the same word-separator pattern that allows up to 4 trailing words.
_NAME_TOK = r"[A-Z][A-Za-z'.\-]*"
_NAME_GROUP = rf"{_NAME_TOK}(?:\s+{_NAME_TOK}){{0,4}}"

_ELIM_PATTERNS = [
    re.compile(rf"^Will\s+(?P<name>{_NAME_GROUP}?)\s+be\s+eliminated", re.IGNORECASE),
    re.compile(rf"^Will\s+(?P<name>{_NAME_GROUP}?)\s+be\s+voted\s+out", re.IGNORECASE),
    re.compile(rf"^Is\s+(?P<name>{_NAME_GROUP}?)\s+the\s+next\s+boot", re.IGNORECASE),
    re.compile(rf"^Will\s+(?P<name>{_NAME_GROUP}?)\s+survive", re.IGNORECASE),
]

_SEASON_WIN_PATTERNS = [
    re.compile(rf"^Will\s+(?P<name>{_NAME_GROUP}?)\s+win\s+Survivor", re.IGNORECASE),
    re.compile(rf"^Will\s+(?P<name>{_NAME_GROUP}?)\s+win\s+the\s+season", re.IGNORECASE),
]

# Fallback — first capitalised name run. Last resort only.
_NAME_FALLBACK = re.compile(rf"\b(?P<name>{_NAME_GROUP})\b")

_EPISODE_RE = re.compile(r"episode\s*(\d+)", re.IGNORECASE)
_SEASON_RE = re.compile(r"season\s*(\d+)", re.IGNORECASE)


def parse_market_type(title: str) -> str:
    """Return one of ``elimination``, ``season_win``, or ``unknown``.

    The watchlist exporter uses this to decide which side of the
    YES contract represents "model predicts elimination" — they're
    opposite directions on the two market types.
    """
    if not title:
        return "unknown"
    for p in _ELIM_PATTERNS:
        if p.search(title):
            return "elimination"
    for p in _SEASON_WIN_PATTERNS:
        if p.search(title):
            return "season_win"
    return "unknown"


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _yes_price_dollars(market: dict) -> float | None:
    """Best-effort YES implied probability. Tries the new
    ``yes_ask_dollars`` field, then the legacy ``yes_ask`` (cents),
    then derives from no_ask. Same idiom as the tennis bot."""
    ya = (_to_float(market.get("yes_ask_dollars"))
           or (_to_float(market.get("yes_ask")) or 0.0) / 100.0)
    if ya:
        return max(0.01, min(0.99, ya))
    na = (_to_float(market.get("no_ask_dollars"))
           or (_to_float(market.get("no_ask")) or 0.0) / 100.0)
    if na:
        return max(0.01, min(0.99, 1.0 - na))
    yb = (_to_float(market.get("yes_bid_dollars"))
           or (_to_float(market.get("yes_bid")) or 0.0) / 100.0)
    if yb:
        return max(0.01, min(0.99, yb))
    return None


def _ask_cents(market: dict, side: str) -> int | None:
    if side == "yes":
        d = _to_float(market.get("yes_ask_dollars"))
        if d is not None:
            return int(round(d * 100))
        c = _to_float(market.get("yes_ask"))
        return int(c) if c is not None else None
    d = _to_float(market.get("no_ask_dollars"))
    if d is not None:
        return int(round(d * 100))
    c = _to_float(market.get("no_ask"))
    return int(c) if c is not None else None


def _spread_cents(market: dict) -> float | None:
    ya_d = _to_float(market.get("yes_ask_dollars"))
    yb_d = _to_float(market.get("yes_bid_dollars"))
    if ya_d is not None and yb_d is not None:
        return (ya_d - yb_d) * 100.0
    ya = _to_float(market.get("yes_ask"))
    yb = _to_float(market.get("yes_bid"))
    if ya is not None and yb is not None:
        return float(ya) - float(yb)
    return None


def _volume(market: dict) -> float | None:
    return _to_float(market.get("volume_fp")) or _to_float(market.get("volume"))


def _open_interest(market: dict) -> float | None:
    return (_to_float(market.get("open_interest_fp"))
            or _to_float(market.get("open_interest")))


def parse_contestant(title: str) -> str | None:
    if not title:
        return None
    for pat in _ELIM_PATTERNS + _SEASON_WIN_PATTERNS:
        m = pat.search(title)
        if m:
            name = m.group("name").strip()
            if name.lower() in {"the", "next", "first", "no", "yes",
                                 "survivor", "season"}:
                continue
            return name
    m = _NAME_FALLBACK.search(title)
    if m:
        name = m.group("name").strip()
        if name.lower() not in {"the", "next", "first", "no", "yes",
                                  "survivor", "season", "will"}:
            return name
    return None


def parse_episode(market: dict) -> int | None:
    """Episode number from the event title / market title / rules."""
    for key in ("event_title", "title", "yes_sub_title", "subtitle",
                 "rules_primary", "rules_secondary"):
        v = market.get(key)
        if not v:
            continue
        m = _EPISODE_RE.search(str(v))
        if m:
            return int(m.group(1))
    return None


def parse_season(series_ticker: str | None,
                 market: dict | None = None) -> int | None:
    """Season number from the series ticker (KXSURVIVOR50 -> 50).
    Falls back to the rules text if the ticker doesn't carry digits."""
    if series_ticker:
        digits = re.search(r"(\d{2,3})$", series_ticker)
        if digits:
            return int(digits.group(1))
    if market:
        for key in ("rules_primary", "rules_secondary", "title"):
            v = market.get(key)
            if not v:
                continue
            m = _SEASON_RE.search(str(v))
            if m:
                return int(m.group(1))
    return None


def _client():
    try:
        from kalshi_sdk import KalshiClient
    except ImportError as exc:
        raise RuntimeError(
            "kalshi_sdk not installed in this venv — pip install -e "
            "../kalshi_sdk (or set up the editable dep in your local "
            "checkout)"
        ) from exc
    api_key = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    pkey = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not api_key or not pkey:
        raise RuntimeError(
            "KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set in "
            "the env (see deploy/survivor-elimination.service)"
        )
    return KalshiClient(api_key_id=api_key, private_key_path=pkey)


def _list_survivor_series(c, prefix: str) -> List[str]:
    """Enumerate every series whose ticker starts with `prefix`.

    Probes — in order:
      1. The bare prefix itself (Kalshi currently lists every
         contestant's win-season market under the single
         ``KXSURVIVOR`` series rather than per-season).
      2. ``iter_series`` if the SDK exposes it.
      3. Probe ``{prefix}{n}`` for n=40..79 in case a future
         season ships as ``KXSURVIVOR50`` / ``KXSURVIVOR51``.
    """
    tickers: List[str] = []
    iter_open = getattr(c, "iter_open_markets", None)
    if iter_open is not None:
        try:
            for _ in iter_open(series_ticker=prefix):
                tickers.append(prefix)
                break
        except Exception as exc:  # noqa: BLE001
            log.warning("probe of %s failed: %s", prefix, exc)

    iter_series = getattr(c, "iter_series", None)
    if iter_series is not None:
        try:
            for s in iter_series():
                t = (s.get("ticker") or "").strip()
                if t.startswith(prefix) and t not in tickers:
                    tickers.append(t)
        except Exception as exc:  # noqa: BLE001
            log.warning("iter_series failed: %s — using probe fallback", exc)

    if iter_open is not None:
        for n in range(40, 80):
            cand = f"{prefix}{n}"
            if cand in tickers:
                continue
            try:
                for _ in iter_open(series_ticker=cand):
                    tickers.append(cand)
                    break
            except Exception:  # noqa: BLE001
                continue

    return tickers or [prefix]


def fetch_survivor_markets(prefix: str | None = None,
                            inter_series_pause_s: float = 1.0,
                            ) -> List[Dict[str, Any]]:
    """Pull every open market across every Survivor series.

    Returns a flat list of raw market dicts. Each carries an extra
    ``_series_ticker`` key with the parent series so callers can
    group by season without re-querying.
    """
    cfg = load_config()
    prefix = (prefix or cfg["kalshi"]["series_ticker_prefix"]).strip()
    c = _client()
    series_list = _list_survivor_series(c, prefix)
    log.info("discovered survivor series: %s", series_list)
    out: List[Dict[str, Any]] = []
    for s in series_list:
        try:
            for m in c.iter_open_markets(series_ticker=s):
                m = dict(m)
                m["_series_ticker"] = s
                out.append(m)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch %s failed: %s", s, exc)
        time.sleep(inter_series_pause_s)
    log.info("fetched %d survivor markets across %d series",
              len(out), len(series_list))
    return out


def normalise_markets(markets: List[Dict[str, Any]]
                       ) -> List[Dict[str, Any]]:
    """Convert raw Kalshi markets into the live-scorer's input schema.

    One record per market (= per contestant in most cases). The scorer
    expects:

      market_id, ticker, series_ticker, season, episode, contestant,
      market_prob_eliminated, yes_ask_cents, no_ask_cents,
      spread_cents, volume, open_interest,
      expected_expiration_time, last_updated, title
    """
    out: List[Dict[str, Any]] = []
    for m in markets:
        series_ticker = m.get("_series_ticker") or ""
        season = parse_season(series_ticker, m)
        episode = parse_episode(m)
        title = m.get("title") or m.get("yes_sub_title") or ""
        contestant = parse_contestant(title)
        if not contestant:
            # Wildcard markets ("Will there be a tribal council?" etc.)
            # don't have a parseable contestant — skip from the scoring
            # set but keep them visible in the dashboard via the raw
            # market view.
            continue
        market_type = parse_market_type(title)
        yes_p = _yes_price_dollars(m)
        # For season-winner markets the YES price is P(wins season).
        # For elimination markets it's P(eliminated). Surviving-type
        # titles ("Will Sam survive ep 7?") match _ELIM_PATTERNS but
        # mean the opposite direction — we flip them here.
        is_survive = bool(title and re.search(r"\bsurvive\b", title, re.I))
        if market_type == "elimination" and is_survive and yes_p is not None:
            market_prob_eliminated = 1.0 - yes_p
            market_prob_win_season = None
        elif market_type == "elimination":
            market_prob_eliminated = yes_p
            market_prob_win_season = None
        elif market_type == "season_win":
            market_prob_eliminated = None
            market_prob_win_season = yes_p
        else:
            market_prob_eliminated = yes_p
            market_prob_win_season = None
        out.append({
            "market_id": m.get("ticker") or "",
            "ticker": m.get("ticker") or "",
            "event_ticker": m.get("event_ticker") or "",
            "series_ticker": series_ticker,
            "season": season,
            "episode": episode,
            "contestant": contestant,
            "market_type": market_type,
            "market_prob_eliminated": market_prob_eliminated,
            "market_prob_win_season": market_prob_win_season,
            "yes_ask_cents": _ask_cents(m, "yes"),
            "no_ask_cents": _ask_cents(m, "no"),
            "spread_cents": _spread_cents(m),
            "volume": _volume(m),
            "open_interest": _open_interest(m),
            "expected_expiration_time": m.get("expected_expiration_time"),
            "last_updated": m.get("last_updated_time")
                              or m.get("last_quote_update_time")
                              or m.get("last_price_update_time"),
            "title": title,
            "rules_primary": m.get("rules_primary"),
            "status": (m.get("status") or "").lower(),
        })
    return out
