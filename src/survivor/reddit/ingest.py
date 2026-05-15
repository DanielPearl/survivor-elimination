"""Reddit ingestion for Survivor contestant signals.

For each active contestant we want to compute, every tick:

  reddit_mention_count       count of comments/posts mentioning them
  reddit_boot_pick_count     count of "X is going home" / "boot is X" hits
  reddit_sentiment           mean sentiment polarity of their mentions
  reddit_visibility_score    mention count / total mentions across cast
  reddit_target_share        boot_pick_count / sum(boot_picks)

Implementation
--------------
Pulls the configured subreddits' recent comments + new submissions
using PRAW. Filters to the configured `lookback_hours`. Tags each
comment with every contestant whose name (first + first+last, +
common short forms) appears in it. Sentiment via TextBlob's polarity
(works offline, no extra model download).

If PRAW or Reddit creds are missing the module exposes a `safe_pull`
that returns an all-zero record for every contestant — the model is
still scored, just without the Reddit features. The dashboard surfaces
"Reddit feed unavailable" in that case.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..utils.config import load_config

log = logging.getLogger("survivor.reddit.ingest")


@dataclass
class ContestantSignal:
    contestant: str
    mention_count: int = 0
    boot_pick_count: int = 0
    sentiment_sum: float = 0.0
    sentiment_n: int = 0
    target_terms_hit: int = 0
    snippets: List[str] = field(default_factory=list)

    @property
    def sentiment_mean(self) -> float:
        return (self.sentiment_sum / self.sentiment_n) if self.sentiment_n else 0.0


def _name_patterns(contestant: str) -> List[re.Pattern]:
    """Build regexes that match the contestant in comment text.

    A contestant entered as "Sam Phalen" matches:
      - "Sam Phalen"
      - "Sam" (as a standalone word — generous, but Survivor uses
        first names almost exclusively in discussion)
    """
    parts = contestant.split()
    pats: List[re.Pattern] = []
    if len(parts) >= 2:
        pats.append(re.compile(rf"\b{re.escape(contestant)}\b", re.IGNORECASE))
    pats.append(re.compile(rf"\b{re.escape(parts[0])}\b", re.IGNORECASE))
    return pats


def _reddit_client():
    """Return an authenticated PRAW client, or None if creds missing."""
    cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    csec = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    ua = os.environ.get("REDDIT_USER_AGENT", "survivor-elimination/0.1").strip()
    if not cid or not csec:
        return None
    try:
        import praw  # type: ignore
    except ImportError:
        log.warning("praw not installed — Reddit ingestion disabled")
        return None
    try:
        return praw.Reddit(client_id=cid, client_secret=csec,
                            user_agent=ua, ratelimit_seconds=30)
    except Exception as exc:  # noqa: BLE001
        log.warning("PRAW init failed: %s", exc)
        return None


def _polarity(text: str) -> float:
    """TextBlob polarity in [-1, 1]. Returns 0 if textblob missing."""
    try:
        from textblob import TextBlob  # type: ignore
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def _iter_comments(client, subreddits: Iterable[str], lookback_seconds: int):
    """Yield reddit comments newer than `lookback_seconds` across subs.

    PRAW's `.new(limit=None)` returns the most recent stream of
    comments; we stop on the first comment older than the cutoff.
    """
    cutoff = time.time() - lookback_seconds
    for sub in subreddits:
        try:
            sr = client.subreddit(sub)
            for c in sr.comments(limit=500):
                if getattr(c, "created_utc", 0) < cutoff:
                    break
                yield c
            for s in sr.new(limit=200):
                if getattr(s, "created_utc", 0) < cutoff:
                    break
                yield s
        except Exception as exc:  # noqa: BLE001
            log.warning("reddit pull from r/%s failed: %s", sub, exc)


def _text_of(item) -> str:
    """Concatenate the text from a comment or submission."""
    parts: List[str] = []
    body = getattr(item, "body", None)
    if body:
        parts.append(str(body))
    title = getattr(item, "title", None)
    if title:
        parts.append(str(title))
    selftext = getattr(item, "selftext", None)
    if selftext:
        parts.append(str(selftext))
    return " ".join(parts)


def pull_signals(contestants: List[str]) -> Dict[str, ContestantSignal]:
    """Return a contestant -> ContestantSignal map.

    Empty-but-keyed for every contestant in `contestants` so the live
    scorer can blindly read the map even when Reddit is unavailable.
    """
    out: Dict[str, ContestantSignal] = {
        c: ContestantSignal(contestant=c) for c in contestants
    }
    cfg = load_config()
    rcfg = cfg.get("reddit", {})
    lookback = int(rcfg.get("lookback_hours", 168)) * 3600
    boot_patterns = [re.compile(p) for p in rcfg.get("boot_prediction_patterns", [])]
    target_terms = [t.lower() for t in rcfg.get("negative_edit_terms", [])]
    subreddits = rcfg.get("subreddits", ["survivor"])

    client = _reddit_client()
    if client is None:
        log.info("reddit creds not configured — returning zero signals")
        return out

    name_patterns = {c: _name_patterns(c) for c in contestants}
    pulled = 0
    for item in _iter_comments(client, subreddits, lookback):
        text = _text_of(item).strip()
        if not text:
            continue
        pulled += 1
        # Sentiment is computed once per comment — cheap if textblob
        # is available, zero otherwise.
        pol = _polarity(text) if len(text) > 12 else 0.0
        for c, pats in name_patterns.items():
            if not any(p.search(text) for p in pats):
                continue
            sig = out[c]
            sig.mention_count += 1
            sig.sentiment_sum += pol
            sig.sentiment_n += 1
            if any(t in text.lower() for t in target_terms):
                sig.target_terms_hit += 1
            # Boot-pick detection — regex captures "X is going home"
            # / "boot is X" / "voting out X" patterns.
            for bp in boot_patterns:
                m = bp.search(text)
                if m and "name" in m.groupdict():
                    named = m.group("name").strip()
                    if any(p.search(named) for p in pats):
                        sig.boot_pick_count += 1
                        if len(sig.snippets) < 5:
                            sig.snippets.append(text[:240])
                        break
    log.info("pulled %d reddit items across %d subreddits",
              pulled, len(subreddits))
    return out


def signals_to_features(signals: Dict[str, ContestantSignal]
                         ) -> Dict[str, Dict[str, float]]:
    """Convert raw signals into the columns the model expects.

    Per-contestant output keys:
      reddit_mention_count       (raw count)
      reddit_boot_pick_count     (raw count)
      reddit_sentiment           (mean polarity)
      reddit_visibility_score    (mention_count / total_mentions, 0..1)
      reddit_target_share        (boot_pick_count / sum(boot_picks))
    """
    total_mentions = sum(s.mention_count for s in signals.values()) or 1
    total_boot_picks = sum(s.boot_pick_count for s in signals.values()) or 1
    out: Dict[str, Dict[str, float]] = {}
    for c, s in signals.items():
        out[c] = {
            "reddit_mention_count": float(s.mention_count),
            "reddit_boot_pick_count": float(s.boot_pick_count),
            "reddit_sentiment": float(s.sentiment_mean),
            "reddit_visibility_score": s.mention_count / total_mentions,
            "reddit_target_share": s.boot_pick_count / total_boot_picks,
        }
    return out


def safe_pull(contestants: List[str]) -> Dict[str, Dict[str, float]]:
    """One-shot helper: pull + featurise + guard against any internal
    failure (network, rate-limit, missing dep) so the live scorer's
    feed never goes down because of Reddit."""
    try:
        signals = pull_signals(contestants)
    except Exception as exc:  # noqa: BLE001
        log.warning("reddit pull crashed: %s — returning zero signals", exc)
        signals = {c: ContestantSignal(contestant=c) for c in contestants}
    return signals_to_features(signals)
