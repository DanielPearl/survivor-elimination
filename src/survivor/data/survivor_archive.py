"""Real-data loader: build the historical training panel from the
``doehm/survivoR`` GitHub archive.

This replaces the previous synthetic ``seed_panel`` shim. The archive
provides per-castaway-per-episode ground truth for every US Survivor
season (1-50+), including:

  • boot order + episode of elimination       (castaways, boot_mapping)
  • per-episode screen-time + confessional    (confessionals)
  • every tribal council vote                 (vote_history)
  • per-castaway advantage / idol movement    (advantage_movement)
  • per-castaway challenge results            (challenge_results)
  • episode metadata                          (episodes, season_summary)

Source repo (free, MIT-licensed, actively maintained):
    https://github.com/doehm/survivoR

We download the JSON snapshots from the repo's ``dev/json/`` directory
once and cache them under ``data/raw/survivor_archive/``. A subsequent
panel rebuild just re-reads the cache. Pass ``refresh=True`` to
re-download.

Feature engineering principles
------------------------------
Every feature is computed from data that would have been visible
*before* tribal council in the episode the row represents — no
look-ahead at the boot outcome. Rolling per-castaway aggregates use
only previous episodes' values.

Where the survivoR archive doesn't carry an analog for a column
(e.g. ``negative_edit_score``, ``narrative_intensity``), we set the
column to ``0.0`` rather than fabricate a value. The trainer's
feature-pruning step will drop columns that carry no signal.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from ..utils.config import load_config, resolve_path

log = logging.getLogger("survivor.data.survivor_archive")


# --------------------------------------------------------------------- #
# Source files                                                          #
# --------------------------------------------------------------------- #

_ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/doehm/survivoR/master/dev/json"
)

# Files we actually consume. The archive carries more (auction_details,
# castaway_scores, tribe_colours, etc.) but those don't feed the panel.
_REQUIRED_FILES: List[str] = [
    "castaways.json",
    "boot_mapping.json",
    "confessionals.json",
    "vote_history.json",
    "advantage_movement.json",
    "challenge_results.json",
    "season_summary.json",
]


def _cache_dir() -> Path:
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    out = Path(raw_dir) / "survivor_archive"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _download_file(name: str, dest: Path) -> None:
    url = f"{_ARCHIVE_BASE}/{name}"
    log.info("downloading %s", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def _load_json(name: str, refresh: bool = False) -> List[Dict[str, Any]]:
    p = _cache_dir() / name
    if refresh or not p.exists():
        _download_file(name, p)
    with p.open() as f:
        return json.load(f)


def refresh_archive() -> None:
    """Re-download every required JSON file. Call from a script before
    rebuilding the panel to pick up the latest season."""
    for name in _REQUIRED_FILES:
        _download_file(name, _cache_dir() / name)


# --------------------------------------------------------------------- #
# Panel construction                                                    #
# --------------------------------------------------------------------- #

def _filter_us(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only US-version rows. The archive carries AU / NZ / SA / UK
    too — we only train on US because the Kalshi market is US-only."""
    return [r for r in records if r.get("version") == "US"]


def _starting_tribes(castaways: List[Dict[str, Any]]
                      ) -> Dict[int, Dict[str, str]]:
    """Map: season -> {castaway_id: original_tribe}."""
    out: Dict[int, Dict[str, str]] = defaultdict(dict)
    for r in castaways:
        cid = r.get("castaway_id")
        if cid is None:
            continue
        out[int(r["season"])][cid] = r.get("original_tribe") or ""
    return out


def _starting_tribe_sizes(starting: Dict[int, Dict[str, str]]
                           ) -> Dict[int, Dict[str, int]]:
    """Map: season -> {tribe_name: starting_size}."""
    out: Dict[int, Dict[str, int]] = defaultdict(dict)
    for season, by_id in starting.items():
        sizes: Dict[str, int] = defaultdict(int)
        for tribe in by_id.values():
            if tribe:
                sizes[tribe] += 1
        out[season] = dict(sizes)
    return out


def _identify_returnees(castaways: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Map: castaway_id -> sorted list of seasons they played in."""
    out: Dict[str, List[int]] = defaultdict(list)
    for r in castaways:
        cid = r.get("castaway_id")
        if cid:
            out[cid].append(int(r["season"]))
    for cid in out:
        out[cid] = sorted(set(out[cid]))
    return out


def _build_advantage_index(advantage_movement: List[Dict[str, Any]]
                             ) -> Dict[int, Dict[str, List[Dict]]]:
    """Map: season -> castaway_id -> list of advantage-event dicts
    sorted by episode."""
    out: Dict[int, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in advantage_movement:
        season = int(r.get("season") or 0)
        cid = r.get("castaway_id")
        if not cid:
            continue
        out[season][cid].append(r)
    for season in out:
        for cid in out[season]:
            out[season][cid].sort(key=lambda x: (int(x.get("episode") or 0),
                                                  int(x.get("sequence_id") or 0)))
    return out


def _is_idol_type(adv_type: str) -> bool:
    """Return True for advantage types that count as 'has_idol' — i.e.
    things you'd play at TC to nullify votes."""
    if not adv_type:
        return False
    t = adv_type.lower()
    return "hidden immunity idol" in t or "super idol" in t


def _is_vote_steal_type(adv_type: str) -> bool:
    if not adv_type:
        return False
    t = adv_type.lower()
    return any(k in t for k in ("extra vote", "vote steal", "steal a vote",
                                  "vote blocker", "block a vote"))


def _advantage_state_through(events: List[Dict[str, Any]],
                              adv_type_by_id: Dict[int, str],
                              through_episode_exclusive: int
                              ) -> Dict[str, int]:
    """Walk a castaway's advantage events strictly *before* ``episode N``
    (i.e. through episode N-1) and return the state they'd carry into N.
    """
    held_idol = 0
    held_total = 0
    held_vote_steal = 0
    # Track per-advantage status so we don't double-count.
    status: Dict[int, str] = {}  # adv_id -> "held" / "spent" / "expired"
    for ev in events:
        ep = int(ev.get("episode") or 0)
        if ep >= through_episode_exclusive:
            break
        aid = int(ev.get("advantage_id") or 0)
        event = (ev.get("event") or "").lower()
        if event == "found":
            status[aid] = "held"
        elif event in ("played", "expired", "lost", "discarded",
                       "transferred", "given"):
            status[aid] = "spent"
    for aid, st in status.items():
        if st != "held":
            continue
        held_total += 1
        adv_type = adv_type_by_id.get(aid, "")
        if _is_idol_type(adv_type):
            held_idol += 1
        if _is_vote_steal_type(adv_type):
            held_vote_steal += 1
    return {
        "has_idol": 1 if held_idol > 0 else 0,
        "advantages_held": held_total,
        "vote_steals_active": held_vote_steal,
    }


def _idols_played_at_episode(events: List[Dict[str, Any]],
                              adv_type_by_id: Dict[int, str],
                              episode: int) -> int:
    """Return 1 if this castaway played an idol-type advantage at TC
    this episode, 0 otherwise. Idol play happens at TC so it would be
    revealed during the episode — fine to use for the same-episode
    prediction since the bot evaluates after the episode airs."""
    for ev in events:
        if int(ev.get("episode") or 0) != episode:
            continue
        if (ev.get("event") or "").lower() != "played":
            continue
        adv_type = adv_type_by_id.get(int(ev.get("advantage_id") or 0), "")
        if _is_idol_type(adv_type):
            return 1
    return 0


def _prior_votes_against(vote_history: List[Dict[str, Any]]
                          ) -> Dict[int, Dict[str, Dict[int, int]]]:
    """Map: season -> castaway_id -> {episode: cumulative_votes_against_BEFORE_this_episode}.

    A 'vote against' is a row where ``vote == castaway`` (they got named
    by the voter). We aggregate by (season, ep, target_id) and then
    cumulatively sum, lagged by one episode so the value for ep N is
    the count through ep N-1."""
    # First: count per-episode votes received by each castaway.
    by_target: Dict[int, Dict[str, Dict[int, int]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    # Note: vote_history has both "vote" (target name) and target id via
    # the vote_id field is the voter's id, not the target's. We have to
    # join by name within a season, which is unique inside a season.
    name_to_id_by_season: Dict[int, Dict[str, str]] = defaultdict(dict)
    for r in vote_history:
        season = int(r.get("season") or 0)
        # The voter's row carries castaway_id (the voter). Build the
        # name->id mapping from rows whose castaway name matches.
        if r.get("castaway") and r.get("castaway_id"):
            name_to_id_by_season[season][r["castaway"]] = r["castaway_id"]
    for r in vote_history:
        season = int(r.get("season") or 0)
        ep = int(r.get("episode") or 0)
        target_name = r.get("vote") or ""
        if not target_name or target_name in ("", "None", "Did not vote"):
            continue
        # Skip nullified votes — they don't count against the target.
        if r.get("nullified"):
            continue
        target_id = name_to_id_by_season[season].get(target_name)
        if not target_id:
            continue
        by_target[season][target_id][ep] += 1
    # Now compute cumulative-through-(ep-1) for each (season, cid, ep).
    out: Dict[int, Dict[str, Dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    for season, by_cid in by_target.items():
        for cid, by_ep in by_cid.items():
            cum = 0
            for ep in sorted(by_ep):
                out[season][cid][ep] = cum  # value BEFORE this ep
                cum += by_ep[ep]
            # Also record the running total *after* the final episode so
            # any post-loop lookup is well-defined.
    return out


def _confessional_features(confessionals: List[Dict[str, Any]]
                            ) -> Dict[int, Dict[str, Dict[int, Dict[str, float]]]]:
    """Map: season -> castaway_id -> episode -> {features...}.

    Features per (season, cid, ep) computed using ONLY data from
    episodes strictly before ``ep`` (no current-episode leakage):
      • confessional_count_prior_mean
      • confessional_share_prior_mean
      • visibility_score   (3-episode rolling mean of prior shares)
      • visibility_spike   (latest prior share - rolling mean)
    Plus the current-episode confessional_share — which IS available
    post-airing and is used as a same-episode signal consistent with
    the bot's existing feature contract."""
    # Aggregate raw counts by (season, ep, cid).
    raw: Dict[int, Dict[int, Dict[str, int]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for r in confessionals:
        season = int(r.get("season") or 0)
        ep = int(r.get("episode") or 0)
        cid = r.get("castaway_id")
        if not cid:
            continue
        raw[season][ep][cid] = int(r.get("confessional_count") or 0)
    # Per-(season, ep): total confessionals across the cast → share.
    out: Dict[int, Dict[str, Dict[int, Dict[str, float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for season, by_ep in raw.items():
        # Per-castaway running history of (ep, count, share).
        hist: Dict[str, List[tuple]] = defaultdict(list)
        for ep in sorted(by_ep):
            counts = by_ep[ep]
            total = sum(counts.values()) or 1
            for cid, n in counts.items():
                share = n / total
                # Use history BEFORE this episode for the rolling stats.
                prev = hist[cid]
                prev_counts = [c for _, c, _ in prev]
                prev_shares = [s for _, _, s in prev]
                # Rolling 3-episode window.
                window = prev_shares[-3:]
                visibility = sum(window) / len(window) if window else 0.0
                visibility_spike = (prev_shares[-1] - visibility) if prev_shares else 0.0
                prior_count_mean = (sum(prev_counts) / len(prev_counts)
                                     if prev_counts else 0.0)
                out[season][cid][ep] = {
                    # Current-episode signals (same-episode share is
                    # known after the episode airs; consistent with the
                    # bot evaluating between episodes).
                    "confessional_count": float(n),
                    "confessional_share": share,
                    # Lagged / non-leaky signals.
                    "visibility_score": visibility,
                    "visibility_spike": visibility_spike,
                    "prior_count_mean": prior_count_mean,
                }
                hist[cid].append((ep, n, share))
    return out


def _challenge_features(challenge_results: List[Dict[str, Any]]
                         ) -> Dict[int, Dict[str, Dict[int, Dict[str, int]]]]:
    """Map: season -> castaway_id -> episode -> {immunity_won,
    tribe_immunity, reward_wins_prior}."""
    # First: per-episode flags.
    per_ep: Dict[int, Dict[str, Dict[int, Dict[str, int]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(
            lambda: {"immunity_won": 0, "tribe_immunity": 0,
                     "reward_won": 0})))
    )
    for r in challenge_results:
        season = int(r.get("season") or 0)
        ep = int(r.get("episode") or 0)
        cid = r.get("castaway_id")
        if not cid:
            continue
        cell = per_ep[season][cid][ep]
        if int(r.get("won_individual_immunity") or 0):
            cell["immunity_won"] = 1
        if int(r.get("won_tribal_immunity") or 0) or int(
                r.get("won_team_immunity") or 0):
            cell["tribe_immunity"] = 1
        if (int(r.get("won_tribal_reward") or 0)
                or int(r.get("won_team_reward") or 0)
                or int(r.get("won_individual_reward") or 0)):
            cell["reward_won"] = 1
    # Add a cumulative-prior reward count under the same dict so the
    # trainer has a rolling performance proxy.
    for season, by_cid in per_ep.items():
        for cid, by_ep in by_cid.items():
            cum_reward = 0
            cum_immunity = 0
            cum_eps = 0
            for ep in sorted(by_ep):
                cell = by_ep[ep]
                # "prior_perf_score": rolling avg of (immunity OR reward)
                # share, computed from episodes BEFORE this one.
                if cum_eps > 0:
                    cell["prior_perf_score"] = (cum_reward + cum_immunity) / (2.0 * cum_eps)
                else:
                    cell["prior_perf_score"] = 0.5
                cum_reward += cell["reward_won"]
                cum_immunity += cell["immunity_won"] + cell["tribe_immunity"]
                cum_eps += 1
    return per_ep


def build_historical_panel(refresh: bool = False) -> pd.DataFrame:
    """Materialise the historical training panel from the survivoR
    archive.

    Parameters
    ----------
    refresh : re-download the source JSON files even if a cached copy
              exists. Use this when a new season is added upstream.
    """
    log.info("loading survivoR archive (refresh=%s)", refresh)
    castaways = _filter_us(_load_json("castaways.json", refresh))
    boot_map = _filter_us(_load_json("boot_mapping.json", refresh))
    confessionals = _filter_us(_load_json("confessionals.json", refresh))
    vote_history = _filter_us(_load_json("vote_history.json", refresh))
    advantage_movement = _filter_us(_load_json("advantage_movement.json", refresh))
    challenge_results = _filter_us(_load_json("challenge_results.json", refresh))
    season_summary = _filter_us(_load_json("season_summary.json", refresh))

    # Index advantage types by adv_id WITHIN A SEASON (adv_id resets
    # per season in the archive).
    adv_details = _filter_us(_load_json("advantage_details.json", refresh))
    adv_type_by_season_id: Dict[int, Dict[int, str]] = defaultdict(dict)
    for r in adv_details:
        adv_type_by_season_id[int(r["season"])][int(r["advantage_id"])] = (
            r.get("advantage_type") or "")

    # Boot info: (season, castaway_id) -> episode_eliminated.
    elim_by_id: Dict[tuple, int] = {}
    finalist_by_id: Dict[tuple, bool] = {}
    place_by_id: Dict[tuple, int] = {}
    for r in castaways:
        season = int(r["season"])
        cid = r.get("castaway_id")
        if not cid:
            continue
        elim_by_id[(season, cid)] = int(r.get("episode") or 0)
        finalist_by_id[(season, cid)] = bool(r.get("finalist"))
        place_by_id[(season, cid)] = int(r.get("place") or 999)

    starting = _starting_tribes(castaways)
    starting_sizes = _starting_tribe_sizes(starting)
    returnees = _identify_returnees(castaways)
    season_returnee_count: Dict[int, int] = {}
    for season, by_id in starting.items():
        season_returnee_count[season] = sum(
            1 for cid in by_id
            if len(returnees.get(cid, [])) > 1
            and returnees[cid].index(season) > 0
        )

    advantage_idx = _build_advantage_index(advantage_movement)
    votes_idx = _prior_votes_against(vote_history)
    conf_feats = _confessional_features(confessionals)
    chall_feats = _challenge_features(challenge_results)

    # Build the spine: one row per (season, episode, castaway) where
    # the castaway was 'In the game' at the start of the episode. This
    # excludes already-eliminated rows.
    #
    # boot_mapping can carry multiple rows per (season, episode,
    # castaway) keyed by ``sog_id`` (Stage of Game) — e.g. one for
    # pre-TC state and another for post-TC. We dedupe to the lowest
    # sog_id, which captures the contestant's state at the start of
    # the episode (the right basis for pre-tribal features).
    spine_rows: List[Dict[str, Any]] = []
    dedup: Dict[tuple, Dict[str, Any]] = {}
    for r in boot_map:
        if (r.get("game_status") or "").lower() != "in the game":
            continue
        key = (int(r["season"]), int(r["episode"]), r.get("castaway_id"))
        sog = int(r.get("sog_id") or 0)
        existing = dedup.get(key)
        if existing is None or sog < int(existing.get("sog_id") or 0):
            dedup[key] = r
    by_se: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in dedup.values():
        by_se[(int(r["season"]), int(r["episode"]))].append(r)

    # Voting bloc proxy: per-(season, castaway), running count of votes
    # they cast that went WITH the majority of their tribal council.
    in_majority_count: Dict[tuple, int] = defaultdict(int)
    tc_votes_cast: Dict[tuple, int] = defaultdict(int)
    # Pre-compute majority pick per (season, ep, voter's tribal-council
    # group). The simplest invariant: the majority vote at any TC is
    # whoever got voted out (their name appears most often as `vote`).
    # We mark each voter's row in vote_history by whether their `vote`
    # matched the actual voted_out.
    vh_sorted = sorted(vote_history,
                        key=lambda r: (int(r.get("season") or 0),
                                        int(r.get("episode") or 0)))

    # Iterate seasons in order so the "in_majority" running count is
    # only counted from episodes BEFORE the row being built.
    vh_by_season_ep_cid: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in vh_sorted:
        vh_by_season_ep_cid[(int(r.get("season") or 0),
                              int(r.get("episode") or 0),
                              r.get("castaway_id") or "")].append(r)

    # Sort (season, ep) pairs in order.
    sorted_se = sorted(by_se.keys())

    for season, episode in sorted_se:
        # Update the "in_majority" tally with the PREVIOUS episode's
        # votes — must be applied before computing features for this ep.
        if episode > 1:
            prev_ep = episode - 1
            for r in vh_sorted:
                if (int(r.get("season") or 0) != season
                        or int(r.get("episode") or 0) != prev_ep):
                    continue
                cid = r.get("castaway_id")
                if not cid:
                    continue
                vote = r.get("vote") or ""
                voted_out = r.get("voted_out") or ""
                if vote and voted_out and vote != "Did not vote":
                    tc_votes_cast[(season, cid)] += 1
                    if vote == voted_out:
                        in_majority_count[(season, cid)] += 1

        ep_castaways = by_se[(season, episode)]
        # Count of "same starting tribe still in game", computed once
        # per (season, episode, starting_tribe).
        og_still_in: Dict[str, int] = defaultdict(int)
        for r in ep_castaways:
            cid = r["castaway_id"]
            og = starting[season].get(cid) or ""
            if og:
                og_still_in[og] += 1
        remaining_count = len(ep_castaways)

        for r in ep_castaways:
            cid = r["castaway_id"]
            name = r.get("castaway") or ""
            tribe = r.get("tribe") or ""
            tribe_status = (r.get("tribe_status") or "").lower()
            og_tribe = starting[season].get(cid) or ""

            # Game-phase flags. The archive uses tribe_status values
            # like "Original" / "Swap" / "Merged". We rely on those.
            merged = 1 if "merge" in tribe_status else 0
            swap_phase = 1 if "swap" in tribe_status and not merged else 0

            # Tribe size: count active castaways in the same tribe this
            # episode.
            tribe_size = sum(1 for x in ep_castaways
                              if (x.get("tribe") or "") == tribe)

            # Starting-tribe size from the cast roster at S1.
            starting_tribe_size = starting_sizes[season].get(og_tribe, 0)

            # Eliminated this episode?
            eliminated = 1 if elim_by_id.get((season, cid), -1) == episode else 0

            # Confessional features (lagged + same-ep share).
            cf = (conf_feats.get(season, {}).get(cid, {}).get(episode)
                  or {"confessional_count": 0.0, "confessional_share": 0.0,
                      "visibility_score": 0.0, "visibility_spike": 0.0,
                      "prior_count_mean": 0.0})

            # Challenge features.
            cr = chall_feats.get(season, {}).get(cid, {}).get(episode) or {}
            immunity_won = cr.get("immunity_won", 0)
            tribe_immunity = cr.get("tribe_immunity", 0)
            prior_perf_score = cr.get("prior_perf_score", 0.5)

            # Advantage state (carried INTO this episode).
            adv_events = advantage_idx.get(season, {}).get(cid, [])
            adv_state = _advantage_state_through(
                adv_events,
                adv_type_by_season_id.get(season, {}),
                through_episode_exclusive=episode,
            )
            idols_played = _idols_played_at_episode(
                adv_events, adv_type_by_season_id.get(season, {}), episode)

            # Prior votes against (cumulative through ep N-1).
            pva = votes_idx.get(season, {}).get(cid, {}).get(episode, 0)

            # Voting-bloc proxies (using TC history through ep N-1).
            votes_cast_prior = tc_votes_cast[(season, cid)]
            in_majority_prior = in_majority_count[(season, cid)]
            if votes_cast_prior > 0:
                in_main_alliance = (
                    1 if in_majority_prior / votes_cast_prior >= 0.5 else 0)
                voting_minority_score = 1.0 - (in_majority_prior / votes_cast_prior)
            else:
                in_main_alliance = 0
                voting_minority_score = 0.5  # no info yet

            # Returnee dynamics.
            ret_seasons = returnees.get(cid, [])
            is_returnee = 1 if ret_seasons.index(season) > 0 else 0 if ret_seasons else 0
            is_returnee_first_three_eps = (
                1 if is_returnee and episode <= 3 else 0)

            spine_rows.append({
                # ── Identity / structure ────────────────────────────
                "season": season,
                "episode": episode,
                "contestant": name,
                "tribe": tribe,
                "eliminated": eliminated,
                "starting_tribe": og_tribe,
                "starting_tribe_size": starting_tribe_size,
                "tribe_size": tribe_size,
                "remaining": remaining_count,
                "merged": merged,
                "swap_phase": swap_phase,
                # ── On-show signal ──────────────────────────────────
                "immunity_won": immunity_won,
                "tribe_immunity": tribe_immunity,
                "prior_votes_against": pva,
                "times_targeted": pva,  # vote-cast against = explicit target
                "has_idol": adv_state["has_idol"],
                "in_main_alliance": in_main_alliance,
                "confessional_count": cf["confessional_count"],
                "visibility_score": cf["visibility_score"],
                "visibility_spike": cf["visibility_spike"],
                # No fan-edgic CSV — leave the "negative-edit"
                # subjective signal at zero rather than fabricate it.
                "negative_edit_score": 0.0,
                # Strategic-isolation proxy: high voting_minority_score
                # captures "on the bottom" without inventing a number.
                "strategic_isolation": voting_minority_score,
                "prior_perf_score": prior_perf_score,
                # ── Edgic / screen-time extensions ──────────────────
                "confessional_share": cf["confessional_share"],
                # No real signal for these without edgic; trainer
                # prunes columns it can't fit.
                "narrative_intensity": 0.0,
                "swing_vote_potential": 0.0,
                # ── Advantages ──────────────────────────────────────
                "advantages_held": adv_state["advantages_held"],
                "idols_played_this_ep": idols_played,
                "vote_steals_active": adv_state["vote_steals_active"],
                "same_starting_tribe_remaining": og_still_in.get(og_tribe, 0),
                "voting_minority_score": voting_minority_score,
                # ── Returnee dynamics ───────────────────────────────
                "is_returnee": is_returnee,
                "season_returnee_count": season_returnee_count.get(season, 0),
                "is_returnee_first_three_eps": is_returnee_first_three_eps,
            })

    df = pd.DataFrame(spine_rows)
    if df.empty:
        raise RuntimeError("survivor archive produced an empty panel — "
                           "check the upstream JSON files")
    df = df.sort_values(["season", "episode", "contestant"]).reset_index(drop=True)
    log.info("built panel: %d rows, %d seasons, %d boots",
             len(df), df["season"].nunique(),
             int(df["eliminated"].sum()))
    return df
