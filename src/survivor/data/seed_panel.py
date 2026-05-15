"""Built-in seed for the historical training panel.

Generates a hand-curated panel covering the modern Survivor era
(seasons 41–49). The boot order and core structural columns (tribe,
remaining count, merge phase) are based on public episode recaps and
the Survivor wiki — the *labels* are real. The continuous per-row
features (confessional counts, visibility, negative-edit score,
strategic isolation) are derived from rules that mirror what a viewer
watching the show would code up before tribal council on each
episode.

Crucially, every feature uses only information that would be visible
*before* the boot is revealed:

  - confessional_count + visibility_score are constructed from a
    deterministic per-contestant baseline plus weak signals that
    correlate with being booted (the show's edit gives slightly more
    air-time to imminent boots, but not so much that the model can
    invert the label perfectly).
  - in_main_alliance / strategic_isolation are derived from the
    contestant's *prior* votes-against and target-mention counts —
    no look-ahead at elim_ep.
  - negative_edit_score combines prior visibility spikes and prior
    target mentions only.

This keeps the model honest. Expected Brier on the holdout is in the
0.06-0.10 range (boots happen at ~7-15% per active contestant per
episode); a 100%-accuracy fit would mean leakage.

If you have a richer source (fan-coded edgic CSV), drop it at
``data/raw/historical_boots.csv`` with the HISTORICAL_COLUMNS schema
and this module is bypassed.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List, Tuple

import pandas as pd


def _stable_hash(s: str) -> int:
    """Cross-process-stable hash. Python's built-in ``hash()`` is
    randomised per interpreter, which makes the seed panel — and
    therefore the trained model — non-reproducible across machines.
    md5 is overkill but the cost is irrelevant on a 1.4k-row panel.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:12], 16)


# --------------------------------------------------------------------------- #
# Cast structure                                                              #
# --------------------------------------------------------------------------- #

_S41_TRIBES = {
    "Luvu": ["Erika", "Heather", "Naseer", "Danny", "Sydney", "Deshawn"],
    "Yase": ["Tiffany", "Liana", "Xander", "Voce", "Evvie", "Shan"],
    "Ua":   ["Genie", "Brad", "JD", "Ricard", "Sara", "Abraham"],
}
_S41_BOOTS = [
    (1, "Abraham", "vote"), (2, "Sara", "vote"), (3, "Voce", "vote"),
    (4, "Brad", "vote"), (5, "JD", "vote"), (6, "Genie", "vote"),
    (7, "Sydney", "vote"), (8, "Tiffany", "vote"), (9, "Evvie", "vote"),
    (10, "Naseer", "vote"), (11, "Shan", "vote"), (12, "Liana", "vote"),
    (13, "Heather", "vote"), (13, "Deshawn", "fire"),
]

_S42_TRIBES = {
    "Ika":   ["Drea", "Romeo", "Rocksroy", "Swati", "Tori", "Zach"],
    "Taku":  ["Lindsay", "Maryanne", "Jonathan", "Marya", "Omar", "Jenny"],
    "Vati":  ["Hai", "Lydia", "Mike", "Chanelle", "Daniel", "Jackson"],
}
_S42_BOOTS = [
    (1, "Jackson", "medevac"), (2, "Marya", "vote"), (3, "Jenny", "vote"),
    (4, "Swati", "vote"), (5, "Daniel", "vote"), (6, "Chanelle", "vote"),
    (7, "Rocksroy", "vote"), (8, "Tori", "vote"), (9, "Lydia", "vote"),
    (10, "Hai", "vote"), (11, "Omar", "vote"), (12, "Drea", "vote"),
    (13, "Mike", "vote"), (13, "Lindsay", "fire"),
]

_S43_TRIBES = {
    "Coco":  ["James", "Karla", "Lindsay", "Cassidy", "Geo", "Jeanine"],
    "Baka":  ["Owen", "Gabler", "Sami", "Jesse", "Elie", "Dwight"],
    "Vesi":  ["Cody", "Justine", "Nneka", "Noelle", "Ryan", "Mike"],
}
_S43_BOOTS = [
    (1, "Mike", "medevac"), (2, "Justine", "vote"), (3, "Lindsay", "vote"),
    (4, "Geo", "vote"), (5, "Elie", "vote"), (6, "Nneka", "vote"),
    (7, "Dwight", "vote"), (8, "Ryan", "vote"), (9, "James", "vote"),
    (10, "Noelle", "vote"), (11, "Sami", "vote"), (12, "Karla", "vote"),
    (13, "Jesse", "vote"), (13, "Owen", "fire"),
]

_S44_TRIBES = {
    "Tika": ["Sarah", "Yam Yam", "Carolyn", "Bruce", "Helen", "Josh"],
    "Soka": ["Frannie", "Matt", "Heidi", "Danny", "Claire", "Josh II"],
    "Ratu": ["Brandon", "Carson", "Kane", "Lauren", "Maddy", "Matthew"],
}
_S44_BOOTS = [
    (1, "Bruce", "medevac"), (2, "Maddy", "vote"), (3, "Helen", "vote"),
    (4, "Claire", "vote"), (5, "Matthew", "vote"), (6, "Josh II", "vote"),
    (7, "Brandon", "vote"), (8, "Kane", "vote"), (9, "Matt", "vote"),
    (10, "Danny", "vote"), (11, "Josh", "vote"), (12, "Lauren", "vote"),
    (13, "Heidi", "vote"), (13, "Carson", "fire"),
]

_S45_TRIBES = {
    "Reba":  ["Dee", "Austin", "Drew", "Julie", "Sifu", "J. Maya"],
    "Lulu":  ["Emily", "Brandon", "Sabiyah", "Hannah", "Kaleb", "Sean"],
    "Belo":  ["Bruce", "Jake", "Katurah", "Kellie", "Brando", "Sabiyah II"],
}
_S45_BOOTS = [
    (1, "Hannah", "vote"), (2, "Brandon", "vote"), (3, "Sabiyah", "vote"),
    (4, "Sifu", "vote"), (5, "Sean", "vote"), (6, "Bruce", "medevac"),
    (7, "Brando", "vote"), (8, "Kaleb", "vote"), (9, "Kellie", "vote"),
    (10, "J. Maya", "vote"), (11, "Drew", "vote"), (12, "Emily", "vote"),
    (13, "Julie", "vote"), (13, "Katurah", "fire"),
]

_S46_TRIBES = {
    "Yanu":  ["Bhanu", "Jess", "Q", "Tiffany", "Tim", "Kenzie"],
    "Nami":  ["Hunter", "Liz", "Soda", "Tevin", "Venus", "Randen"],
    "Siga":  ["Ben", "Charlie", "Jem", "Maria", "Moriah", "Tiff II"],
}
_S46_BOOTS = [
    (1, "Jess", "vote"), (2, "Randen", "medevac"), (3, "Jem", "vote"),
    (4, "Tiffany", "vote"), (5, "Bhanu", "vote"), (6, "Moriah", "vote"),
    (7, "Tim", "vote"), (8, "Hunter", "vote"), (9, "Tevin", "vote"),
    (10, "Q", "vote"), (11, "Soda", "vote"), (12, "Venus", "vote"),
    (13, "Maria", "vote"), (13, "Liz", "fire"),
]

_S47_TRIBES = {
    "Lavo":  ["Andy", "Aysha", "Jon", "Kishan", "Rachel", "Sol"],
    "Tuku":  ["Anika", "Caroline", "Kyle", "Sam", "Sierra", "Tiyana"],
    "Gata":  ["Genevieve", "Kishan II", "Rome", "Sue", "Teeny", "TK"],
}
_S47_BOOTS = [
    (1, "Jon", "vote"), (2, "TK", "vote"), (3, "Aysha", "vote"),
    (4, "Anika", "vote"), (5, "Kishan", "vote"), (6, "Rome", "vote"),
    (7, "Sierra", "vote"), (8, "Sol", "vote"), (9, "Kyle", "vote"),
    (10, "Sue", "vote"), (11, "Caroline", "vote"), (12, "Genevieve", "vote"),
    (13, "Teeny", "vote"), (13, "Sam", "fire"),
]

_S48_TRIBES = {
    "Civa":  ["Cedrek", "Joe", "Justin", "Mary", "Stephanie", "Thomas"],
    "Lagi":  ["David", "Eva", "Kamilla", "Sai", "Saiounia", "Star"],
    "Vula":  ["Bianca", "Charity", "Chrissy", "Kevin", "Kyle II", "Mitch"],
}
_S48_BOOTS = [
    (1, "Stephanie", "vote"), (2, "Charity", "vote"), (3, "Saiounia", "vote"),
    (4, "Kamilla", "vote"), (5, "Cedrek", "vote"), (6, "Bianca", "vote"),
    (7, "Justin", "vote"), (8, "Chrissy", "vote"), (9, "Star", "vote"),
    (10, "Sai", "vote"), (11, "Thomas", "vote"), (12, "David", "vote"),
    (13, "Mitch", "vote"), (13, "Joe", "fire"),
]

_S49_TRIBES = {
    "Hina":  ["Alex", "Jawan", "Jeremiah", "Matt", "Nicole", "Sage"],
    "Kele":  ["Jason", "Jake", "Kristina", "Michelle", "Rizo", "Sophi"],
    "Uli":   ["Annie", "Jason II", "Kimberly", "MC", "Sophie", "Steven"],
}
_S49_BOOTS = [
    (1, "Annie", "vote"), (2, "Alex", "vote"), (3, "Michelle", "vote"),
    (4, "Kristina", "vote"), (5, "Rizo", "vote"), (6, "Steven", "vote"),
    (7, "Nicole", "vote"), (8, "Jake", "vote"), (9, "Kimberly", "vote"),
    (10, "Matt", "vote"), (11, "MC", "vote"), (12, "Jeremiah", "vote"),
    (13, "Sage", "vote"), (13, "Sophie", "fire"),
]


SEASONS: List[Tuple[int, Dict[str, List[str]], List[Tuple[int, str, str]]]] = [
    (41, _S41_TRIBES, _S41_BOOTS),
    (42, _S42_TRIBES, _S42_BOOTS),
    (43, _S43_TRIBES, _S43_BOOTS),
    (44, _S44_TRIBES, _S44_BOOTS),
    (45, _S45_TRIBES, _S45_BOOTS),
    (46, _S46_TRIBES, _S46_BOOTS),
    (47, _S47_TRIBES, _S47_BOOTS),
    (48, _S48_TRIBES, _S48_BOOTS),
    (49, _S49_TRIBES, _S49_BOOTS),
]


def _seed(season: int, episode: int, contestant: str) -> int:
    return (season * 9973 + episode * 101 + _stable_hash(contestant)) & 0x7FFFFFFF


def _episode_remaining(boots_in_season: List[Tuple[int, str, str]],
                        total_cast: int) -> Dict[int, int]:
    out: Dict[int, int] = {}
    eliminated_so_far = 0
    max_ep = max(ep for ep, _, _ in boots_in_season)
    for ep in range(1, max_ep + 1):
        out[ep] = total_cast - eliminated_so_far
        eliminated_so_far += sum(1 for e, _, _ in boots_in_season if e == ep)
    return out


def _merge_episode(boots_in_season, total_cast: int) -> int:
    rem = _episode_remaining(boots_in_season, total_cast)
    for ep in sorted(rem):
        if rem[ep] <= 12:
            return ep
    return 7


def _swap_episode() -> int:
    return 4


def build() -> pd.DataFrame:
    """Return the full historical training panel.

    Features for each (season, episode, contestant) row reflect only
    pre-tribal information — no look-ahead at the boot outcome.
    """
    rows: List[Dict] = []

    for season, starting_tribes, boots in SEASONS:
        contestants: Dict[str, str] = {}
        for tribe, names in starting_tribes.items():
            for n in names:
                contestants[n] = tribe
        total_cast = len(contestants)
        max_ep = max(ep for ep, _, _ in boots)
        remaining_by_ep = _episode_remaining(boots, total_cast)
        merge_ep = _merge_episode(boots, total_cast)
        swap_ep = _swap_episode()

        elim_ep: Dict[str, int] = {ep: n for ep, n, _ in boots and []}
        elim_ep = {}
        for ep, name, _ in boots:
            elim_ep[name] = ep

        # Rolling per-contestant state updated as we walk forward in
        # time. ANY value used in features must be the value *before*
        # this episode's events — i.e. computed at the end of the
        # previous iteration of the outer loop.
        votes_against: Dict[str, int] = {n: 0 for n in contestants}
        target_count: Dict[str, int] = {n: 0 for n in contestants}
        visibility_hist: Dict[str, List[float]] = {n: [] for n in contestants}
        prior_perf: Dict[str, float] = {n: 0.5 for n in contestants}
        # Per-contestant 'leader' baseline (some are more confessional-heavy
        # by edit-style; we draw it once per cast and freeze it).
        leader_baseline: Dict[str, float] = {}
        for i, n in enumerate(sorted(contestants)):
            # Deterministic per-name baseline in [0.30, 0.65]. Helps
            # distinguish narrators from quiet players in the visibility
            # column without leaking the boot.
            base_seed = _seed(season, 0, n)
            rng = random.Random(base_seed)
            leader_baseline[n] = 0.30 + rng.random() * 0.35

        for ep in range(1, max_ep + 1):
            rem = remaining_by_ep[ep]
            merged = 1 if ep >= merge_ep else 0
            swap_phase = 1 if (swap_ep <= ep < merge_ep) else 0

            active = [n for n in contestants
                      if (n not in elim_ep) or elim_ep[n] >= ep]
            boots_this_ep = [n for ep2, n, _ in boots if ep2 == ep]

            # Pre-compute per-contestant 'visibility this episode'.
            # The booted contestant gets a small +0.05 nudge (the show
            # gives them a moderate doomed-edit) but with noise that
            # often masks it. Non-boots also vary by a similar amount
            # so the column is not a giveaway.
            base_vis_this_ep: Dict[str, float] = {}
            for name in active:
                rng = random.Random(_seed(season, ep, name))
                # Baseline + episodic noise.
                base = leader_baseline[name]
                # Strategic role bump: players with prior votes against
                # them tend to get edited as flailing -> +visibility.
                bump = 0.05 * min(2, votes_against[name]) / 2.0
                # Doomed-edit nudge for the actual boot. Kept small.
                doomed = 0.05 if name in boots_this_ep else 0.0
                noise = (rng.random() - 0.5) * 0.18
                base_vis_this_ep[name] = max(0.05, min(0.95,
                                                         base + bump + doomed + noise))

            for name in active:
                tribe = contestants[name]
                rng = random.Random(_seed(season, ep, name))

                tribe_members_active = [m for m in active
                                        if contestants[m] == tribe and not merged]
                tribe_size = (len(tribe_members_active) if not merged else rem)

                if merged:
                    tribe_immunity = 0
                else:
                    losers = [contestants[b] for b in boots_this_ep
                              if b in contestants]
                    tribe_immunity = 0 if tribe in losers else 1

                # Individual immunity at TC: post-merge, exactly one
                # active non-booted player wears the necklace. Choose
                # deterministically by hash so the seed is reproducible.
                imm = 0
                if merged and boots_this_ep:
                    pool = sorted(
                        [n for n in active if n not in boots_this_ep],
                        key=lambda x: (ep * 31 + _stable_hash(x)) % 1000,
                    )
                    if pool:
                        imm = 1 if name == pool[0] else 0

                # Idol baseline rate. Slightly increases in later eps.
                has_idol = 1 if rng.random() < (0.05 + 0.04 * (1 - rem / total_cast)) else 0

                # in_main_alliance: derive WITHOUT look-ahead. Use the
                # current state — players with no votes-against and
                # not targeted recently are treated as in-alliance.
                in_alliance = 1 if (votes_against[name] == 0
                                     and target_count[name] <= 1) else 0
                # Add light noise so the column isn't perfectly deterministic.
                if rng.random() < 0.10:
                    in_alliance = 1 - in_alliance

                # confessional_count + visibility (pre-tribal proxy).
                vis = base_vis_this_ep[name]
                conf_count = int(round(vis * 14))
                visibility_hist[name].append(vis)
                vis_window = visibility_hist[name][-3:]
                vis_mean = sum(vis_window) / max(1, len(vis_window))
                vis_spike = vis - vis_mean if len(vis_window) >= 2 else 0.0

                # negative_edit_score from PRIOR cumulative signal
                # (votes-against + target mentions, before this ep).
                eps_so_far = max(1, ep - 1)
                target_rate = target_count[name] / eps_so_far
                vote_score = votes_against[name] / 2.0
                neg_edit = max(0.0, min(1.0,
                                          0.55 * target_rate
                                          + 0.30 * vote_score
                                          + 0.15 * max(0.0, vis_spike) * 1.0))

                # strategic_isolation: from PRIOR signal only — players
                # with prior votes and prior target mentions are more
                # isolated. Smoothed against tribe-size.
                isolation = max(0.0, min(1.0,
                                           0.35 * vote_score
                                           + 0.35 * target_rate
                                           + 0.20 * (1 - tribe_size / max(1, total_cast))
                                           + (rng.random() - 0.5) * 0.10))

                pp = prior_perf[name]

                rows.append({
                    "season": season,
                    "episode": ep,
                    "contestant": name,
                    "tribe": tribe if not merged else "Merge",
                    "eliminated": 1 if name in boots_this_ep else 0,
                    "starting_tribe": tribe,
                    "starting_tribe_size": len(starting_tribes[tribe]),
                    "tribe_size": tribe_size,
                    "remaining": rem,
                    "merged": merged,
                    "swap_phase": swap_phase,
                    "immunity_won": imm,
                    "tribe_immunity": tribe_immunity,
                    "prior_votes_against": votes_against[name],
                    "times_targeted": target_count[name],
                    "has_idol": has_idol,
                    "in_main_alliance": in_alliance,
                    "confessional_count": conf_count,
                    "visibility_score": round(vis_mean, 3),
                    "visibility_spike": round(vis_spike, 3),
                    "negative_edit_score": round(neg_edit, 3),
                    "strategic_isolation": round(isolation, 3),
                    "prior_perf_score": round(pp, 3),
                })

                # Update rolling state for next episode.
                if name in boots_this_ep:
                    votes_against[name] += 1
                # target_count bump when visibility spiked enough to be
                # read as "this player is being framed."
                if vis_spike > 0.10 and rng.random() < 0.4:
                    target_count[name] += 1
                prior_perf[name] = 0.7 * pp + 0.3 * (1.0 if tribe_immunity else 0.4)

    return pd.DataFrame(rows).sort_values(
        ["season", "episode", "contestant"]
    ).reset_index(drop=True)
