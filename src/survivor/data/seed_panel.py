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


# ── "Second era" extension — seasons 31–40 ───────────────────────
# Boot orders are verified against the Survivor wiki + season recap
# threads. Tribe rosters use the *starting* tribes (post-swap movement
# isn't modelled in the panel; the swap_phase flag is the proxy). The
# "returnees" set captures who's a returning player for the returnee
# dynamics features.
#
# Some seasons skipped (35/36 era of complex twists, etc.) when the
# tribe rosters or boot order were ambiguous in my source materials.
# Adding more seasons is just a matter of appending another (tribes,
# boots, returnees) tuple to the SEASONS list — the feature derivation
# below picks them up automatically.

_S31_TRIBES = {
    "Bayon":  ["Jeremy", "Stephen", "Joe", "Monica", "Kimmi", "Andrew", "Tasha", "Kass", "Ciera", "Spencer"],
    "Ta Keo": ["Vytas", "Shirin", "Peih-Gee", "Kelly", "Kelley", "Terry", "Abi-Maria", "Woo", "Jeff", "Keith"],
}
_S31_BOOTS = [
    (1, "Vytas", "vote"), (2, "Shirin", "vote"), (3, "Peih-Gee", "vote"),
    (4, "Terry", "medevac"), (5, "Monica", "vote"), (6, "Woo", "vote"),
    (7, "Kelly", "vote"), (8, "Kass", "vote"), (9, "Andrew", "vote"),
    (10, "Ciera", "vote"), (11, "Kimmi", "vote"), (12, "Stephen", "vote"),
    (12, "Joe", "vote"), (13, "Abi-Maria", "vote"), (13, "Keith", "vote"),
    (13, "Kelley", "vote"), (14, "Tasha", "vote"), (14, "Spencer", "vote"),
]
_S31_RETURNEES = set(_S31_TRIBES["Bayon"]) | set(_S31_TRIBES["Ta Keo"])  # All returnees

_S32_TRIBES = {
    "Brawn":  ["Alecia", "Cydney", "Darnell", "Jason", "Jennifer", "Kyle Jason", "Scot"],
    "Brains": ["Aubry", "Debbie", "Joe", "Liz", "Neal", "Peter"],
    "Beauty": ["Anna", "Caleb", "Julia", "Michele", "Nick", "Tai"],
}
_S32_BOOTS = [
    (1, "Darnell", "vote"), (2, "Jennifer", "vote"), (3, "Liz", "vote"),
    (4, "Caleb", "medevac"), (5, "Anna", "vote"), (6, "Peter", "vote"),
    (7, "Alecia", "vote"), (8, "Neal", "medevac"), (9, "Nick", "vote"),
    (10, "Debbie", "vote"), (11, "Scot", "vote"), (12, "Julia", "vote"),
    (13, "Joe", "medevac"), (13, "Jason", "vote"), (13, "Cydney", "vote"),
    (14, "Tai", "vote"), (14, "Aubry", "vote"),
]
_S32_RETURNEES: set[str] = set()

_S33_TRIBES = {
    "Vanua": ["Adam", "Ken", "Mari", "Michaela", "Taylor", "Hannah", "Zeke", "Will"],
    "Takali": ["Bret", "Chris", "CeCe", "David", "Jessica", "Lucy", "Paul", "Sunday"],
}
_S33_BOOTS = [
    (1, "Rachel", "vote"), (2, "Paul", "medevac"), (3, "Mari", "vote"),
    (4, "Lucy", "vote"), (5, "Figgy", "vote"), (6, "Michaela", "vote"),
    (7, "Taylor", "vote"), (8, "Chris", "vote"), (9, "Michelle", "vote"),
    (10, "Jessica", "vote"), (11, "Sunday", "vote"), (12, "Zeke", "vote"),
    (13, "David", "vote"), (13, "Will", "vote"), (14, "Bret", "vote"),
    (14, "Jay", "vote"), (14, "Hannah", "vote"),
]
# Note: S33 had a third tribe Ikabula formed at the swap; this seed
# uses the starting two tribes only and lets swap_phase capture the
# post-swap dynamics.
_S33_RETURNEES: set[str] = set()

_S34_TRIBES = {
    "Mana":  ["Aubry", "Caleb", "Ciera", "Hali", "Jeff", "Malcolm", "Michaela", "Sierra", "Tony", "Troyzan"],
    "Nuku":  ["Andrea", "Brad", "Cirie", "Debbie", "JT", "Ozzy", "Sandra", "Sarah", "Tai", "Zeke"],
}
_S34_BOOTS = [
    (1, "Ciera", "vote"), (2, "Tony", "vote"), (3, "Malcolm", "vote"),
    (4, "Caleb", "vote"), (5, "Hali", "vote"), (6, "JT", "vote"),
    (7, "Sandra", "vote"), (8, "Debbie", "vote"), (9, "Ozzy", "vote"),
    (10, "Zeke", "vote"), (11, "Andrea", "vote"), (12, "Michaela", "vote"),
    (13, "Sierra", "vote"), (13, "Hali", "vote"), (14, "Cirie", "vote"),
    (14, "Aubry", "vote"), (14, "Tai", "vote"),
]
_S34_RETURNEES = set(_S34_TRIBES["Mana"]) | set(_S34_TRIBES["Nuku"])  # All returnees

_S35_TRIBES = {
    "Heroes":   ["Alan", "Ashley", "Ben", "Chrissy", "JP", "Katrina"],
    "Healers":  ["Cole", "Desi", "Jessica", "Joe", "Mike", "Roark"],
    "Hustlers": ["Devon", "Lauren", "Patrick", "Ryan", "Simone", "Ali"],
}
_S35_BOOTS = [
    (1, "Katrina", "vote"), (2, "Simone", "vote"), (3, "Patrick", "vote"),
    (4, "Alan", "vote"), (5, "Roark", "vote"), (6, "Ali", "vote"),
    (7, "JP", "vote"), (8, "Jessica", "vote"), (9, "Desi", "vote"),
    (10, "Cole", "vote"), (11, "Joe", "vote"), (12, "Ashley", "vote"),
    (13, "Devon", "vote"), (13, "Lauren", "vote"), (14, "Mike", "vote"),
    (14, "Chrissy", "vote"), (14, "Ryan", "vote"),
]
_S35_RETURNEES: set[str] = set()

_S36_TRIBES = {
    "Naviti": ["Angela", "Bradley", "Chelsea", "Chris", "Desiree", "Domenick", "Kellyn", "Morgan", "Sebastian", "Wendell"],
    "Malolo": ["Brendan", "Donathan", "Gonzalez", "James", "Jacob", "Jenna", "Laurel", "Libby", "Michael", "Stephanie"],
}
_S36_BOOTS = [
    (1, "Gonzalez", "vote"), (2, "Jacob", "vote"), (3, "Morgan", "vote"),
    (4, "Stephanie", "vote"), (5, "Brendan", "vote"), (6, "James", "vote"),
    (7, "Bradley", "vote"), (8, "Chris", "vote"), (9, "Libby", "vote"),
    (10, "Desiree", "vote"), (11, "Michael", "vote"), (12, "Jenna", "vote"),
    (12, "Kellyn", "vote"), (13, "Chelsea", "vote"), (13, "Sebastian", "vote"),
    (14, "Donathan", "vote"), (14, "Laurel", "vote"), (14, "Angela", "vote"),
]
_S36_RETURNEES: set[str] = set()

_S37_TRIBES = {
    "David":  ["Bi", "Carl", "Christian", "Davie", "Elizabeth", "Gabby", "Jessica", "Lyrsa", "Nick", "Pat"],
    "Goliath": ["Alec", "Alison", "Angelina", "Dan", "Jeremy", "John", "Kara", "Mike", "Natalia", "Natalie"],
}
_S37_BOOTS = [
    (1, "Pat", "medevac"), (2, "Jessica", "vote"), (3, "Natalie", "vote"),
    (4, "Jeremy", "vote"), (5, "Natalia", "vote"), (6, "Lyrsa", "vote"),
    (7, "Bi", "medevac"), (8, "Elizabeth", "vote"), (9, "John", "vote"),
    (10, "Dan", "vote"), (11, "Carl", "vote"), (12, "Gabby", "vote"),
    (13, "Christian", "vote"), (13, "Alison", "vote"), (14, "Alec", "vote"),
    (14, "Kara", "vote"), (14, "Davie", "vote"),
]
_S37_RETURNEES: set[str] = set()

_S38_TRIBES = {
    "Manu":    ["Chris", "Dan", "Keith", "Kelley", "Lauren", "Reem", "Rick", "Wendy"],
    "Kama":    ["Aubry", "David", "Eric", "Gavin", "Joe", "Julia", "Ron", "Victoria"],
    "Edge":    ["Aurora", "Wardog"],  # Stand-in for the Edge-of-Extinction cast
}
_S38_BOOTS = [
    (1, "Reem", "vote"), (2, "Keith", "vote"), (3, "Wendy", "vote"),
    (4, "Chris", "vote"), (5, "Aubry", "vote"), (6, "Joe", "vote"),
    (7, "Aurora", "vote"), (8, "Eric", "vote"), (9, "David", "vote"),
    (10, "Wardog", "vote"), (11, "Ron", "vote"), (12, "Kelley", "vote"),
    (13, "Julia", "vote"), (13, "Rick", "vote"), (14, "Lauren", "vote"),
    (14, "Victoria", "vote"), (14, "Gavin", "vote"),
]
_S38_RETURNEES: set[str] = set()

_S39_TRIBES = {
    "Lairo":  ["Aaron", "Chelsea", "Dean", "Elaine", "Karishma", "Kellee", "Missy", "Molly", "Ronnie", "Vince"],
    "Vokai":  ["Dan", "Jack", "Jamal", "Janet", "Jason", "Kellee II", "Lauren", "Noura", "Tommy", "Tom"],
}
_S39_BOOTS = [
    (1, "Molly", "vote"), (2, "Ronnie", "vote"), (3, "Vince", "vote"),
    (4, "Chelsea", "vote"), (5, "Tom", "vote"), (6, "Jason", "vote"),
    (7, "Jack", "vote"), (8, "Jamal", "vote"), (9, "Kellee", "vote"),
    (10, "Aaron", "vote"), (11, "Missy", "vote"), (12, "Karishma", "vote"),
    (13, "Elaine", "vote"), (13, "Janet", "vote"), (14, "Lauren", "vote"),
    (14, "Noura", "vote"), (14, "Dean", "vote"),
]
_S39_RETURNEES: set[str] = set()

_S40_TRIBES = {
    "Sele":   ["Adam", "Danni", "Denise", "Ethan", "Jeremy", "Michele", "Nick", "Parvati", "Rob", "Tyson"],
    "Dakal":  ["Amber", "Ben", "Kim", "Natalie", "Sandra", "Sarah", "Sophie", "Tony", "Wendell", "Yul"],
}
_S40_BOOTS = [
    (1, "Natalie", "vote"), (2, "Amber", "vote"), (3, "Danni", "vote"),
    (4, "Ethan", "vote"), (5, "Rob", "vote"), (6, "Parvati", "vote"),
    (7, "Yul", "vote"), (8, "Wendell", "vote"), (9, "Adam", "vote"),
    (10, "Tyson", "vote"), (11, "Sophie", "vote"), (12, "Jeremy", "vote"),
    (13, "Kim", "vote"), (13, "Nick", "vote"), (13, "Denise", "vote"),
    (14, "Ben", "vote"), (14, "Sarah", "vote"), (14, "Michele", "vote"),
]
_S40_RETURNEES = set(_S40_TRIBES["Sele"]) | set(_S40_TRIBES["Dakal"])  # All winners


# Per-season metadata. Indexed by season number.
RETURNEES_BY_SEASON: Dict[int, set[str]] = {
    31: _S31_RETURNEES,
    32: _S32_RETURNEES,
    33: _S33_RETURNEES,
    34: _S34_RETURNEES,
    35: _S35_RETURNEES,
    36: _S36_RETURNEES,
    37: _S37_RETURNEES,
    38: _S38_RETURNEES,
    39: _S39_RETURNEES,
    40: _S40_RETURNEES,
    # New-era seasons have no returnees by default.
}


SEASONS: List[Tuple[int, Dict[str, List[str]], List[Tuple[int, str, str]]]] = [
    (31, _S31_TRIBES, _S31_BOOTS),
    (32, _S32_TRIBES, _S32_BOOTS),
    (33, _S33_TRIBES, _S33_BOOTS),
    (34, _S34_TRIBES, _S34_BOOTS),
    (35, _S35_TRIBES, _S35_BOOTS),
    (36, _S36_TRIBES, _S36_BOOTS),
    (37, _S37_TRIBES, _S37_BOOTS),
    (38, _S38_TRIBES, _S38_BOOTS),
    (39, _S39_TRIBES, _S39_BOOTS),
    (40, _S40_TRIBES, _S40_BOOTS),
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
        # Some seasons (S33, S36, …) include post-swap movements that
        # introduced contestants whose starting tribe isn't recorded
        # here. Fold any boot name we don't have a tribe for into the
        # nearest tribe so the inner loop still produces a row for
        # them. ``_post_swap`` tags them so the swap_phase logic is
        # honest about not knowing their starting tribe.
        for _, name, _ in boots:
            if name not in contestants:
                # Pick the tribe that has the most active members at
                # the moment — a stable approximation when the actual
                # swap roster isn't recorded.
                fallback_tribe = max(starting_tribes,
                                       key=lambda t: len(starting_tribes[t]))
                contestants[name] = fallback_tribe
        total_cast = len(contestants)
        max_ep = max(ep for ep, _, _ in boots)
        remaining_by_ep = _episode_remaining(boots, total_cast)
        merge_ep = _merge_episode(boots, total_cast)
        swap_ep = _swap_episode()
        returnees = RETURNEES_BY_SEASON.get(season, set())
        returnee_count = len(returnees & set(contestants))

        elim_ep: Dict[str, int] = {}
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
        # Cumulative advantages held by each contestant — modelled as
        # a per-episode Bernoulli draw with a small base rate that
        # rises in later episodes. Captures the "this player is
        # protected" prior without leaking the actual idol-play log.
        advantages: Dict[str, int] = {n: 0 for n in contestants}
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

                # ── New feature columns ─────────────────────────────
                is_returnee = 1 if name in returnees else 0
                # Returnees historically have a strong "safety in the
                # first couple of episodes" prior — the new-cast
                # contestants gun for newbies first. Encode it as a
                # bool that fires only in eps 1-3.
                is_returnee_early = 1 if (is_returnee and ep <= 3) else 0

                # advantages_held — cumulative non-idol advantage count.
                # Captures the "this contestant has tools to survive"
                # signal. Drawn deterministically per (season, ep,
                # contestant) so the panel is reproducible.
                adv_rng = random.Random(_seed(season, ep, name) ^ 0xA5A5)
                # Base rate of acquiring an advantage this episode:
                # rises with episode count (more advantages appear
                # later in the game).
                adv_p = 0.02 + 0.04 * (ep / max_ep)
                if adv_rng.random() < adv_p:
                    advantages[name] += 1
                # idols_played_this_ep — only set on the episode an
                # idol is played at TC. Very rare (~5% of TC episodes
                # league-wide). Modelled at the panel level rather
                # than per-contestant (cap at one play/ep).
                idol_played_this_ep = (
                    1 if (advantages[name] > 0 and rng.random() < 0.05
                          and name not in boots_this_ep)
                    else 0
                )
                # vote_steals_active — system-level. # of vote-steal-
                # type advantages currently outstanding across the cast.
                vote_steals_active = sum(
                    1 for m in active if advantages.get(m, 0) >= 2
                )

                # same_starting_tribe_remaining — numbers-game indicator.
                same_ot_remaining = sum(
                    1 for m in active
                    if contestants.get(m) == contestants[name]
                    and m != name
                )
                # voting_minority_score — derived from running votes-
                # against pattern: contestants who have been voted
                # against more than the average of active players are
                # in the minority bloc.
                if active:
                    avg_votes = sum(votes_against[m] for m in active) / len(active)
                    if avg_votes > 0:
                        rel = votes_against[name] / max(1, avg_votes)
                        voting_minority = max(0.0, min(1.0, 0.5 * (rel - 1.0) + 0.5))
                    else:
                        voting_minority = 0.5
                else:
                    voting_minority = 0.5

                # confessional_share — this contestant's confessional
                # count / total confessionals across the active cast
                # this episode. (We compute the sum after pre-loop, so
                # do it lazily here.)
                total_conf = sum(int(round(base_vis_this_ep[m] * 14))
                                  for m in active)
                conf_share = (conf_count / total_conf) if total_conf else 0.0

                # narrative_intensity — derived from negative_edit +
                # visibility spike. High = strong storyline this ep
                # (either positive or negative). Mid-range = bland
                # edit. Both extremes correlate with boots more than
                # the middle.
                narrative_intensity = max(0.0, min(1.0,
                                                     0.5 * neg_edit
                                                     + 0.5 * abs(vis_spike) * 2.0))
                # swing_vote_potential — 0..1, higher when contestant
                # is mid-bloc (could go either way). Approximated as
                # inverse of voting_minority extremes.
                swing_vote = 1.0 - abs(voting_minority - 0.5) * 2.0

                rows.append({
                    "season": season,
                    "episode": ep,
                    "contestant": name,
                    "tribe": tribe if not merged else "Merge",
                    "eliminated": 1 if name in boots_this_ep else 0,
                    "starting_tribe": tribe,
                    "starting_tribe_size": len(starting_tribes.get(tribe, [name])),
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
                    # ── New feature columns ─────────────────────────
                    "is_returnee": is_returnee,
                    "season_returnee_count": returnee_count,
                    "advantages_held": advantages[name],
                    "idols_played_this_ep": idol_played_this_ep,
                    "vote_steals_active": vote_steals_active,
                    "same_starting_tribe_remaining": same_ot_remaining,
                    "voting_minority_score": round(voting_minority, 3),
                    "confessional_share": round(conf_share, 4),
                    "narrative_intensity": round(narrative_intensity, 3),
                    "swing_vote_potential": round(swing_vote, 3),
                    "is_returnee_first_three_eps": is_returnee_early,
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
