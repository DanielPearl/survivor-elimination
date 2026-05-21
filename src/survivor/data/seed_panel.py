"""DEPRECATED — synthetic seed panel, no longer in use.

This module previously generated a hand-curated panel for seasons 41–49
with REAL boot labels but SYNTHETIC per-row features (confessional
counts, visibility, negative-edit score, strategic isolation derived
from RNG rules).

It has been replaced by ``survivor_archive.py`` which sources every
column from the ``doehm/survivoR`` GitHub archive — real confessionals,
real vote history, real advantage movements, real challenge results
for every US season.

The module is retained only so historical artifacts that imported it
still parse. Calling ``build()`` raises immediately. Delete this file
once a release has been cut without any external importers.
"""
from __future__ import annotations

import pandas as pd


def build() -> pd.DataFrame:
    raise RuntimeError(
        "seed_panel.build() is deprecated — the synthetic seed has been "
        "removed. Use scripts/build_historical_panel.py to materialise "
        "the real panel from the doehm/survivoR archive."
    )
