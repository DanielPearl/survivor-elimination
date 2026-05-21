"""Materialise the historical training panel from the doehm/survivoR
GitHub archive.

Usage::

    # Build from cached JSON snapshots under data/raw/survivor_archive/.
    python scripts/build_historical_panel.py

    # Re-download the upstream snapshots first (use after a new season
    # has been added upstream).
    python scripts/build_historical_panel.py --refresh

Writes ``data/raw/historical_boots.csv`` with the column schema in
``survivor.data.historical.HISTORICAL_COLUMNS``. After this runs the
trainer (``python scripts/run_daily_train.py``) can fit the model on
real per-episode survivor data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/build_historical_panel.py` to import survivor.*
# without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from survivor.data.historical import (  # noqa: E402
    HISTORICAL_COLUMNS, write_historical_panel,
)
from survivor.data.survivor_archive import build_historical_panel  # noqa: E402
from survivor.utils.config import load_config, resolve_path  # noqa: E402
from survivor.utils.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="re-download archive JSON files before building")
    args = parser.parse_args()

    setup_logging("data.build_historical_panel")
    cfg = load_config()
    out_path = resolve_path(cfg["paths"]["historical_csv"])

    df = build_historical_panel(refresh=args.refresh)
    # Project to the canonical column order — write_historical_panel
    # already does this but being explicit makes the script
    # self-documenting.
    df = df[HISTORICAL_COLUMNS]
    write_historical_panel(df, out_path)
    print(f"wrote {out_path} ({len(df)} rows, "
          f"{df['season'].nunique()} seasons, "
          f"{int(df['eliminated'].sum())} boots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
