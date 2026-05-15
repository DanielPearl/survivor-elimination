"""Daily train entrypoint. Re-train the model and rebuild watchlist.

Usage:
    python scripts/run_daily_train.py
    python scripts/run_daily_train.py --skip-train       # just rebuild watchlist
    python scripts/run_daily_train.py --offline          # build watchlist with no Kalshi fetch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.survivor.dashboard.export_watchlist import export
from src.survivor.models.predict import reset_model_cache
from src.survivor.models.train import train_and_persist
from src.survivor.utils.logging_setup import setup_logging

log = setup_logging("scripts.daily_train")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--offline", action="store_true",
                   help="Skip the live Kalshi fetch — build a watchlist "
                        "from local state only.")
    args = p.parse_args()

    if not args.skip_train:
        log.info("training elimination model…")
        metrics = train_and_persist()
        log.info("done. blended brier=%.4f acc=%.3f",
                  metrics["blended"]["brier"], metrics["blended"]["accuracy"])
        reset_model_cache()

    if args.offline:
        log.info("offline mode — building watchlist with no Kalshi records")
        csv_path, json_path = export(kalshi_records=[])
    else:
        log.info("building live watchlist…")
        csv_path, json_path = export()
    log.info("watchlist ready: %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
