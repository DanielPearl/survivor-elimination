"""Live-monitor loop.

Polls Kalshi every `poll_interval_seconds`, rebuilds the watchlist,
writes the JSON, sleeps. The dashboard reads watchlist.json on each
page load, so updates appear without restarting the server.

Run via systemd (deploy/survivor-elimination.service).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.survivor.dashboard.export_watchlist import export
from src.survivor.utils.config import load_config
from src.survivor.utils.logging_setup import setup_logging

log = setup_logging("scripts.live_monitor")


def main() -> int:
    cfg = load_config()
    poll = int(cfg["kalshi"].get("poll_interval_seconds", 300))
    log.info("starting live monitor with %ds poll interval", poll)
    while True:
        try:
            _, json_path = export()
            log.info("wrote %s", json_path)
        except Exception as exc:  # noqa: BLE001
            log.exception("monitor tick failed: %s", exc)
        time.sleep(poll)


if __name__ == "__main__":
    raise SystemExit(main())
