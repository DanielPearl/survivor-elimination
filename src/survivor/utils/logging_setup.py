"""Stdlib logging setup. Same one-liner the other bots use."""
from __future__ import annotations

import logging
import sys


def setup_logging(name: str = "survivor", level: int = logging.INFO
                   ) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(h)
    log.propagate = False
    return log
