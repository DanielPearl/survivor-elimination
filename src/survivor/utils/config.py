"""Config loader. Same idiom as the tennis bot — read one YAML, resolve
paths relative to the repo root, hand the resulting dict back."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Repo root = three parents above this file (utils → survivor → src → repo).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(rel: str | Path) -> Path:
    """Return an absolute Path for any of the repo's config paths."""
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p)
