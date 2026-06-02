"""Shared project utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_taxonomy(filepath: str | Path, config_module: Any) -> dict:
    """Load taxonomy JSON and merge blacklist settings into the config module."""

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file cau hinh tai: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    blacklist = data.get("blacklist", [])
    if blacklist:
        config_module.BLACKLIST_KEYWORDS = list(set(config_module.BLACKLIST_KEYWORDS + blacklist))

    title_prefixes = data.get("title_blacklist_prefixes", [])
    if title_prefixes:
        config_module.STUB_PREFIXES = list(set(config_module.STUB_PREFIXES + title_prefixes))

    return data.get("domains", {})
