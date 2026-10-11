"""Small YAML configuration helpers without a framework dependency."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML and apply ``section.key=value`` overrides."""
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise KeyError(f"Unknown configuration path: {dotted_key}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise KeyError(f"Unknown configuration key: {dotted_key}")
        cursor[parts[-1]] = value
    return cfg
