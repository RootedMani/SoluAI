"""Load the YAML configuration file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from . import ROOT


def load_config(path: str | Path = ROOT / "config.yaml") -> dict[str, Any]:
    """Read the configuration file and return it as a dictionary."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_hash(cfg: dict[str, Any]) -> str:
    """Short fingerprint of the config, stored with the model for versioning."""
    return hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()[:12]
