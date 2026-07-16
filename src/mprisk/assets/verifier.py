"""Asset validation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetCheck:
    key: str
    exists: bool
    message: str


def verify_path(key: str, path: str | Path) -> AssetCheck:
    candidate = Path(path)
    exists = candidate.exists()
    message = "found" if exists else f"missing: {candidate}"
    return AssetCheck(key=key, exists=exists, message=message)
