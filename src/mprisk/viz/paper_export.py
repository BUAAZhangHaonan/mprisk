"""Shared paper export contracts."""

from __future__ import annotations

from pathlib import Path


def ensure_export_dir(path: str | Path) -> Path:
    export_dir = Path(path)
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir
