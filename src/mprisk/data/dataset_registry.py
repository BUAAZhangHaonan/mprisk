"""Dataset registry types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    role: str
    modalities: tuple[str, ...]
