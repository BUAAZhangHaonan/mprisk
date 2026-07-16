"""Base dataset loader contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLoader:
    dataset_key: str
    root: Path

    def load(self) -> list[dict[str, object]]:
        raise NotImplementedError
