"""CMU-MOSI loader placeholder."""

from __future__ import annotations

from mprisk.data.loaders.base import DatasetLoader


class CmuMosiLoader(DatasetLoader):
    def load(self) -> list[dict[str, object]]:
        return []
