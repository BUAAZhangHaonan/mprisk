"""CMU-MOSEI loader placeholder."""

from __future__ import annotations

from mprisk.data.loaders.base import DatasetLoader


class CmuMoseiLoader(DatasetLoader):
    def load(self) -> list[dict[str, object]]:
        return []
