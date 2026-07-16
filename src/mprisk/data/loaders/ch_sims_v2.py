"""CH-SIMS v2 loader placeholder."""

from __future__ import annotations

from mprisk.data.loaders.base import DatasetLoader


class ChSimsV2Loader(DatasetLoader):
    def load(self) -> list[dict[str, object]]:
        return []
