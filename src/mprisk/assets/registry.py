"""Load and query model asset metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.config.loader import load_yaml


@dataclass(frozen=True)
class ModelAsset:
    key: str
    display_name: str
    family: str
    source: str
    protocols: tuple[str, ...]
    status: str


def load_model_assets(path: str | Path) -> list[ModelAsset]:
    data = load_yaml(path)
    return [
        ModelAsset(
            key=item["key"],
            display_name=item.get("display_name", item["key"]),
            family=item["family"],
            source=item.get("source", ""),
            protocols=tuple(item.get("protocols", [])),
            status=item.get("status", "unknown"),
        )
        for item in data.get("models", [])
    ]


def index_assets(assets: list[ModelAsset]) -> dict[str, ModelAsset]:
    return {asset.key: asset for asset in assets}


def assets_to_rows(assets: list[ModelAsset]) -> list[dict[str, Any]]:
    return [
        {
            "model_key": asset.key,
            "family": asset.family,
            "source": asset.source,
            "protocols": ",".join(asset.protocols),
            "status": asset.status,
        }
        for asset in assets
    ]
