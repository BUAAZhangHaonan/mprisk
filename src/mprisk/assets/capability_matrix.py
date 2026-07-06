"""Build small model capability tables."""

from __future__ import annotations

from mprisk.assets.registry import ModelAsset


def build_capability_rows(assets: list[ModelAsset]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for asset in assets:
        for protocol in asset.protocols:
            rows.append(
                {
                    "model_key": asset.key,
                    "family": asset.family,
                    "protocol": protocol,
                    "status": asset.status,
                }
            )
    return rows
