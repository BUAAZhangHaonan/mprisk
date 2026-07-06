from __future__ import annotations

from pathlib import Path

from mprisk.assets.registry import load_model_assets


def test_load_model_assets() -> None:
    assets = load_model_assets(Path("configs/assets/model_assets.yaml"))
    assert {asset.key for asset in assets} >= {"qwen2_5_omni_7b", "qwen3_vl_8b"}
