from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mprisk.assets.registry import (
    load_model_assets,
    validate_model_panel_references,
    validate_model_reference_config,
)

ASSET_PATH = Path("configs/assets/model_assets.yaml")
REFERENCE_PATHS = (
    Path("configs/protocols/vt.yaml"),
    Path("configs/protocols/va.yaml"),
    Path("configs/experiments/main_vt.yaml"),
    Path("configs/experiments/main_va.yaml"),
)


def test_protocol_and_main_experiment_references_are_consistent() -> None:
    assets = load_model_assets(ASSET_PATH)
    references = validate_model_panel_references(assets, REFERENCE_PATHS)

    assert references[Path("configs/protocols/vt.yaml")] == references[
        Path("configs/experiments/main_vt.yaml")
    ]
    assert references[Path("configs/protocols/va.yaml")] == references[
        Path("configs/experiments/main_va.yaml")
    ]
    assert len(references[Path("configs/protocols/vt.yaml")]) == 13
    assert len(references[Path("configs/protocols/va.yaml")]) == 3


def test_unknown_and_protocol_incompatible_references_fail(tmp_path: Path) -> None:
    assets = load_model_assets(ASSET_PATH)
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump({"protocol": "vt", "models": ["unknown_model"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown model keys"):
        validate_model_reference_config(assets, path)

    path.write_text(
        yaml.safe_dump({"protocol": "vt", "models": ["qwen2_5_omni_7b"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible models"):
        validate_model_reference_config(assets, path)
