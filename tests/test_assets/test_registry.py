from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from mprisk.assets.registry import (
    MODEL_ASSET_SCHEMA,
    index_assets,
    load_model_assets,
)
from mprisk.config.loader import load_yaml

ASSET_PATH = Path("configs/assets/model_assets.yaml")

VT_KEYS = (
    "gemma3_4b",
    "gemma3_12b",
    "glm4_6v_flash",
    "internvl3_5_8b",
    "llava_v1_5_7b",
    "llava_onevision_qwen2_7b",
    "minicpm_v_2_6",
    "minicpm_v_4_5",
    "phi3_5_vision",
    "qwen2_5_vl_7b",
    "qwen3_vl_8b",
    "qwen3_5_4b",
    "qwen3_5_9b",
)
VA_VTA_KEYS = ("gemma4_12b", "phi4_multimodal", "qwen2_5_omni_7b")

EXPECTED_HF_IDS = {
    "gemma3_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "glm4_6v_flash": "zai-org/GLM-4.6V-Flash",
    "internvl3_5_8b": "OpenGVLab/InternVL3_5-8B",
    "llava_v1_5_7b": "llava-hf/llava-1.5-7b-hf",
    "llava_onevision_qwen2_7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "minicpm_v_2_6": "openbmb/MiniCPM-V-2_6",
    "minicpm_v_4_5": "openbmb/MiniCPM-V-4_5",
    "phi3_5_vision": "microsoft/Phi-3.5-vision-instruct",
    "qwen2_5_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
    "qwen3_5_9b": "Qwen/Qwen3.5-9B",
    "gemma4_12b": "google/gemma-4-12B-it",
    "phi4_multimodal": "microsoft/Phi-4-multimodal-instruct",
    "qwen2_5_omni_7b": "Qwen/Qwen2.5-Omni-7B",
}


def _write_registry(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "model_assets.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_frozen_model_panel_exact_order_and_official_ids() -> None:
    assets = load_model_assets(ASSET_PATH)
    assert tuple(asset.key for asset in assets) == VT_KEYS + VA_VTA_KEYS
    assert {asset.key: asset.hf_model_id for asset in assets} == EXPECTED_HF_IDS
    assert len(assets) == 16
    assert all("molmo" not in asset.key.lower() for asset in assets)
    assert all("molmo" not in asset.hf_model_id.lower() for asset in assets)


def test_panel_groups_protocols_and_thinking_are_frozen() -> None:
    assets = load_model_assets(ASSET_PATH)
    by_key = index_assets(assets)
    assert tuple(asset.key for asset in assets if asset.panel_group == "vt") == VT_KEYS
    assert tuple(asset.key for asset in assets if asset.panel_group == "va_vta") == VA_VTA_KEYS
    assert all(by_key[key].protocols == ("vt",) for key in VT_KEYS)
    assert all(by_key[key].protocols == ("va",) for key in VA_VTA_KEYS)
    assert all(not asset.thinking.enabled for asset in assets)
    assert all(not asset.policy.allow_thinking for asset in assets)
    assert all(
        asset.thinking.disable_argument == "enable_thinking=false"
        for asset in assets
        if asset.thinking.supported
    )


def test_qwen3_instruct_omni_glm_and_phi4_boundaries() -> None:
    by_key = index_assets(load_model_assets(ASSET_PATH))

    qwen3 = by_key["qwen3_vl_8b"]
    assert qwen3.hf_model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert qwen3.local_path.endswith("/Qwen3-VL-8B-Instruct")
    assert "Thinking" not in qwen3.hf_model_id

    omni = by_key["qwen2_5_omni_7b"]
    assert omni.hf_model_id == "Qwen/Qwen2.5-Omni-7B"
    assert omni.local_path == "/home/team/lvshuyang/Models/Qwen/Qwen2.5-Omni-7B"
    assert omni.input_modalities == ("text", "image", "video", "audio")

    glm = by_key["glm4_6v_flash"]
    assert glm.display_name == "GLM-4.6V-Flash"
    assert glm.parameter_scale == "~9B"
    assert glm.video_mode == "native_video"

    phi4 = by_key["phi4_multimodal"]
    assert phi4.video_mode == "extracted_frames"
    assert phi4.max_video_frames == 64
    assert "video" not in phi4.input_modalities


def test_all_configured_local_model_paths_resolve() -> None:
    assets = load_model_assets(ASSET_PATH, require_local_paths=True)
    assert all(asset.local_model_path.is_absolute() for asset in assets)


def test_schema_is_versioned_and_rejects_extra_fields(tmp_path: Path) -> None:
    payload = load_yaml(ASSET_PATH)
    assert payload["schema"] == MODEL_ASSET_SCHEMA
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        load_model_assets(_write_registry(tmp_path, payload))


def test_duplicate_model_key_is_rejected(tmp_path: Path) -> None:
    payload = load_yaml(ASSET_PATH)
    payload["models"].append(deepcopy(payload["models"][0]))
    with pytest.raises(ValueError, match="Duplicate model asset key"):
        load_model_assets(_write_registry(tmp_path, payload))


def test_video_mode_and_thinking_policy_enums_are_strict(tmp_path: Path) -> None:
    payload = load_yaml(ASSET_PATH)
    payload["models"][0]["video_mode"] = "heuristic_video"
    with pytest.raises(ValueError, match="video_mode must be one of"):
        load_model_assets(_write_registry(tmp_path, payload))

    payload = load_yaml(ASSET_PATH)
    payload["models"][0]["thinking"]["enabled"] = True
    payload["models"][0]["policy"]["allow_thinking"] = True
    with pytest.raises(ValueError, match="enabled must be false"):
        load_model_assets(_write_registry(tmp_path, payload))
