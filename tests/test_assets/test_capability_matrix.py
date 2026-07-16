from __future__ import annotations

from pathlib import Path

from mprisk.assets.capability_matrix import build_capability_rows
from mprisk.assets.registry import load_model_assets


def test_capability_matrix_preserves_native_and_frame_video_boundaries() -> None:
    assets = load_model_assets(Path("configs/assets/model_assets.yaml"))
    rows = build_capability_rows(assets)
    by_key = {row["model_key"]: row for row in rows}

    assert len(rows) == 16
    assert by_key["glm4_6v_flash"]["native_video"] is True
    assert by_key["qwen2_5_omni_7b"]["native_video"] is True
    assert by_key["phi4_multimodal"]["native_video"] is False
    assert by_key["phi4_multimodal"]["requires_frame_extraction"] is True
    assert by_key["phi4_multimodal"]["max_video_frames"] == 64
    assert all(row["thinking_enabled"] is False for row in rows)
    assert all(row["allow_thinking"] is False for row in rows)
