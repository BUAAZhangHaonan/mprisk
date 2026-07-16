"""Build one explicit capability row per model in the experiment panel."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mprisk.assets.registry import ModelAsset


def build_capability_rows(assets: Iterable[ModelAsset]) -> list[dict[str, Any]]:
    return [
        {
            "model_key": asset.key,
            "display_name": asset.display_name,
            "family": asset.family,
            "parameter_scale": asset.parameter_scale,
            "panel_group": asset.panel_group,
            "protocols": ",".join(asset.protocols),
            "input_modalities": ",".join(asset.input_modalities),
            "video_mode": asset.video_mode,
            "native_video": asset.video_mode
            in {"native_video", "native_video_or_multi_image"},
            "requires_frame_extraction": asset.video_mode == "extracted_frames",
            "max_video_frames": asset.max_video_frames,
            "thinking_supported": asset.thinking.supported,
            "thinking_enabled": asset.thinking.enabled,
            "allow_thinking": asset.policy.allow_thinking,
            "status": asset.status,
        }
        for asset in assets
    ]
