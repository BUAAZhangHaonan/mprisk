"""Qwen2.5-VL native visual prefill extraction."""

from __future__ import annotations

from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import (
    HfVisualPrefillWrapper,
    request_video_fps,
    template_kwargs,
)


class Qwen2_5VlWrapper(HfVisualPrefillWrapper):
    family = "qwen2_5_vl"
    model_type = "qwen2_5_vl"
    architecture = "Qwen2_5_VLForConditionalGeneration"
    processor_class = "Qwen2_5_VLProcessor"
    provenance_schema = "mprisk_qwen2_5_vl_prefill_provenance_v1"
    contract_location = "root"
    loaded_contract_location = "text_config"

    def _load_dependencies(self) -> tuple[Any, Any]:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

        processor = Qwen2_5_VLProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype_name),
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        return model, processor

    def _prepare_inputs(self, request: PrefillRequest) -> tuple[Any, dict[str, Any]]:
        fps = request_video_fps(request)
        model_inputs = self.processor.apply_chat_template(
            [dict(message) for message in request.messages],
            **template_kwargs(enable_thinking=False, video_fps=fps),
        )
        return model_inputs, {
            "visual_input_types": _visual_types(request),
            "video_fps": fps,
        }


def _visual_types(request: PrefillRequest) -> list[str]:
    return [
        str(item.get("type"))
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") in {"image", "video"}
    ]
