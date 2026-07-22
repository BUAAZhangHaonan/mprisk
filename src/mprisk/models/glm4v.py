"""GLM-4.6V-Flash native visual prefill extraction."""

from __future__ import annotations

from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import (
    HfVisualPrefillWrapper,
    request_video_fps,
    template_kwargs,
)


class Glm4vWrapper(HfVisualPrefillWrapper):
    family = "glm4v"
    model_type = "glm4v"
    architecture = "Glm4vForConditionalGeneration"
    processor_class = "Glm46VProcessor"
    provenance_schema = "mprisk_glm4_6v_prefill_provenance_v1"
    supports_thinking = True

    def _load_dependencies(self) -> tuple[Any, Any]:
        import torch
        from transformers import Glm4vForConditionalGeneration, Glm46VProcessor

        processor = Glm46VProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        model = Glm4vForConditionalGeneration.from_pretrained(
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
            **template_kwargs(enable_thinking=True, video_fps=fps),
        )
        return model_inputs, {
            "visual_input_types": _visual_types(request),
            "video_fps": fps,
            "native_video": "video" in _visual_types(request),
        }


def _visual_types(request: PrefillRequest) -> list[str]:
    return [
        str(item.get("type"))
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") in {"image", "video"}
    ]
