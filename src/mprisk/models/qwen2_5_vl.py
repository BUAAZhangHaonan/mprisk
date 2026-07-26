"""Qwen2.5-VL native visual prefill extraction."""

from __future__ import annotations

from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import (
    HfVisualPrefillWrapper,
    template_kwargs,
)
from mprisk.models.video_frame_utils import (
    request_messages_with_uniform_video,
    validate_video_grid_frames,
)


class Qwen2_5VlWrapper(HfVisualPrefillWrapper):
    family = "qwen2_5_vl"
    model_type = "qwen2_5_vl"
    architecture = "Qwen2_5_VLForConditionalGeneration"
    processor_class = "Qwen2_5_VLProcessor"
    provenance_schema = "mprisk_qwen2_5_vl_prefill_provenance_v1"
    contract_location = "root"
    loaded_contract_location = "text_config"

    def __init__(self, *, video_num_segments: int = 8, **kwargs: Any) -> None:
        self.video_num_segments = int(video_num_segments)
        if not 1 <= self.video_num_segments <= 64:
            raise ValueError("Qwen2.5-VL video_num_segments must be in [1, 64]")
        super().__init__(**kwargs)

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
        messages, sampling = request_messages_with_uniform_video(
            request, requested_frames=self.video_num_segments
        )
        kwargs = template_kwargs(enable_thinking=False, video_fps=None)
        if sampling["requested_frames"]:
            kwargs["processor_kwargs"] = {
                "videos_kwargs": {
                    "do_sample_frames": False,
                    "video_metadata": sampling["video_metadata"],
                }
            }
        model_inputs = self.processor.apply_chat_template(
            messages,
            **kwargs,
        )
        if sampling["requested_frames"]:
            sampling["actual_frames"] = validate_video_grid_frames(
                model_inputs,
                processor=self.processor,
                requested_frames=int(sampling["requested_frames"]),
                family=self.family,
            )
        sampling.pop("video_metadata")
        return model_inputs, {
            "visual_input_types": _visual_types(request),
            **sampling,
        }


def _visual_types(request: PrefillRequest) -> list[str]:
    return [
        str(item.get("type"))
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") in {"image", "video"}
    ]
