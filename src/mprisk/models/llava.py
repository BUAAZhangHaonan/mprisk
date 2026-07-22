"""LLaVA VT prefill wrappers with deterministic multi-image video simulation."""

from __future__ import annotations

from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import HfVisualPrefillWrapper
from mprisk.models.video_frame_utils import request_text_and_frames


class _LlavaFrameWrapper(HfVisualPrefillWrapper):
    contract_location = "text_config"
    dtype_location = "root"

    def __init__(self, *, video_num_segments: int = 8, **kwargs: Any) -> None:
        self.video_num_segments = int(video_num_segments)
        if not 1 <= self.video_num_segments <= 64:
            raise ValueError("LLaVA video_num_segments must be in [1, 64]")
        super().__init__(**kwargs)

    def _prepare_inputs(self, request: PrefillRequest) -> tuple[Any, dict[str, Any]]:
        text, images, provenance = request_text_and_frames(
            request,
            video_num_segments=self.video_num_segments,
        )
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": text})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            kwargs["images"] = images
        return self.processor(**kwargs), provenance


class LlavaV15Wrapper(_LlavaFrameWrapper):
    family = "llava_v15"
    model_type = "llava"
    architecture = "LlavaForConditionalGeneration"
    processor_class = "LlavaProcessor"
    provenance_schema = "mprisk_llava_v15_prefill_provenance_v1"

    def _load_dependencies(self) -> tuple[Any, Any]:
        import torch
        from transformers import LlavaForConditionalGeneration, LlavaProcessor

        processor = LlavaProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype_name),
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        return model, processor


class LlavaOneVisionWrapper(_LlavaFrameWrapper):
    family = "llava_onevision"
    model_type = "llava_onevision"
    architecture = "LlavaOnevisionForConditionalGeneration"
    processor_class = "LlavaOnevisionProcessor"
    provenance_schema = "mprisk_llava_onevision_prefill_provenance_v1"

    def _load_dependencies(self) -> tuple[Any, Any]:
        import torch
        from transformers import (
            LlavaOnevisionForConditionalGeneration,
            LlavaOnevisionProcessor,
        )

        processor = LlavaOnevisionProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype_name),
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        return model, processor
