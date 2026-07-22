"""Phi-3.5-Vision VT prefill extraction with deterministic frame sampling."""

from __future__ import annotations

from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import HfVisualPrefillWrapper
from mprisk.models.video_frame_utils import request_text_and_frames


class Phi3VisionWrapper(HfVisualPrefillWrapper):
    family = "phi3_vision"
    model_type = "phi3_v"
    architecture = "Phi3VForCausalLM"
    processor_class = "Phi3VProcessor"
    provenance_schema = "mprisk_phi3_vision_prefill_provenance_v1"
    contract_location = "root"
    loaded_contract_location = "root"
    forward_logits_to_keep = False

    def __init__(
        self,
        *,
        video_num_segments: int = 8,
        processor_num_crops: int = 4,
        **kwargs: Any,
    ) -> None:
        self.video_num_segments = int(video_num_segments)
        self.processor_num_crops = int(processor_num_crops)
        if not 1 <= self.video_num_segments <= 64:
            raise ValueError("Phi-3.5 video_num_segments must be in [1, 64]")
        if self.processor_num_crops <= 0:
            raise ValueError("Phi-3.5 processor_num_crops must be positive")
        super().__init__(**kwargs)

    def _load_dependencies(self) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            num_crops=self.processor_num_crops,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype=getattr(torch, self.dtype_name),
            _attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval()
        return model, processor

    def _prepare_inputs(self, request: PrefillRequest) -> tuple[Any, dict[str, Any]]:
        text, images, provenance = request_text_and_frames(
            request,
            video_num_segments=self.video_num_segments,
        )
        image_tags = "\n".join(
            f"<|image_{index}|>" for index in range(1, len(images) + 1)
        )
        body = f"{image_tags}\n{text}" if image_tags else text
        prompt = f"<|user|>\n{body}<|end|>\n<|assistant|>\n"
        kwargs: dict[str, Any] = {
            "text": prompt,
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = images
        return self.processor(**kwargs), provenance
