"""Gemma-3 VT prefill extraction using deterministic multi-image video sampling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import HfVisualPrefillWrapper
from mprisk.models.video_frame_utils import required_media_path, uniform_video_sample


class Gemma3Wrapper(HfVisualPrefillWrapper):
    family = "gemma3"
    model_type = "gemma3"
    architecture = "Gemma3ForConditionalGeneration"
    processor_class = "Gemma3Processor"
    provenance_schema = "mprisk_gemma3_prefill_provenance_v1"

    def __init__(self, *, video_num_segments: int = 8, **kwargs: Any) -> None:
        self.video_num_segments = int(video_num_segments)
        if not 1 <= self.video_num_segments <= 64:
            raise ValueError("Gemma-3 video_num_segments must be in [1, 64]")
        super().__init__(**kwargs)

    def _load_dependencies(self) -> tuple[Any, Any]:
        import torch
        from transformers import Gemma3ForConditionalGeneration, Gemma3Processor

        processor = Gemma3Processor.from_pretrained(
            self.model_path, local_files_only=True
        )
        model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype_name),
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        return model, processor

    def _prepare_inputs(self, request: PrefillRequest) -> tuple[Any, dict[str, Any]]:
        from PIL import Image

        messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        video_sources: list[str] = []
        video_metadata: list[dict[str, Any]] = []
        for message in request.messages:
            content: list[dict[str, Any]] = []
            for item in message.get("content", []):
                if not isinstance(item, Mapping):
                    raise TypeError("Gemma-3 content items must be mappings")
                item_type = str(item.get("type"))
                if item_type == "text":
                    content.append({"type": "text", "text": str(item.get("text", ""))})
                elif item_type == "image":
                    path = required_media_path(item.get("image"), "image")
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                    content.append({"type": "image"})
                elif item_type == "video":
                    path = required_media_path(item.get("video"), "video")
                    frames, metadata = uniform_video_sample(
                        path, self.video_num_segments
                    )
                    images.extend(frames)
                    content.extend({"type": "image"} for _ in frames)
                    video_sources.append(path)
                    video_metadata.append(metadata)
                else:
                    raise ValueError(f"Unsupported Gemma-3 content type: {item_type!r}")
            messages.append({"role": str(message.get("role")), "content": content})
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = [images]
        model_inputs = self.processor(**processor_kwargs)
        return model_inputs, {
            "visual_input_types": ["image"] * len(images),
            "video_sampling_method": "uniform_midpoint_decord_v1" if video_sources else None,
            "video_frame_count": len(images),
            "video_num_segments": self.video_num_segments,
            "video_sources": video_sources,
            "requested_frames": self.video_num_segments * len(video_sources),
            "actual_frames": sum(
                len(row["frames_indices"]) for row in video_metadata
            ),
            "video_frame_indices": [
                row["frames_indices"] for row in video_metadata
            ],
            "video_source_total_frames": [
                row["total_num_frames"] for row in video_metadata
            ],
        }
