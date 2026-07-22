"""Gemma-3 VT prefill extraction using deterministic multi-image video sampling."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import HfVisualPrefillWrapper


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
        for message in request.messages:
            content: list[dict[str, Any]] = []
            for item in message.get("content", []):
                if not isinstance(item, Mapping):
                    raise TypeError("Gemma-3 content items must be mappings")
                item_type = str(item.get("type"))
                if item_type == "text":
                    content.append({"type": "text", "text": str(item.get("text", ""))})
                elif item_type == "image":
                    path = _required_path(item.get("image"), "image")
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                    content.append({"type": "image"})
                elif item_type == "video":
                    path = _required_path(item.get("video"), "video")
                    frames = _uniform_video_frames(path, self.video_num_segments)
                    images.extend(frames)
                    content.extend({"type": "image"} for _ in frames)
                    video_sources.append(path)
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
        }


def _required_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Gemma-3 {label} requires a local path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _uniform_video_frames(path: str, count: int) -> list[Any]:
    import decord
    from PIL import Image

    reader = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
    length = len(reader)
    if length <= 0:
        raise ValueError(f"Video has no frames: {path}")
    indices = [min(length - 1, int((index + 0.5) * length / count)) for index in range(count)]
    array = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in array]
