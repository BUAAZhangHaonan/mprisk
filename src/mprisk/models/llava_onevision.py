"""LLaVA-OneVision native-video prefill extraction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import HfVisualPrefillWrapper, template_kwargs
from mprisk.models.video_frame_utils import request_messages_with_uniform_video


class LlavaOneVisionWrapper(HfVisualPrefillWrapper):
    """Extract all language-layer t0 states from one native F8 video input."""

    family = "llava_onevision"
    model_type = "llava_onevision"
    architecture = "LlavaOnevisionForConditionalGeneration"
    processor_class = "LlavaOnevisionProcessor"
    provenance_schema = "mprisk_llava_onevision_native_video_prefill_provenance_v1"
    contract_location = "text_config"
    dtype_location = "root"

    def __init__(self, *, video_num_segments: int = 8, **kwargs: Any) -> None:
        self.video_num_segments = int(video_num_segments)
        if self.video_num_segments != 8:
            raise ValueError("LLaVA-OneVision requires the frozen native-video F8 protocol")
        super().__init__(**kwargs)

    def _load_contract(self) -> dict[str, Any]:
        contract = super()._load_contract()
        config_path = self.model_path / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = payload.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError("LLaVA-OneVision config has no text_config")
        from transformers import LlavaOnevisionConfig

        effective_config = LlavaOnevisionConfig.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        effective_text = getattr(effective_config, "text_config", None)
        if effective_text is None:
            raise ValueError("LLaVA-OneVision effective config has no text_config")
        max_position_embeddings = int(effective_text.max_position_embeddings)
        raw_limit = text_config.get("max_position_embeddings")
        if raw_limit is not None and int(raw_limit) != max_position_embeddings:
            raise ValueError(
                "LLaVA-OneVision raw and effective context limits do not match"
            )
        if max_position_embeddings <= 0:
            raise ValueError("LLaVA-OneVision context limit must be positive")
        contract["max_position_embeddings"] = max_position_embeddings
        return contract

    def _load_dependencies(self) -> tuple[Any, Any]:
        import torch
        from transformers import (
            LlavaOnevisionForConditionalGeneration,
            LlavaOnevisionProcessor,
        )

        processor = LlavaOnevisionProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype_name),
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        return model, processor

    def _validate_request(self, request: PrefillRequest) -> None:
        super()._validate_request(request)
        types = _content_types(request)
        if "image" in types:
            raise ValueError("LLaVA-OneVision cache extraction requires native video, not images")
        expected_videos = 0 if request.condition == "M2" else 1
        if types.count("video") != expected_videos:
            raise ValueError(
                f"LLaVA-OneVision {request.condition} requires exactly "
                f"{expected_videos} video item(s)"
            )

    def _prepare_inputs(self, request: PrefillRequest) -> tuple[Any, dict[str, Any]]:
        messages, sampling = request_messages_with_uniform_video(
            request,
            requested_frames=self.video_num_segments,
        )
        kwargs = template_kwargs(enable_thinking=False, video_fps=None)
        if sampling["requested_frames"]:
            kwargs["processor_kwargs"] = {
                "videos_kwargs": {"do_sample_frames": False}
            }
        model_inputs = self.processor.apply_chat_template(messages, **kwargs)
        expected_frames = (
            self.video_num_segments if request.condition in {"M1", "M12"} else 0
        )
        token_count, actual_frames = _validate_native_video_processor_output(
            model_inputs,
            expected_frames=expected_frames,
            max_position_embeddings=int(self._contract["max_position_embeddings"]),
        )
        if int(sampling["requested_frames"]) != expected_frames:
            raise ValueError("LLaVA-OneVision decoded request does not match F8 protocol")
        sampling["actual_frames"] = actual_frames
        sampling.pop("video_metadata")
        return model_inputs, {
            "visual_input_types": [
                item for item in _content_types(request) if item in {"image", "video"}
            ],
            "video_input_mode": "native_video",
            "context_limit": int(self._contract["max_position_embeddings"]),
            "processor_token_count": token_count,
            "no_truncation": True,
            **sampling,
        }


def _validate_native_video_processor_output(
    model_inputs: Mapping[str, Any],
    *,
    expected_frames: int,
    max_position_embeddings: int,
) -> tuple[int, int]:
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None or getattr(attention_mask, "ndim", None) != 2:
        raise ValueError("LLaVA-OneVision processor requires a 2D attention mask")
    if int(attention_mask.shape[0]) != 1:
        raise ValueError("LLaVA-OneVision processor requires batch size one")
    token_count = int(attention_mask.shape[-1])
    if token_count > max_position_embeddings:
        raise ValueError(
            f"LLaVA-OneVision processor produced {token_count} tokens, which exceeds "
            f"the checkpoint limit {max_position_embeddings}"
        )
    if "pixel_values" in model_inputs or "image_sizes" in model_inputs:
        raise ValueError("LLaVA-OneVision native-video path produced image tensors")
    video = model_inputs.get("pixel_values_videos")
    if expected_frames == 0:
        if video is not None:
            raise ValueError("LLaVA-OneVision M2 unexpectedly produced video tensors")
        return token_count, 0
    if video is None:
        raise ValueError("LLaVA-OneVision native-video path produced no video tensor")
    shape = tuple(int(value) for value in video.shape)
    if len(shape) != 5 or shape[0] != 1 or shape[1] != expected_frames:
        raise ValueError(
            "LLaVA-OneVision native-video tensor must have shape "
            f"[1,{expected_frames},C,H,W], got {shape}"
        )
    if shape[2] != 3 or shape[3] <= 0 or shape[4] <= 0:
        raise ValueError(f"Invalid LLaVA-OneVision video tensor shape: {shape}")
    return token_count, shape[1]


def _content_types(request: PrefillRequest) -> list[str]:
    return [
        str(item.get("type"))
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, Mapping)
    ]
