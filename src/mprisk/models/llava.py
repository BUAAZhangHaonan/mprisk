"""LLaVA VT prefill wrappers with deterministic multi-image video simulation."""

from __future__ import annotations

import json
import re
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

    def __init__(self, *, video_num_segments: int = 7, **kwargs: Any) -> None:
        if int(video_num_segments) > 7:
            raise ValueError(
                "LLaVA-v1.5 supports at most 7 sampled frames under its 4096-token context"
            )
        super().__init__(video_num_segments=video_num_segments, **kwargs)

    def _load_contract(self) -> dict[str, Any]:
        """Derive the legacy Vicuna language contract from checkpoint tensors.

        This local HF conversion predates complete nested Llama config metadata.
        The safetensors index and embedding tensor are the exact checkpoint
        contract, so extraction fails closed if their layer map is incomplete.
        """
        config_path = self.model_path / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("model_type") != self.model_type:
            raise ValueError(f"Unexpected model_type in {config_path}")
        if payload.get("architectures") != [self.architecture]:
            raise ValueError(f"Unexpected architecture in {config_path}")
        text_config = payload.get("text_config")
        if not isinstance(text_config, dict) or text_config.get("model_type") != "llama":
            raise ValueError("LLaVA-v1.5 requires a nested Llama text_config")
        index_path = self.model_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("LLaVA-v1.5 weight index has no weight_map")
        layer_pattern = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
        layers = sorted(
            {
                int(match.group(1))
                for key in weight_map
                if (match := layer_pattern.match(str(key))) is not None
            }
        )
        if not layers or layers != list(range(layers[-1] + 1)):
            raise ValueError("LLaVA-v1.5 language layer index is not contiguous")
        embedding_key = "language_model.model.embed_tokens.weight"
        shard_name = weight_map.get(embedding_key)
        if not isinstance(shard_name, str):
            raise ValueError("LLaVA-v1.5 embedding tensor is absent from the index")
        from safetensors import safe_open

        with safe_open(
            self.model_path / shard_name,
            framework="pt",
            device="cpu",
        ) as handle:
            shape = tuple(int(value) for value in handle.get_slice(embedding_key).get_shape())
        if len(shape) != 2 or shape[1] <= 0:
            raise ValueError(f"Invalid LLaVA-v1.5 embedding shape: {shape}")
        dtype = str(payload.get("dtype") or payload.get("torch_dtype") or "")
        return {
            "num_hidden_layers": len(layers),
            "hidden_size": shape[1],
            "torch_dtype": dtype,
        }

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
