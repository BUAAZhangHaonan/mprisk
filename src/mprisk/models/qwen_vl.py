"""Qwen3-VL native visual prefill extraction."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mprisk.models.base_wrapper import BaseModelWrapper, PrefillRequest, PrefillResult
from mprisk.models.video_frame_utils import (
    request_messages_with_uniform_video,
    validate_video_grid_frames,
)


class QwenVlWrapper(BaseModelWrapper):
    """Extract all Qwen3-VL language-block states at the first reply position."""

    family = "qwen_vl"

    def __init__(
        self,
        *,
        model_key: str,
        model_path: str | Path,
        device: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        video_num_segments: int = 8,
        model: Any | None = None,
        processor: Any | None = None,
        runtime_versions: Mapping[str, str] | None = None,
        **_: Any,
    ) -> None:
        self.model_key = model_key
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.video_num_segments = int(video_num_segments)
        if not 1 <= self.video_num_segments <= 64:
            raise ValueError("Qwen3-VL video_num_segments must be in [1, 64]")
        self._contract = _load_model_contract(self.model_path)
        if dtype != self._contract["torch_dtype"]:
            raise ValueError(
                f"Requested dtype {dtype!r} does not match model config "
                f"{self._contract['torch_dtype']!r}"
            )
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be injected together")
        self.model = model
        self.processor = processor
        self._injected = model is not None
        if self._injected and runtime_versions is None:
            raise ValueError("Injected model dependencies require explicit runtime_versions")
        if not self._injected and runtime_versions is not None:
            raise ValueError("runtime_versions is only valid with injected model dependencies")
        self._runtime_versions = dict(runtime_versions or {})

    @property
    def expected_layer_count(self) -> int:
        return int(self._contract["num_hidden_layers"])

    @property
    def expected_hidden_dim(self) -> int:
        return int(self._contract["hidden_size"])

    def load(self) -> None:
        if self.model is not None:
            self._validate_loaded_contract()
            return
        import torch
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        processor_kwargs: dict[str, Any] = {"local_files_only": True}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self.processor = Qwen3VLProcessor.from_pretrained(self.model_path, **processor_kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype_name),
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        if self.model.__class__.__name__ != "Qwen3VLForConditionalGeneration":
            raise TypeError(f"Unexpected Qwen3-VL model class: {self.model.__class__.__name__}")
        self._validate_loaded_contract()

    def extract_prefill(self, request: PrefillRequest) -> PrefillResult:
        self._validate_request(request)
        if self.model is None:
            self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen3-VL wrapper is not fully loaded")

        import torch

        started_at = time.perf_counter()
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        messages, sampling = request_messages_with_uniform_video(
            request, requested_frames=self.video_num_segments
        )
        if sampling["requested_frames"]:
            template_kwargs["processor_kwargs"] = {
                "videos_kwargs": {
                    "do_sample_frames": False,
                    "video_metadata": sampling["video_metadata"],
                }
            }
        model_inputs = self.processor.apply_chat_template(
            messages,
            **template_kwargs,
        )
        if sampling["requested_frames"]:
            sampling["actual_frames"] = validate_video_grid_frames(
                model_inputs,
                processor=self.processor,
                requested_frames=int(sampling["requested_frames"]),
                family=self.family,
            )
        sampling.pop("video_metadata")
        model_inputs = _move_inputs_to_device(model_inputs, self.device)
        attention_mask = _require_attention_mask(model_inputs)
        token_count, t0_token_index = _token_position(attention_mask)

        track_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if track_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
            )
        trajectory = _trajectory_from_outputs(
            outputs,
            t0_token_index=t0_token_index,
            layer_count=self.expected_layer_count,
            hidden_dim=self.expected_hidden_dim,
        )
        peak_gpu_bytes = (
            int(torch.cuda.max_memory_allocated(torch.device(self.device))) if track_cuda else None
        )
        transformers_version = (
            self._runtime_versions["transformers"]
            if self._injected
            else __import__("transformers").__version__
        )
        return PrefillResult(
            request=request,
            trajectory=trajectory.detach().to(dtype=torch.float32, device="cpu").numpy(),
            token_count=token_count,
            t0_token_index=t0_token_index,
            provenance={
                "schema": "mprisk_qwen3_vl_prefill_provenance_v1",
                "model_path": str(self.model_path),
                "model_class": self.model.__class__.__name__,
                "processor_class": self.processor.__class__.__name__,
                "transformers_version": transformers_version,
                "torch_version": torch.__version__,
                "source_dtype": self.dtype_name,
                "stored_dtype": "float32",
                "device": self.device,
                "attn_implementation": self.attn_implementation,
                "num_hidden_layers": self.expected_layer_count,
                "hidden_size": self.expected_hidden_dim,
                "hidden_state_index_offset": 1,
                "model_config_sha256": _sha256(self.model_path / "config.json"),
                "weight_index_sha256": _sha256(
                    self.model_path / "model.safetensors.index.json"
                ),
                "elapsed_seconds": time.perf_counter() - started_at,
                "peak_gpu_memory_bytes": peak_gpu_bytes,
                "visual_input_types": _visual_input_types(request),
                **sampling,
            },
        )

    def close(self) -> None:
        if self._injected:
            return
        import torch

        self.model = None
        self.processor = None
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _validate_request(self, request: PrefillRequest) -> None:
        if request.model_key != self.model_key:
            raise ValueError(
                f"Request model_key {request.model_key!r} does not match {self.model_key!r}"
            )
        if request.protocol != "vt":
            raise ValueError("Qwen3-VL prefill extraction supports protocol VT only")
        if request.use_audio_in_video:
            raise ValueError("Qwen3-VL VT extraction must not enable audio from video")
        unsupported = set(_content_types(request)) - {"text", "image", "video"}
        if unsupported:
            raise ValueError(f"Qwen3-VL VT request has unsupported content: {sorted(unsupported)}")

    def _validate_loaded_contract(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        text_config = getattr(getattr(self.model, "config", None), "text_config", None)
        if text_config is None:
            raise ValueError("Qwen3-VL config does not expose text_config")
        actual = (int(text_config.num_hidden_layers), int(text_config.hidden_size))
        expected = (self.expected_layer_count, self.expected_hidden_dim)
        if actual != expected:
            raise ValueError(f"Loaded Qwen3-VL contract {actual} does not match config {expected}")


def _load_model_contract(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Qwen3-VL config is missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen3_vl":
        raise ValueError(f"Unexpected model_type in {config_path}: {payload.get('model_type')!r}")
    if payload.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise ValueError(f"Unexpected Qwen3-VL architecture in {config_path}")
    text = payload.get("text_config")
    if not isinstance(text, dict):
        raise ValueError(f"Qwen3-VL text_config is missing from {config_path}")
    contract = {
        "num_hidden_layers": int(text["num_hidden_layers"]),
        "hidden_size": int(text["hidden_size"]),
        "torch_dtype": str(text.get("dtype") or text.get("torch_dtype") or ""),
    }
    if contract["num_hidden_layers"] <= 0 or contract["hidden_size"] <= 0:
        raise ValueError(f"Invalid Qwen3-VL dimensions: {contract}")
    return contract


def _content_types(request: PrefillRequest) -> list[str]:
    return [
        str(item.get("type"))
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, Mapping)
    ]


def _visual_input_types(request: PrefillRequest) -> list[str]:
    return [item for item in _content_types(request) if item in {"image", "video"}]


def _request_video_fps(request: PrefillRequest) -> float | None:
    values = {
        float(item["fps"])
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, Mapping) and item.get("type") == "video" and "fps" in item
    }
    if len(values) > 1:
        raise ValueError("Qwen3-VL request cannot mix video fps values")
    return next(iter(values), None)


def _move_inputs_to_device(model_inputs: Any, device: str) -> Any:
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    if not isinstance(model_inputs, Mapping):
        raise TypeError("Qwen3-VL processor output must be a BatchFeature or mapping")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def _require_attention_mask(model_inputs: Any) -> Any:
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None or attention_mask.ndim != 2 or int(attention_mask.shape[0]) != 1:
        raise ValueError("Qwen3-VL extraction requires one two-dimensional attention_mask")
    return attention_mask


def _token_position(attention_mask: Any) -> tuple[int, int]:
    import torch

    token_count = int(attention_mask.shape[-1])
    non_padding = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
    if non_padding.numel() == 0:
        raise ValueError("attention_mask contains no conditioning tokens")
    return token_count, int(non_padding[-1].item())


def _trajectory_from_outputs(
    outputs: Any,
    *,
    t0_token_index: int,
    layer_count: int,
    hidden_dim: int,
) -> Any:
    import torch

    hidden_states = getattr(outputs, "hidden_states", None)
    expected_state_count = layer_count + 1
    if hidden_states is None or len(hidden_states) != expected_state_count:
        actual = None if hidden_states is None else len(hidden_states)
        raise ValueError(f"Expected {expected_state_count} hidden-state tensors, got {actual}")
    trajectory = torch.stack(
        [state[0, t0_token_index, :] for state in hidden_states[1:]], dim=0
    )
    if tuple(trajectory.shape) != (layer_count, hidden_dim):
        raise ValueError(
            f"Expected Qwen3-VL trajectory shape {(layer_count, hidden_dim)}, "
            f"got {tuple(trajectory.shape)}"
        )
    if not torch.isfinite(trajectory).all().item():
        raise ValueError("Qwen3-VL trajectory contains non-finite values")
    return trajectory


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required provenance file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
