"""Shared fail-closed prefill extraction for native Transformers visual models."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mprisk.models.base_wrapper import BaseModelWrapper, PrefillRequest, PrefillResult


class HfVisualPrefillWrapper(BaseModelWrapper, ABC):
    """Common all-layer t0 extraction contract for Transformers visual backbones."""

    model_type: str
    architecture: str
    processor_class: str
    provenance_schema: str
    contract_location: str = "text_config"
    loaded_contract_location: str = "text_config"
    supports_thinking: bool = False

    def __init__(
        self,
        *,
        model_key: str,
        model_path: str | Path,
        device: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
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
        self._contract = self._load_contract()
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
            raise ValueError("Injected dependencies require explicit runtime_versions")
        if not self._injected and runtime_versions is not None:
            raise ValueError("runtime_versions is valid only with injected dependencies")
        self._runtime_versions = dict(runtime_versions or {})

    @property
    def expected_layer_count(self) -> int:
        return int(self._contract["num_hidden_layers"])

    @property
    def expected_hidden_dim(self) -> int:
        return int(self._contract["hidden_size"])

    @abstractmethod
    def _load_dependencies(self) -> tuple[Any, Any]:
        """Return the exact model and processor instances for this family."""

    @abstractmethod
    def _prepare_inputs(self, request: PrefillRequest) -> tuple[Any, Mapping[str, Any]]:
        """Return processor output and media-specific provenance."""

    def load(self) -> None:
        if self.model is None:
            self.model, self.processor = self._load_dependencies()
        self._validate_loaded_contract()

    def extract_prefill(self, request: PrefillRequest) -> PrefillResult:
        self._validate_request(request)
        self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError(f"{self.family} wrapper is not fully loaded")

        import torch

        started_at = time.perf_counter()
        model_inputs, media_provenance = self._prepare_inputs(request)
        model_inputs = _move_inputs_to_device(model_inputs, self.device)
        attention_mask = _require_attention_mask(model_inputs, self.family)
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
            family=self.family,
        )
        peak_gpu_bytes = (
            int(torch.cuda.max_memory_allocated(torch.device(self.device)))
            if track_cuda
            else None
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
                "schema": self.provenance_schema,
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
                "thinking_enabled": False,
                **dict(media_provenance),
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
            raise ValueError(f"{self.family} prefill extraction supports VT only")
        if request.use_audio_in_video:
            raise ValueError(f"{self.family} VT extraction must not enable video audio")
        unsupported = set(_content_types(request)) - {"text", "image", "video"}
        if unsupported:
            raise ValueError(f"Unsupported {self.family} content: {sorted(unsupported)}")

    def _load_contract(self) -> dict[str, Any]:
        config_path = self.model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("model_type") != self.model_type:
            raise ValueError(f"Unexpected model_type in {config_path}")
        if payload.get("architectures") != [self.architecture]:
            raise ValueError(f"Unexpected architecture in {config_path}")
        config = payload if self.contract_location == "root" else payload.get("text_config")
        if not isinstance(config, dict):
            raise ValueError(f"Missing {self.contract_location} contract in {config_path}")
        contract = {
            "num_hidden_layers": int(config["num_hidden_layers"]),
            "hidden_size": int(config["hidden_size"]),
            "torch_dtype": str(
                config.get("dtype")
                or config.get("torch_dtype")
                or payload.get("dtype")
                or payload.get("torch_dtype")
                or ""
            ),
        }
        if contract["num_hidden_layers"] <= 0 or contract["hidden_size"] <= 0:
            raise ValueError(f"Invalid model dimensions: {contract}")
        return contract

    def _validate_loaded_contract(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        config = getattr(self.model, "config", None)
        if self.loaded_contract_location != "root":
            config = getattr(config, "text_config", None)
        if config is None:
            raise ValueError(f"Loaded {self.family} model has no language config")
        actual = (int(config.num_hidden_layers), int(config.hidden_size))
        expected = (self.expected_layer_count, self.expected_hidden_dim)
        if actual != expected:
            raise ValueError(f"Loaded {self.family} contract {actual} != {expected}")
        if self.model.__class__.__name__ != self.architecture:
            raise TypeError(
                f"Unexpected {self.family} model class: {self.model.__class__.__name__}"
            )
        if self.processor.__class__.__name__ != self.processor_class:
            raise TypeError(
                f"Unexpected {self.family} processor class: "
                f"{self.processor.__class__.__name__}"
            )


def template_kwargs(*, enable_thinking: bool, video_fps: float | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if enable_thinking:
        kwargs["enable_thinking"] = False
    if video_fps is not None:
        kwargs["fps"] = video_fps
    return kwargs


def request_video_fps(request: PrefillRequest) -> float | None:
    values = {
        float(item["fps"])
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, Mapping) and item.get("type") == "video" and "fps" in item
    }
    if len(values) > 1:
        raise ValueError("A request cannot mix video fps values")
    return next(iter(values), None)


def _content_types(request: PrefillRequest) -> list[str]:
    return [
        str(item.get("type"))
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, Mapping)
    ]


def _move_inputs_to_device(model_inputs: Any, device: str) -> Any:
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    if not isinstance(model_inputs, Mapping):
        raise TypeError("Processor output must be a BatchFeature or mapping")
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in model_inputs.items()}


def _require_attention_mask(model_inputs: Any, family: str) -> Any:
    mask = model_inputs.get("attention_mask")
    if mask is None or mask.ndim != 2 or int(mask.shape[0]) != 1:
        raise ValueError(f"{family} requires a batch-one two-dimensional attention_mask")
    return mask


def _token_position(attention_mask: Any) -> tuple[int, int]:
    import torch

    non_padding = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
    if non_padding.numel() == 0:
        raise ValueError("attention_mask has no conditioning token")
    return int(attention_mask.shape[-1]), int(non_padding[-1].item())


def _trajectory_from_outputs(
    outputs: Any, *, t0_token_index: int, layer_count: int, hidden_dim: int, family: str
) -> Any:
    import torch

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) != layer_count + 1:
        actual = None if hidden_states is None else len(hidden_states)
        raise ValueError(f"{family} expected {layer_count + 1} states, got {actual}")
    trajectory = torch.stack(
        [state[0, t0_token_index, :] for state in hidden_states[1:]], dim=0
    )
    if tuple(trajectory.shape) != (layer_count, hidden_dim):
        raise ValueError(f"Unexpected {family} trajectory shape: {tuple(trajectory.shape)}")
    if not torch.isfinite(trajectory).all().item():
        raise ValueError(f"{family} trajectory contains non-finite values")
    return trajectory


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
