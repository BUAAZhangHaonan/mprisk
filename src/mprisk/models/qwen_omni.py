"""Qwen2.5-Omni Thinker-only prefill extraction."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mprisk.models.base_wrapper import (
    BaseModelWrapper,
    GenerationRequest,
    GenerationResult,
    PrefillRequest,
    PrefillResult,
)

JointAudioMode = Literal["embedded_video", "separate_file"]


def build_condition_request(
    *,
    sample_id: str,
    model_key: str,
    protocol: str,
    condition: str,
    dataset_key: str,
    split: str,
    media_paths: Mapping[str, str],
    transcript: str | None,
    task_prompt: str,
    joint_audio_mode: JointAudioMode = "embedded_video",
    video_fps: float = 1.0,
) -> PrefillRequest:
    """Build an explicit VT, VA, or VTA conditioning view.

    VTA treats text as shared context: M1 is vision+text, M2 is audio+text,
    and M12 is vision+audio+text.
    """
    protocol = protocol.lower()
    condition = condition.upper()
    prompt = task_prompt.strip()
    if not prompt:
        raise ValueError("task_prompt must not be empty")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if joint_audio_mode not in {"embedded_video", "separate_file"}:
        raise ValueError(f"Unsupported joint_audio_mode: {joint_audio_mode!r}")

    vision_path = media_paths.get("vision")
    audio_path = media_paths.get("audio")
    include_transcript = protocol == "vt" and condition in {"M2", "M12"}
    include_transcript = include_transcript or protocol == "vta"
    text = _prompt_text(prompt, transcript=transcript, include_transcript=include_transcript)

    content: list[dict[str, Any]] = []
    use_audio_in_video = False
    if protocol == "vt":
        if condition in {"M1", "M12"}:
            content.append(_video_content(vision_path, video_fps))
    elif protocol in {"va", "vta"}:
        if condition == "M1":
            content.append(_video_content(vision_path, video_fps))
        elif condition == "M2":
            content.append(_audio_content(audio_path))
        elif condition == "M12":
            content.append(_video_content(vision_path, video_fps))
            if joint_audio_mode == "embedded_video":
                _require_same_media(vision_path, audio_path)
                use_audio_in_video = True
            else:
                content.append(_audio_content(audio_path))
    else:
        raise ValueError(f"Unsupported prefill protocol: {protocol!r}")

    content.append({"type": "text", "text": text})
    return PrefillRequest(
        sample_id=sample_id,
        model_key=model_key,
        protocol=protocol,
        condition=condition,
        dataset_key=dataset_key,
        split=split,
        messages=({"role": "user", "content": content},),
        media_paths=media_paths,
        use_audio_in_video=use_audio_in_video,
    )


class QwenOmniWrapper(BaseModelWrapper):
    """Extract the first-reply-token conditioning trajectory from the Thinker."""

    family = "qwen_omni"

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
        model: Any | None = None,
        processor: Any | None = None,
        process_mm_info_fn: Any | None = None,
        runtime_versions: Mapping[str, str] | None = None,
    ) -> None:
        self.model_key = model_key
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._contract = _load_model_contract(self.model_path)
        if dtype != self._contract["torch_dtype"]:
            raise ValueError(
                f"Requested dtype {dtype!r} does not match model config "
                f"{self._contract['torch_dtype']!r}"
            )
        injected = (model, processor, process_mm_info_fn)
        if any(value is not None for value in injected) and not all(
            value is not None for value in injected
        ):
            raise ValueError("model, processor, and process_mm_info_fn must be injected together")
        self.model = model
        self.processor = processor
        self._process_mm_info = process_mm_info_fn
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
        from qwen_omni_utils import process_mm_info
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        processor_kwargs: dict[str, Any] = {"local_files_only": True, "use_fast": False}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path,
            **processor_kwargs,
        )
        dtype = getattr(torch, self.dtype_name)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        self._process_mm_info = process_mm_info
        if self.model.__class__.__name__ != "Qwen2_5OmniThinkerForConditionalGeneration":
            raise TypeError(f"Unexpected Qwen Omni model class: {self.model.__class__.__name__}")
        if hasattr(self.model, "talker") or hasattr(self.model, "token2wav"):
            raise RuntimeError("Thinker-only extraction must not load Talker or token2wav")
        self._validate_loaded_contract()

    def extract_prefill(self, request: PrefillRequest) -> PrefillResult:
        if request.model_key != self.model_key:
            raise ValueError(
                f"Request model_key {request.model_key!r} does not match {self.model_key!r}"
            )
        if self.model is None:
            self.load()
        if self.processor is None or self._process_mm_info is None:
            raise RuntimeError("Qwen Omni wrapper is not fully loaded")

        import torch

        _validate_message_audio_contract(request)
        started_at = time.perf_counter()
        prompt = self.processor.apply_chat_template(
            list(request.messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        audios, images, videos = self._process_mm_info(
            list(request.messages),
            use_audio_in_video=request.use_audio_in_video,
        )
        model_inputs = self.processor(
            text=[prompt],
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=request.use_audio_in_video,
        )
        model_inputs = _move_inputs_to_device(model_inputs, self.device)
        attention_mask = _require_attention_mask(model_inputs)
        token_count = int(attention_mask.shape[-1])
        non_padding = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
        if non_padding.numel() == 0:
            raise ValueError("attention_mask contains no conditioning tokens")
        t0_token_index = int(non_padding[-1].item())

        track_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if track_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                use_audio_in_video=request.use_audio_in_video,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        expected_state_count = self.expected_layer_count + 1
        if hidden_states is None or len(hidden_states) != expected_state_count:
            actual = None if hidden_states is None else len(hidden_states)
            raise ValueError(f"Expected {expected_state_count} hidden-state tensors, got {actual}")
        trajectory = torch.stack(
            [state[0, t0_token_index, :] for state in hidden_states[1:]],
            dim=0,
        )
        expected_shape = (self.expected_layer_count, self.expected_hidden_dim)
        if tuple(trajectory.shape) != expected_shape:
            raise ValueError(
                f"Expected thinker trajectory shape {expected_shape}, got {tuple(trajectory.shape)}"
            )
        if not torch.isfinite(trajectory).all().item():
            raise ValueError("Thinker trajectory contains non-finite values")
        peak_gpu_bytes = (
            int(torch.cuda.max_memory_allocated(torch.device(self.device))) if track_cuda else None
        )
        elapsed_seconds = time.perf_counter() - started_at
        stored_trajectory = trajectory.detach().to(dtype=torch.float32, device="cpu").numpy()
        if self._injected:
            transformers_version = self._runtime_versions["transformers"]
            qwen_omni_utils_version = self._runtime_versions["qwen-omni-utils"]
        else:
            import transformers

            transformers_version = transformers.__version__
            qwen_omni_utils_version = importlib.metadata.version("qwen-omni-utils")
        provenance = {
            "schema": "mprisk_qwen_omni_prefill_provenance_v1",
            "model_path": str(self.model_path),
            "model_class": self.model.__class__.__name__,
            "processor_class": self.processor.__class__.__name__,
            "talker_loaded": False,
            "transformers_version": transformers_version,
            "qwen_omni_utils_version": qwen_omni_utils_version,
            "torch_version": torch.__version__,
            "source_dtype": self.dtype_name,
            "stored_dtype": "float32",
            "device": self.device,
            "attn_implementation": self.attn_implementation,
            "num_hidden_layers": self.expected_layer_count,
            "hidden_size": self.expected_hidden_dim,
            "hidden_state_index_offset": 1,
            "model_config_sha256": _sha256(self.model_path / "config.json"),
            "weight_index_sha256": _sha256(self.model_path / "model.safetensors.index.json"),
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_bytes": peak_gpu_bytes,
        }
        return PrefillResult(
            request=request,
            trajectory=np.asarray(stored_trajectory),
            token_count=token_count,
            t0_token_index=t0_token_index,
            provenance=provenance,
        )

    def generate_conditioned(self, request: GenerationRequest) -> GenerationResult:
        """Generate only with the Thinker and retain only newly generated tokens."""
        if request.model_key != self.model_key:
            raise ValueError(
                f"Request model_key {request.model_key!r} does not match {self.model_key!r}"
            )
        if self.model is None:
            self.load()
        if self.processor is None or self._process_mm_info is None:
            raise RuntimeError("Qwen Omni wrapper is not fully loaded")

        import torch

        _validate_generation_audio_contract(request)
        prompt = self.processor.apply_chat_template(
            list(request.messages), tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = self._process_mm_info(
            list(request.messages), use_audio_in_video=request.use_audio_in_video
        )
        model_inputs = self.processor(
            text=[prompt],
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=request.use_audio_in_video,
        )
        model_inputs = _move_inputs_to_device(model_inputs, self.device)
        _require_attention_mask(model_inputs)
        input_ids = model_inputs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("Qwen Omni generation requires input_ids with batch size exactly one")
        input_token_count = int(input_ids.shape[-1])
        eos_token_ids = _tokenizer_eos_token_ids(self.processor)
        eos_token_id = eos_token_ids[0] if len(eos_token_ids) == 1 else list(eos_token_ids)
        track_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if track_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        started_at = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                **dict(request.generation_kwargs),
                eos_token_id=eos_token_id,
                use_audio_in_video=request.use_audio_in_video,
            )
        if generated.ndim != 2 or int(generated.shape[0]) != 1:
            raise ValueError("Qwen Omni generation requires a single generated sequence")
        new_token_ids = generated[:, input_token_count:]
        if int(new_token_ids.shape[-1]) == 0:
            raise ValueError("Qwen Omni generated no new tokens")
        token_ids = tuple(int(token) for token in new_token_ids[0].detach().cpu().tolist())
        text = self.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        finish_reason = "eos" if token_ids[-1] in eos_token_ids else "max_new_tokens"
        peak_gpu_memory_bytes = (
            int(torch.cuda.max_memory_allocated(torch.device(self.device))) if track_cuda else None
        )
        return GenerationResult(
            request=request,
            text=text,
            token_ids=token_ids,
            eos_token_ids=eos_token_ids,
            finish_reason=finish_reason,
            input_token_count=input_token_count,
            provenance={
                "schema": "mprisk_qwen_omni_generation_provenance_v1",
                "model_path": str(self.model_path),
                "model_class": self.model.__class__.__name__,
                "processor_class": self.processor.__class__.__name__,
                "talker_loaded": False,
                "source_dtype": self.dtype_name,
                "device": self.device,
                "attn_implementation": self.attn_implementation,
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": request.generation_kwargs["max_new_tokens"],
                "elapsed_seconds": time.perf_counter() - started_at,
                "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            },
        )

    def close(self) -> None:
        if self._injected:
            return
        import torch

        self.model = None
        self.processor = None
        self._process_mm_info = None
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _validate_loaded_contract(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        text_config = getattr(getattr(self.model, "config", None), "text_config", None)
        if text_config is None:
            raise ValueError("Thinker model config does not expose text_config")
        actual = (int(text_config.num_hidden_layers), int(text_config.hidden_size))
        expected = (self.expected_layer_count, self.expected_hidden_dim)
        if actual != expected:
            raise ValueError(f"Loaded thinker contract {actual} does not match config {expected}")


def _load_model_contract(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Qwen Omni config is missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen2_5_omni":
        raise ValueError(f"Unexpected model_type in {config_path}: {payload.get('model_type')!r}")
    thinker = payload.get("thinker_config")
    text = thinker.get("text_config") if isinstance(thinker, dict) else None
    if not isinstance(text, dict):
        raise ValueError(f"Qwen Omni thinker text_config is missing from {config_path}")
    contract = {
        "num_hidden_layers": int(text["num_hidden_layers"]),
        "hidden_size": int(text["hidden_size"]),
        "torch_dtype": str(thinker["torch_dtype"]),
    }
    if contract["num_hidden_layers"] <= 0 or contract["hidden_size"] <= 0:
        raise ValueError(f"Invalid Qwen Omni thinker dimensions: {contract}")
    return contract


def _prompt_text(prompt: str, *, transcript: str | None, include_transcript: bool) -> str:
    if not include_transcript:
        return prompt
    if transcript is None or not transcript.strip():
        raise ValueError("This conditioning view requires a non-empty transcript")
    return f"Transcript:\n{transcript.strip()}\n\nTask:\n{prompt}"


def _video_content(path: str | None, fps: float) -> dict[str, Any]:
    if not path:
        raise ValueError("This conditioning view requires media_paths.vision")
    return {"type": "video", "video": path, "fps": fps}


def _audio_content(path: str | None) -> dict[str, Any]:
    if not path:
        raise ValueError("This conditioning view requires media_paths.audio")
    return {"type": "audio", "audio": path}


def _require_same_media(vision_path: str | None, audio_path: str | None) -> None:
    if not vision_path or not audio_path:
        raise ValueError("embedded_video mode requires both vision and audio paths")
    vision = Path(vision_path).expanduser().resolve()
    audio = Path(audio_path).expanduser().resolve()
    if vision != audio:
        raise ValueError(
            "embedded_video mode requires vision and audio to reference the same media file"
        )


def _validate_message_audio_contract(request: PrefillRequest) -> None:
    content_types = [
        str(item.get("type"))
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, Mapping)
    ]
    if request.use_audio_in_video:
        if "video" not in content_types:
            raise ValueError("use_audio_in_video=True requires an explicit video message")
        if "audio" in content_types:
            raise ValueError("Embedded-video audio and explicit audio must not be enabled together")


def _validate_generation_audio_contract(request: GenerationRequest) -> None:
    content_types = [
        str(item.get("type"))
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, Mapping)
    ]
    if request.use_audio_in_video:
        if "video" not in content_types or "audio" in content_types:
            raise ValueError("Embedded-video generation requires video without explicit audio")


def _tokenizer_eos_token_ids(processor: Any) -> tuple[int, ...]:
    tokenizer = getattr(processor, "tokenizer", None)
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int):
        return (value,)
    if isinstance(value, tuple | list) and all(isinstance(token_id, int) for token_id in value):
        return tuple(sorted(set(value)))
    raise ValueError("Qwen Omni tokenizer must define one or more integer eos_token_id values")


def _require_attention_mask(model_inputs: Any) -> Any:
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        raise ValueError("Qwen Omni processor output must include attention_mask")
    if attention_mask.ndim != 2 or int(attention_mask.shape[0]) != 1:
        raise ValueError("Qwen Omni prefill extraction requires batch size exactly one")
    return attention_mask


def _move_inputs_to_device(model_inputs: Any, device: str) -> Any:
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    if not isinstance(model_inputs, Mapping):
        raise TypeError("Processor output must be a BatchFeature or mapping")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required provenance file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
