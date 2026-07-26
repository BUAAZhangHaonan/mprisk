"""Phi-4 Multimodal VA prefill extraction.

Phi-4 does not accept an MP4 as a native video input.  The wrapper therefore
converts the visual stream to a deterministic, uniformly sampled image sequence
and passes audio through the model's native speech processor.  The three VA
conditions are image sequence only (M1), audio only (M2), and both (M12).
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from mprisk.models.base_wrapper import BaseModelWrapper, PrefillRequest, PrefillResult


class Phi4MmWrapper(BaseModelWrapper):
    """Extract all Phi-4 language-block states at the first reply position."""

    family = "phi4_multimodal"
    required_transformers_version = "4.48.2"
    required_peft_version = "0.13.2"

    def __init__(
        self,
        *,
        model_key: str,
        model_path: str | Path,
        device: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "eager",
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
        self.video_num_segments = int(video_num_segments)
        if not 1 <= self.video_num_segments <= 64:
            raise ValueError("Phi-4 video_num_segments must be in [1, 64]")
        if dtype != "bfloat16":
            raise ValueError("Phi-4 cache extraction requires bfloat16 inference")
        if attn_implementation not in {"eager", "sdpa"}:
            raise ValueError("Phi-4 attention implementation must be eager or sdpa")
        self._contract = _load_model_contract(self.model_path)
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be injected together")
        self.model = model
        self.processor = processor
        self._injected = model is not None
        if self._injected and runtime_versions is None:
            raise ValueError("Injected Phi-4 dependencies require explicit runtime_versions")
        if not self._injected and runtime_versions is not None:
            raise ValueError("runtime_versions is only valid for injected dependencies")
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
        import peft
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        if transformers.__version__ != self.required_transformers_version:
            raise RuntimeError(
                "Phi-4 requires the pinned transformers runtime "
                f"{self.required_transformers_version}, got {transformers.__version__}"
            )
        if peft.__version__ != self.required_peft_version:
            raise RuntimeError(
                f"Phi-4 requires peft {self.required_peft_version}, got {peft.__version__}"
            )
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)

        phi4_model = get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMModel",
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        compatibility_patch = False
        if not hasattr(phi4_model, "prepare_inputs_for_generation"):

            def prepare_inputs_for_generation(
                instance: Any, input_ids: Any, **kwargs: Any
            ) -> dict[str, Any]:
                del instance
                return {"input_ids": input_ids, **kwargs}

            phi4_model.prepare_inputs_for_generation = prepare_inputs_for_generation
            compatibility_patch = True

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        config._attn_implementation = self.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
            low_cpu_mem_usage=False,
            device_map=None,
        ).eval().to(self.device)
        meta_parameters = [
            name
            for name, parameter in self.model.named_parameters()
            if str(parameter.device) == "meta"
        ]
        if meta_parameters:
            raise RuntimeError(
                "Phi-4 load left meta parameters: " + ", ".join(meta_parameters[:20])
            )
        self._compatibility_patch = compatibility_patch
        self._runtime_versions = {
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "torch": torch.__version__,
        }
        self._validate_loaded_contract()

    def extract_prefill(self, request: PrefillRequest) -> PrefillResult:
        self._validate_request(request)
        if self.model is None:
            self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError("Phi-4 wrapper is not fully loaded")

        import torch

        started_at = time.perf_counter()
        prompt, images, audios, media_provenance = self._prepare_modal_inputs(request)
        model_inputs = self.processor(
            text=prompt,
            images=images or None,
            audios=audios or None,
            return_tensors="pt",
        )
        _validate_processor_modes(request, model_inputs, images=images, audios=audios)
        model_inputs = _move_inputs_to_device(model_inputs, self.device)
        attention_mask = _require_attention_mask(model_inputs)
        token_count, t0_token_index = _token_position(attention_mask)

        track_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if track_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
                num_logits_to_keep=1,
            )
        trajectory = _trajectory_from_outputs(
            outputs,
            t0_token_index=t0_token_index,
            layer_count=self.expected_layer_count,
            hidden_dim=self.expected_hidden_dim,
        )
        peak_gpu_bytes = (
            int(torch.cuda.max_memory_allocated(torch.device(self.device)))
            if track_cuda
            else None
        )
        return PrefillResult(
            request=request,
            trajectory=trajectory.detach().to(dtype=torch.float32, device="cpu").numpy(),
            token_count=token_count,
            t0_token_index=t0_token_index,
            provenance={
                "schema": "mprisk_phi4_multimodal_prefill_provenance_v1",
                "model_path": str(self.model_path),
                "model_class": self.model.__class__.__name__,
                "processor_class": self.processor.__class__.__name__,
                "transformers_version": self._runtime_versions.get("transformers", "injected"),
                "peft_version": self._runtime_versions.get("peft", "injected"),
                "torch_version": self._runtime_versions.get("torch", torch.__version__),
                "source_dtype": self.dtype_name,
                "stored_dtype": "float32",
                "device": self.device,
                "attn_implementation": self.attn_implementation,
                "num_hidden_layers": self.expected_layer_count,
                "hidden_size": self.expected_hidden_dim,
                "hidden_state_index_offset": 1,
                "model_config_sha256": _sha256(self.model_path / "config.json"),
                "weight_index_sha256": _sha256(self.model_path / "model.safetensors.index.json"),
                "video_sampling": media_provenance,
                "video_sampling_method": media_provenance["method"],
                "requested_frames": media_provenance["requested_frames"],
                "actual_frames": media_provenance["actual_frames"],
                "video_frame_indices": media_provenance["video_frame_indices"],
                "video_source_total_frames": media_provenance[
                    "video_source_total_frames"
                ],
                "peft_compatibility_patch": bool(getattr(self, "_compatibility_patch", False)),
                "elapsed_seconds": time.perf_counter() - started_at,
                "peak_gpu_memory_bytes": peak_gpu_bytes,
            },
        )

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _validate_request(self, request: PrefillRequest) -> None:
        if request.model_key != self.model_key:
            raise ValueError("Phi-4 request model_key mismatch")
        if request.protocol != "va":
            raise ValueError("Phi-4 Multimodal cache extraction supports VA only")
        if request.condition not in {"M1", "M2", "M12"}:
            raise ValueError(f"Unsupported Phi-4 condition: {request.condition}")

    def _validate_loaded_contract(self) -> None:
        config = getattr(self.model, "config", None)
        if config is None:
            raise ValueError("Loaded Phi-4 model has no config")
        actual = (int(config.num_hidden_layers), int(config.hidden_size))
        expected = (self.expected_layer_count, self.expected_hidden_dim)
        if actual != expected:
            raise ValueError(f"Loaded Phi-4 contract {actual} does not match config {expected}")

    def _prepare_modal_inputs(
        self, request: PrefillRequest
    ) -> tuple[str, list[Any], list[tuple[np.ndarray, int]], dict[str, Any]]:
        text_parts: list[str] = []
        images: list[Any] = []
        audios: list[tuple[np.ndarray, int]] = []
        vision_sources: list[str] = []
        video_sources: list[str] = []
        video_metadata: list[dict[str, Any]] = []
        audio_sources: list[str] = []
        actual_video_frames = 0
        for message in request.messages:
            if message.get("role") != "user":
                raise ValueError("Phi-4 extraction accepts exactly one user turn")
            content = message.get("content")
            if not isinstance(content, Sequence) or isinstance(content, str | bytes):
                raise TypeError("Phi-4 message content must be a content-item sequence")
            for item in content:
                if not isinstance(item, Mapping):
                    raise TypeError("Phi-4 content items must be mappings")
                item_type = str(item.get("type"))
                if item_type == "text":
                    text_parts.append(str(item.get("text", "")))
                elif item_type == "image":
                    path = _required_media_path(item.get("image"), "image")
                    images.append(_load_image(path))
                    vision_sources.append(path)
                elif item_type == "video":
                    path = _required_media_path(item.get("video"), "video")
                    frames, metadata = _uniform_video_sample_ffmpeg(
                        path, self.video_num_segments
                    )
                    if len(frames) != self.video_num_segments:
                        raise ValueError(
                            f"Phi-4 requested {self.video_num_segments} video frames "
                            f"but decoder returned {len(frames)}"
                        )
                    images.extend(frames)
                    actual_video_frames += len(frames)
                    vision_sources.append(path)
                    video_sources.append(path)
                    video_metadata.append(metadata)
                    if request.use_audio_in_video:
                        audios.append(_decode_audio(path))
                        audio_sources.append(path)
                elif item_type == "audio":
                    path = _required_media_path(item.get("audio"), "audio")
                    audios.append(_decode_audio(path))
                    audio_sources.append(path)
                else:
                    raise ValueError(f"Unsupported Phi-4 content type: {item_type!r}")
        task_text = "\n".join(part.strip() for part in text_parts if part.strip())
        if not task_text:
            raise ValueError("Phi-4 request has no task text")
        image_tokens = "".join(f"<|image_{index}|>" for index in range(1, len(images) + 1))
        audio_tokens = "".join(f"<|audio_{index}|>" for index in range(1, len(audios) + 1))
        prompt = f"<|user|>{image_tokens}{audio_tokens}{task_text}<|end|><|assistant|>"
        return prompt, images, audios, {
            "method": "uniform_midpoint_ffmpeg_v1" if actual_video_frames else None,
            "frame_count": len(images),
            "requested_frames": self.video_num_segments * len(video_sources)
            if actual_video_frames
            else 0,
            "actual_frames": actual_video_frames,
            "video_frame_indices": [
                row["frames_indices"] for row in video_metadata
            ],
            "video_source_total_frames": [
                row["total_num_frames"] for row in video_metadata
            ],
            "audio_count": len(audios),
            "vision_sources": vision_sources,
            "audio_sources": audio_sources,
        }


def _load_model_contract(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architectures") != ["Phi4MMForCausalLM"]:
        raise ValueError("Phi-4 config architecture is not Phi4MMForCausalLM")
    layers = int(config.get("num_hidden_layers", 0))
    hidden = int(config.get("hidden_size", 0))
    if layers <= 0 or hidden <= 0:
        raise ValueError("Phi-4 config must define positive layer and hidden dimensions")
    return {"num_hidden_layers": layers, "hidden_size": hidden}


def _required_media_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Phi-4 {label} content requires a local path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _load_image(path: str) -> Any:
    from PIL import Image
    with Image.open(path) as image:
        return image.convert("RGB")


def _uniform_video_sample_ffmpeg(
    path: str, count: int
) -> tuple[list[Any], dict[str, Any]]:
    """Decode exact midpoint frame indices with Phi-4's ffmpeg backend."""
    from PIL import Image

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    total_frames = int(stream["nb_read_frames"])
    fps = float(Fraction(str(stream["avg_frame_rate"])))
    if width <= 0 or height <= 0 or total_frames < count or fps <= 0:
        raise ValueError(
            f"Invalid ffprobe video contract for {path}: "
            f"{width=} {height=} {total_frames=} {fps=}"
        )
    indices = [
        min(total_frames - 1, int((index + 0.5) * total_frames / count))
        for index in range(count)
    ]
    if indices != sorted(set(indices)):
        raise ValueError(f"Midpoint frame indices are not unique: {indices}")
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-noautorotate",
            "-i",
            path,
            "-vf",
            f"select={select}",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    expected_bytes = count * height * width * 3
    if len(decoded.stdout) != expected_bytes:
        raise ValueError(
            f"ffmpeg returned {len(decoded.stdout)} bytes; expected {expected_bytes}"
        )
    array = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(
        count, height, width, 3
    )
    frames = [Image.fromarray(frame.copy()).convert("RGB") for frame in array]
    return frames, {
        "total_num_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": total_frames / fps,
        "video_backend": "ffmpeg",
        "frames_indices": indices,
    }


def _decode_audio(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    completed = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vn",
            "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    waveform, sample_rate = sf.read(io.BytesIO(completed.stdout), dtype="float32")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Decoded Phi-4 audio is invalid: {path}")
    if int(sample_rate) != 16000:
        raise ValueError(f"Decoded Phi-4 audio rate is not 16 kHz: {sample_rate}")
    return waveform, int(sample_rate)


def _validate_processor_modes(
    request: PrefillRequest,
    model_inputs: Mapping[str, Any],
    *,
    images: Sequence[Any],
    audios: Sequence[Any],
) -> None:
    expected_mode = {"M1": 1, "M2": 2, "M12": 3}[request.condition]
    input_mode = model_inputs.get("input_mode")
    if input_mode is None or int(input_mode.flatten()[0].item()) != expected_mode:
        raise ValueError(
            f"Phi-4 processor input_mode mismatch for {request.condition}: expected {expected_mode}"
        )
    if bool(images) != (int(model_inputs.get("input_image_embeds").numel()) > 0):
        raise ValueError("Phi-4 processor image tensor does not match request")
    if bool(audios) != (int(model_inputs.get("input_audio_embeds").numel()) > 0):
        raise ValueError("Phi-4 processor audio tensor does not match request")


def _move_inputs_to_device(model_inputs: Any, device: str) -> Any:
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def _require_attention_mask(model_inputs: Mapping[str, Any]) -> Any:
    mask = model_inputs.get("attention_mask")
    if mask is None or mask.ndim != 2 or int(mask.shape[0]) != 1:
        raise ValueError("Phi-4 processor must return a batch-one attention_mask")
    return mask


def _token_position(attention_mask: Any) -> tuple[int, int]:
    indices = attention_mask[0].nonzero(as_tuple=False).flatten()
    if int(indices.numel()) == 0:
        raise ValueError("Phi-4 attention mask has no valid token")
    return int(attention_mask.shape[1]), int(indices[-1].item())


def _trajectory_from_outputs(
    outputs: Any, *, t0_token_index: int, layer_count: int, hidden_dim: int
) -> Any:
    import torch

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) != layer_count + 1:
        raise ValueError(
            f"Phi-4 expected embedding plus {layer_count} block states, got "
            f"{None if hidden_states is None else len(hidden_states)}"
        )
    trajectory = torch.stack(
        [state[0, t0_token_index, :] for state in hidden_states[1:]], dim=0
    )
    if tuple(trajectory.shape) != (layer_count, hidden_dim):
        raise ValueError(f"Unexpected Phi-4 trajectory shape: {tuple(trajectory.shape)}")
    if not bool(torch.isfinite(trajectory).all()):
        raise ValueError("Phi-4 trajectory contains non-finite values")
    return trajectory


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
