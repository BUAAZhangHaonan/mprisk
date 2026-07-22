"""Gemma-4 12B Unified prefill extraction for VA (video+audio) protocol."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import tempfile
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mprisk.models.base_wrapper import (
    BaseModelWrapper,
    PrefillRequest,
    PrefillResult,
)
from mprisk.models.video_frame_utils import uniform_video_sample

DEFAULT_VIDEO_FRAMES: int = 8
_FFMPEG_AVAILABLE: bool = shutil.which("ffmpeg") is not None


class Gemma4Wrapper(BaseModelWrapper):
    """Gemma-4 12B Unified multimodal prefill extraction.

    Supports protocol VA only. For VA, conditioning views map to:
      M1  = video-only (vision frames) + task prompt
      M2  = audio-only (16kHz mono wav) + task prompt
      M12 = video with embedded audio + task prompt
    """

    family = "gemma4"

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
        video_num_segments: int = DEFAULT_VIDEO_FRAMES,
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
            raise ValueError("Gemma-4 video_num_segments must be in [1, 64]")
        self._contract = _load_model_contract(self.model_path)
        if dtype != self._contract["torch_dtype"]:
            raise ValueError(
                f"Requested dtype {dtype!r} does not match model config "
                f"{self._contract['torch_dtype']!r}"
            )
        self.model = model
        self.processor = processor
        self._injected = model is not None
        if self._injected and (processor is None or runtime_versions is None):
            raise ValueError(
                "Injected model requires processor and explicit runtime_versions"
            )
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
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
        )

        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, **processor_kwargs
        )
        if type(self.processor).__name__ != "Gemma4UnifiedProcessor":
            raise TypeError(
                f"Expected Gemma4UnifiedProcessor, got {type(self.processor).__name__}"
            )
        # Left-padding is required so the t0 token sits at the rightmost column.
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise TypeError("Gemma-4 processor does not expose a tokenizer")
        tokenizer.padding_side = "left"
        dtype = getattr(torch, self.dtype_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            attn_implementation=self.attn_implementation,
            local_files_only=True,
            trust_remote_code=True,
            device_map={"": self.device},
        ).eval()
        if self.model.__class__.__name__ != "Gemma4UnifiedForConditionalGeneration":
            raise TypeError(
                f"Unexpected Gemma-4 model class: {self.model.__class__.__name__}"
            )
        self._validate_loaded_contract()

    def extract_prefill(self, request: PrefillRequest) -> PrefillResult:
        self._validate_request(request)
        if self.model is None:
            self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError("Gemma-4 wrapper is not fully loaded")

        import torch

        started_at = time.perf_counter()
        prompt = self.processor.apply_chat_template(
            list(request.messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        media = _collect_media_inputs(request, max_frames=self.video_num_segments)
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        # Prefer raw waveforms (M12 case) over file paths; ensures alignment
        # with the video frames extracted from the same source.
        if media.get("audio_waveforms"):
            processor_kwargs["audio"] = [w for w, _ in media["audio_waveforms"]]
            processor_kwargs["sampling_rate"] = media["audio_waveforms"][0][1]
        elif media.get("audio") is not None:
            processor_kwargs["audio"] = media["audio"]
        if media["videos"] is not None:
            processor_kwargs["videos"] = media["videos"]
            processor_kwargs["video_metadata"] = media["video_metadata"]
            processor_kwargs["do_sample_frames"] = False
            # We pre-decode via PyAV and pass numpy frame stacks, so tell the
            # video processor exactly how many frames we have to avoid the
            # default 32-frame sampler exceeding the supplied frame count.
            frame_counts = [int(v.shape[0]) for v in media["videos"]]
            processor_kwargs["num_frames"] = frame_counts[0]
            actual_frames = sum(frame_counts)
            if actual_frames != self.video_num_segments:
                raise ValueError(
                    f"Gemma-4 requested {self.video_num_segments} video frames but "
                    f"decoder returned {actual_frames}"
                )
        else:
            actual_frames = 0
        if media["images"] is not None:
            processor_kwargs["images"] = media["images"]
        _validate_media_contract(request.condition, media)
        try:
            model_inputs = self.processor(**processor_kwargs)
        finally:
            _cleanup_temporary_paths(media["temporary_paths"])
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
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        expected_state_count = self.expected_layer_count + 1
        if hidden_states is None or len(hidden_states) != expected_state_count:
            actual = None if hidden_states is None else len(hidden_states)
            raise ValueError(
                f"Expected {expected_state_count} hidden-state tensors, got {actual}"
            )
        trajectory = torch.stack(
            [state[0, t0_token_index, :] for state in hidden_states[1:]],
            dim=0,
        )
        expected_shape = (self.expected_layer_count, self.expected_hidden_dim)
        if tuple(trajectory.shape) != expected_shape:
            raise ValueError(
                f"Expected Gemma-4 trajectory shape {expected_shape}, "
                f"got {tuple(trajectory.shape)}"
            )
        if not torch.isfinite(trajectory).all().item():
            raise ValueError("Gemma-4 trajectory contains non-finite values")
        peak_gpu_bytes = (
            int(torch.cuda.max_memory_allocated(torch.device(self.device)))
            if track_cuda
            else None
        )
        elapsed_seconds = time.perf_counter() - started_at
        stored_trajectory = (
            trajectory.detach().to(dtype=torch.float32, device="cpu").numpy()
        )
        if self._injected:
            transformers_version = self._runtime_versions["transformers"]
        else:
            import transformers

            transformers_version = transformers.__version__
        provenance = {
            "schema": "mprisk_gemma_4_prefill_provenance_v1",
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
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "media_keys": _media_keys(media),
            "video_sampling_method": (
                "uniform_midpoint_decord_v1" if actual_frames else None
            ),
            "requested_frames": self.video_num_segments if actual_frames else 0,
            "actual_frames": actual_frames,
            "video_frame_indices": (
                [row["frames_indices"] for row in media["video_metadata"]]
                if actual_frames
                else []
            ),
            "video_source_total_frames": (
                [row["total_num_frames"] for row in media["video_metadata"]]
                if actual_frames
                else []
            ),
        }
        return PrefillResult(
            request=request,
            trajectory=np.asarray(stored_trajectory),
            token_count=token_count,
            t0_token_index=t0_token_index,
            provenance=provenance,
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
                f"Request model_key {request.model_key!r} does not match "
                f"{self.model_key!r}"
            )
        if request.protocol != "va":
            raise ValueError(
                "Gemma-4 wrapper supports protocol VA only; "
                f"got {request.protocol!r}"
            )
        if request.condition not in {"M1", "M2", "M12"}:
            raise ValueError(f"Unsupported Gemma-4 VA condition: {request.condition!r}")
        if request.condition == "M12" and not request.use_audio_in_video:
            raise ValueError("Gemma-4 M12 requires embedded-video audio")
        if request.condition != "M12" and request.use_audio_in_video:
            raise ValueError("Only Gemma-4 M12 may enable embedded-video audio")
        content_types = {
            str(item.get("type"))
            for message in request.messages
            for item in message.get("content", [])
            if isinstance(item, Mapping)
        }
        expected_types = {
            "M1": {"video", "text"},
            "M2": {"audio", "text"},
            "M12": {"video", "audio", "text"},
        }[request.condition]
        if content_types != expected_types:
            raise ValueError(
                f"Gemma-4 {request.condition} content types {sorted(content_types)} "
                f"do not match {sorted(expected_types)}"
            )

    def _validate_loaded_contract(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        text_config = getattr(getattr(self.model, "config", None), "text_config", None)
        if text_config is None:
            raise ValueError("Gemma-4 config does not expose text_config")
        actual = (int(text_config.num_hidden_layers), int(text_config.hidden_size))
        expected = (self.expected_layer_count, self.expected_hidden_dim)
        if actual != expected:
            raise ValueError(f"Loaded Gemma-4 contract {actual} does not match config {expected}")


def build_va_request(
    *,
    sample_id: str,
    model_key: str,
    dataset_key: str,
    split: str,
    media_paths: Mapping[str, str],
    text_content: str,
    task_prompt: str,
    condition: str,
    prompt_set_key: str = "adhoc",
    prompt_id: str = "adhoc",
) -> PrefillRequest:
    """Build one VA conditioning view for Gemma-4 Unified.

    Conditions (mainline VA protocol, no text):
      M1  = video-only (vision frames) + task prompt
      M2  = audio-only (16kHz mono wav) + task prompt
      M12 = video with embedded audio + task prompt
    """
    condition = condition.upper()
    if condition not in {"M1", "M2", "M12"}:
        raise ValueError(f"Unsupported VA condition: {condition!r}")
    vision_path = media_paths.get("vision")
    audio_path = media_paths.get("audio")
    if not vision_path:
        raise ValueError("Gemma-4 VA requires media_paths.vision")
    if not audio_path:
        raise ValueError("Gemma-4 VA requires media_paths.audio")
    prompt = task_prompt.strip()
    if not prompt:
        raise ValueError("task_prompt must not be empty")

    content: list[dict[str, Any]] = []
    if condition == "M1":
        # Vision-only: video frames (no audio path passed to processor)
        content.append({"type": "video", "video": vision_path})
    elif condition == "M2":
        # Audio-only: 16kHz mono wav re-encoded from the source
        content.append({"type": "audio", "audio": audio_path})
    else:  # M12
        # Joint vision + audio: pass both video frames and the audio track.
        # Gemma-4 Unified processor accepts them as separate inputs and
        # fuses via its internal multimodal attention.
        content.append({"type": "video", "video": vision_path})
        content.append({"type": "audio", "audio": audio_path})
    content.append({"type": "text", "text": prompt})

    return PrefillRequest(
        sample_id=sample_id,
        model_key=model_key,
        protocol="va",
        condition=condition,
        dataset_key=dataset_key,
        split=split,
        messages=({"role": "user", "content": content},),
        media_paths=media_paths,
        use_audio_in_video=(condition == "M12"),
        prompt_set_key=prompt_set_key,
        prompt_id=prompt_id,
    )


def _collect_media_inputs(
    request: PrefillRequest,
    *,
    max_frames: int = DEFAULT_VIDEO_FRAMES,
) -> dict[str, Any]:
    """Walk request.messages and load audio/video inputs for the processor.

    For joint V+A conditioning (use_audio_in_video=True), we decode the video's
    own audio track via PyAV and pass it as the audio input alongside the video
    frames. This keeps token/feature counts aligned (processor treats the audio
    as the video's joint audio track).
    """
    audio_paths: list[str] = []
    audio_waveforms: list[Any] = []  # raw np.float32 mono arrays
    video_frames: list[Any] = []
    video_metadata: list[dict[str, Any]] = []
    image_paths: list[str] = []
    temporary_paths: list[str] = []

    for message in request.messages:
        for item in message.get("content", []):
            if not isinstance(item, Mapping):
                continue
            ctype = str(item.get("type"))
            if ctype == "audio" and request.use_audio_in_video:
                # M12 path: this audio came from build_va_request for the same
                # video file. Decode audio directly from the source video instead
                # so token counts match.
                continue  # handled via _video_to_frames_with_audio below
            if ctype == "audio":
                source = str(item.get("audio"))
                converted = _audio_to_wav(source)
                audio_paths.append(converted)
                if converted != source:
                    temporary_paths.append(converted)
            elif ctype == "video":
                vp = str(item.get("video"))
                if request.use_audio_in_video:
                    frames, audio_arr, sr, metadata = _video_to_frames_with_audio(
                        vp, max_frames=max_frames
                    )
                    video_frames.append(frames)
                    video_metadata.append(metadata)
                    if audio_arr is not None:
                        audio_waveforms.append((audio_arr, sr))
                else:
                    frames, metadata = _video_to_frames(vp, max_frames=max_frames)
                    video_frames.append(frames)
                    video_metadata.append(metadata)
            elif ctype == "image":
                image_paths.append(str(item.get("image")))

    media: dict[str, Any] = {
        "audio": None,
        "videos": None,
        "video_metadata": None,
        "images": None,
        "audio_waveforms": None,
        "temporary_paths": temporary_paths,
    }
    if audio_paths:
        media["audio"] = audio_paths
    if audio_waveforms:
        media["audio_waveforms"] = audio_waveforms
    if video_frames:
        media["videos"] = video_frames
        media["video_metadata"] = video_metadata
    if image_paths:
        media["images"] = image_paths
    return media


def _audio_to_wav(source_path: str) -> str:
    """Return a path to a 16 kHz mono wav; re-encode via ffmpeg if needed.

    ffmpeg is required because librosa (the processor's audio loader) refuses
    to read mp4 containers, and our VA samples are mp4 files with embedded
    audio.
    """
    ext = Path(source_path).suffix.lower()
    if ext in {".wav", ".flac", ".mp3", ".ogg", ".opus"}:
        return source_path
    if not _FFMPEG_AVAILABLE:
        raise RuntimeError(
            f"Audio source {source_path!r} is a video container and ffmpeg is "
            "required to extract a wav; install ffmpeg on PATH."
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", source_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-loglevel", "error",
            tmp.name,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return tmp.name


def _video_to_frames(
    video_path: str,
    *,
    max_frames: int = DEFAULT_VIDEO_FRAMES,
) -> tuple[Any, dict[str, Any]]:
    """Uniformly sample exactly ``max_frames`` frames with shared provenance."""
    images, metadata = uniform_video_sample(video_path, max_frames)
    frames = np.stack([np.asarray(image, dtype=np.uint8) for image in images])
    return frames, metadata


def _video_to_frames_with_audio(
    video_path: str,
    *,
    max_frames: int = DEFAULT_VIDEO_FRAMES,
) -> tuple[Any, Any, int, dict[str, Any]]:
    """Decode timestamped video frames and audio using independent containers."""
    import av
    import numpy as np

    frames, metadata = _video_to_frames(video_path, max_frames=max_frames)
    audio_container = av.open(video_path)
    try:
        if not audio_container.streams.audio:
            raise ValueError(f"Video has no audio stream: {video_path}")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks: list[Any] = []
        for frame in audio_container.decode(audio=0):
            chunks.extend(resampler.resample(frame))
        chunks.extend(resampler.resample(None))
    finally:
        audio_container.close()
    if not chunks:
        raise ValueError(f"Video audio stream yielded no samples: {video_path}")
    arrays = [frame.to_ndarray().reshape(-1) for frame in chunks]
    audio = np.concatenate(arrays).astype(np.float32) / 32768.0
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Decoded audio is empty or non-finite: {video_path}")
    return frames, audio, 16000, metadata


def _validate_media_contract(condition: str, media: Mapping[str, Any]) -> None:
    has_video = bool(media.get("videos"))
    has_audio = bool(media.get("audio")) or bool(media.get("audio_waveforms"))
    if has_video and not media.get("video_metadata"):
        raise ValueError("Gemma-4 video input requires source video metadata")
    if condition == "M1" and (not has_video or has_audio):
        raise ValueError("Gemma-4 M1 must contain video and no audio")
    if condition == "M2" and (has_video or not has_audio):
        raise ValueError("Gemma-4 M2 must contain audio and no video")
    if condition == "M12" and (not has_video or not has_audio):
        raise ValueError("Gemma-4 M12 must contain both video and audio")


def _media_keys(media: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    if media.get("videos"):
        keys.append("videos")
    if media.get("audio") or media.get("audio_waveforms"):
        keys.append("audio")
    if media.get("images"):
        keys.append("images")
    return keys


def _cleanup_temporary_paths(paths: list[str]) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            path.unlink()


def _load_model_contract(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Gemma-4 config is missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "gemma4_unified":
        raise ValueError(
            f"Unexpected model_type in {config_path}: {payload.get('model_type')!r}"
        )
    text = payload.get("text_config")
    if not isinstance(text, dict):
        raise ValueError(f"Gemma-4 text_config missing from {config_path}")
    contract = {
        "num_hidden_layers": int(text["num_hidden_layers"]),
        "hidden_size": int(text["hidden_size"]),
        "torch_dtype": str(payload.get("dtype") or payload.get("torch_dtype") or ""),
    }
    if contract["num_hidden_layers"] <= 0 or contract["hidden_size"] <= 0:
        raise ValueError(f"Invalid Gemma-4 dimensions: {contract}")
    if not contract["torch_dtype"]:
        raise ValueError(f"Gemma-4 dtype missing from {config_path}")
    return contract


def _move_inputs_to_device(model_inputs: Any, device: str) -> Any:
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    if not isinstance(model_inputs, Mapping):
        raise TypeError("Gemma-4 processor output must be a BatchFeature or mapping")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def _require_attention_mask(model_inputs: Any) -> Any:
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        raise ValueError("Gemma-4 processor output must include attention_mask")
    if attention_mask.ndim != 2 or int(attention_mask.shape[0]) != 1:
        raise ValueError("Gemma-4 prefill extraction requires batch size exactly one")
    return attention_mask


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required provenance file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
