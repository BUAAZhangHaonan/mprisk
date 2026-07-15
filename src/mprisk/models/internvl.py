"""InternVL3.5 dynamic-visual prefill extraction."""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mprisk.models.base_wrapper import BaseModelWrapper, PrefillRequest, PrefillResult

LoadVideo = Callable[..., tuple[Any, list[int]]]


class InternVlWrapper(BaseModelWrapper):
    """Use InternVL's visual token contract and an explicit language-model forward."""

    family = "internvl"

    def __init__(
        self,
        *,
        model_key: str,
        model_path: str | Path,
        device: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        video_num_segments: int = 8,
        internvl_max_num: int = 1,
        model: Any | None = None,
        tokenizer: Any | None = None,
        load_video_fn: LoadVideo | None = None,
        runtime_versions: Mapping[str, str] | None = None,
        **_: Any,
    ) -> None:
        if video_num_segments <= 0:
            raise ValueError("video_num_segments must be positive")
        if internvl_max_num <= 0:
            raise ValueError("internvl_max_num must be positive")
        self.model_key = model_key
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.video_num_segments = video_num_segments
        self.internvl_max_num = internvl_max_num
        self._contract = _load_model_contract(self.model_path)
        if dtype != self._contract["torch_dtype"]:
            raise ValueError(
                f"Requested dtype {dtype!r} does not match model config "
                f"{self._contract['torch_dtype']!r}"
            )
        injected = (model, tokenizer, load_video_fn)
        if any(value is not None for value in injected) and not all(
            value is not None for value in injected
        ):
            raise ValueError("model, tokenizer, and load_video_fn must be injected together")
        self.model = model
        self.tokenizer = tokenizer
        self._load_video = load_video_fn or load_video
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

    @property
    def input_size(self) -> int:
        return int(self._contract["image_size"])

    def load(self) -> None:
        if self.model is not None:
            self._validate_loaded_contract()
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype_name),
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": self.device},
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        if self.model.__class__.__name__ != "InternVLChatModel":
            raise TypeError(f"Unexpected InternVL model class: {self.model.__class__.__name__}")
        self._validate_loaded_contract()

    def extract_prefill(self, request: PrefillRequest) -> PrefillResult:
        self._validate_request(request)
        if self.model is None:
            self.load()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("InternVL wrapper is not fully loaded")

        import torch

        started_at = time.perf_counter()
        question, pixel_values, num_patches_list = self._prepare_question(request)
        query = self._render_query(question, num_patches_list)
        model_inputs = self.tokenizer(query, return_tensors="pt")
        model_inputs = _move_inputs_to_device(model_inputs, self.device)
        input_ids = model_inputs.get("input_ids")
        attention_mask = _require_attention_mask(model_inputs)
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("InternVL extraction requires one two-dimensional input_ids tensor")
        token_count, t0_token_index = _token_position(attention_mask)

        track_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if track_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        with torch.inference_mode():
            inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
            if pixel_values is not None:
                pixel_values = pixel_values.to(
                    device=self.device,
                    dtype=getattr(torch, self.dtype_name),
                )
                vision_embeds = self.model.extract_feature(pixel_values).reshape(
                    -1, self.expected_hidden_dim
                )
                context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                self.model.img_context_token_id = context_token_id
                selected = input_ids == context_token_id
                if int(selected.sum().item()) != int(vision_embeds.shape[0]):
                    raise ValueError(
                        "InternVL visual token count does not match extracted visual embeddings: "
                        f"{int(selected.sum().item())} != {int(vision_embeds.shape[0])}"
                    )
                inputs_embeds[selected] = vision_embeds.to(dtype=inputs_embeds.dtype)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
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
        decord_version = (
            self._runtime_versions["decord"]
            if self._injected
            else importlib.metadata.version("decord")
        )
        return PrefillResult(
            request=request,
            trajectory=trajectory.detach().to(dtype=torch.float32, device="cpu").numpy(),
            token_count=token_count,
            t0_token_index=t0_token_index,
            provenance={
                "schema": "mprisk_internvl3_5_prefill_provenance_v1",
                "model_path": str(self.model_path),
                "model_class": self.model.__class__.__name__,
                "tokenizer_class": self.tokenizer.__class__.__name__,
                "transformers_version": transformers_version,
                "decord_version": decord_version,
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
                "video_num_segments": self.video_num_segments,
                "dynamic_image_max_num": self.internvl_max_num,
                "num_patches_list": num_patches_list,
                "language_forward_explicit": True,
                "chat_called": False,
            },
        )

    def close(self) -> None:
        if self._injected:
            return
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_question(self, request: PrefillRequest) -> tuple[str, Any | None, list[int]]:
        import torch

        text_parts: list[str] = []
        pixel_batches: list[Any] = []
        num_patches_list: list[int] = []
        visual_prefixes: list[str] = []
        image_index = 0
        for message in request.messages:
            for item in message.get("content", []):
                if not isinstance(item, Mapping):
                    raise ValueError("InternVL message content must contain mappings")
                content_type = str(item.get("type"))
                if content_type == "text":
                    text_parts.append(str(item.get("text", "")))
                elif content_type == "video":
                    path = _content_path(item, "video")
                    pixels, patch_counts = self._load_video(
                        path,
                        input_size=self.input_size,
                        max_num=self.internvl_max_num,
                        num_segments=self.video_num_segments,
                    )
                    pixel_batches.append(pixels)
                    num_patches_list.extend(int(value) for value in patch_counts)
                    visual_prefixes.extend(
                        f"Frame{index + 1}: <image>\n" for index in range(len(patch_counts))
                    )
                elif content_type == "image":
                    image_index += 1
                    path = _content_path(item, "image")
                    pixels = load_image(
                        path,
                        input_size=self.input_size,
                        max_num=max(self.internvl_max_num, 1),
                    )
                    pixel_batches.append(pixels)
                    num_patches_list.append(int(pixels.shape[0]))
                    visual_prefixes.append(f"Image{image_index}: <image>\n")
                else:
                    raise ValueError(f"InternVL VT request has unsupported content: {content_type}")
        text = "\n".join(part.strip() for part in text_parts if part.strip())
        if not text:
            raise ValueError("InternVL request requires non-empty text content")
        pixel_values = torch.cat(pixel_batches, dim=0) if pixel_batches else None
        if pixel_values is not None and int(pixel_values.shape[0]) != sum(num_patches_list):
            raise ValueError("InternVL dynamic patch counts do not match pixel tensor")
        return "".join(visual_prefixes) + text, pixel_values, num_patches_list

    def _render_query(self, question: str, num_patches_list: list[int]) -> str:
        if self.model is None:
            raise RuntimeError("InternVL model is not loaded")
        template = copy.deepcopy(self.model.conv_template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for patch_count in num_patches_list:
            image_tokens = (
                "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token * patch_count + "</img>"
            )
            query = query.replace("<image>", image_tokens, 1)
        if "<image>" in query:
            raise ValueError("InternVL query contains an unmatched image placeholder")
        return query

    def _validate_request(self, request: PrefillRequest) -> None:
        if request.model_key != self.model_key:
            raise ValueError(
                f"Request model_key {request.model_key!r} does not match {self.model_key!r}"
            )
        if request.protocol != "vt":
            raise ValueError("InternVL3.5 prefill extraction supports protocol VT only")
        if request.use_audio_in_video:
            raise ValueError("InternVL3.5 VT extraction must not enable audio from video")

    def _validate_loaded_contract(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        llm_config = getattr(getattr(self.model, "config", None), "llm_config", None)
        if llm_config is None:
            raise ValueError("InternVL config does not expose llm_config")
        actual = (int(llm_config.num_hidden_layers), int(llm_config.hidden_size))
        expected = (self.expected_layer_count, self.expected_hidden_dim)
        if actual != expected:
            raise ValueError(f"Loaded InternVL contract {actual} does not match config {expected}")


def build_transform(input_size: int) -> Any:
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def dynamic_preprocess(
    image: Any,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Any]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width,
        height,
        image_size,
    )
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    resized = image.resize((target_width, target_height))
    images = []
    for index in range(target_ratio[0] * target_ratio[1]):
        column_count = target_width // image_size
        box = (
            (index % column_count) * image_size,
            (index // column_count) * image_size,
            ((index % column_count) + 1) * image_size,
            ((index // column_count) + 1) * image_size,
        )
        images.append(resized.crop(box))
    if use_thumbnail and len(images) != 1:
        images.append(image.resize((image_size, image_size)))
    return images


def load_image(path: str, *, input_size: int = 448, max_num: int = 12) -> Any:
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    transform = build_transform(input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    return torch.stack([transform(item) for item in images])


def load_video(
    path: str,
    *,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 8,
) -> tuple[Any, list[int]]:
    import torch
    from decord import VideoReader, cpu
    from PIL import Image

    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    frame_indices = _video_frame_indices(
        fps=float(reader.get_avg_fps()),
        max_frame=len(reader) - 1,
        num_segments=num_segments,
    )
    transform = build_transform(input_size)
    batches = []
    patch_counts = []
    for frame_index in frame_indices:
        image = Image.fromarray(reader[int(frame_index)].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )
        pixels = torch.stack([transform(tile) for tile in tiles])
        batches.append(pixels)
        patch_counts.append(int(pixels.shape[0]))
    return torch.cat(batches, dim=0), patch_counts


def _video_frame_indices(*, fps: float, max_frame: int, num_segments: int) -> np.ndarray:
    start_idx = 0
    end_idx = min(round(100000 * fps), max_frame)
    segment_size = float(end_idx - start_idx) / num_segments
    return np.array(
        [
            int(start_idx + segment_size / 2 + np.round(segment_size * index))
            for index in range(num_segments)
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_difference = math.inf
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best_ratio = ratio
        elif difference == best_difference:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _load_model_contract(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"InternVL config is missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "internvl_chat":
        raise ValueError(f"Unexpected model_type in {config_path}: {payload.get('model_type')!r}")
    if payload.get("architectures") != ["InternVLChatModel"]:
        raise ValueError(f"Unexpected InternVL architecture in {config_path}")
    llm = payload.get("llm_config")
    if not isinstance(llm, dict) or llm.get("model_type") != "qwen3":
        raise ValueError(f"InternVL Qwen3 llm_config is missing from {config_path}")
    contract = {
        "num_hidden_layers": int(llm["num_hidden_layers"]),
        "hidden_size": int(llm["hidden_size"]),
        "torch_dtype": str(
            payload.get("dtype")
            or payload.get("torch_dtype")
            or llm.get("dtype")
            or llm.get("torch_dtype")
            or ""
        ),
        "image_size": int(payload.get("force_image_size") or 448),
    }
    if contract["num_hidden_layers"] <= 0 or contract["hidden_size"] <= 0:
        raise ValueError(f"Invalid InternVL language dimensions: {contract}")
    return contract


def _content_path(item: Mapping[str, Any], content_type: str) -> str:
    value = item.get(content_type) or item.get("path") or item.get("url")
    if not isinstance(value, str) or not value:
        raise ValueError(f"InternVL {content_type} content requires a path")
    return value


def _move_inputs_to_device(model_inputs: Any, device: str) -> Any:
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    if not isinstance(model_inputs, Mapping):
        raise TypeError("InternVL tokenizer output must be a mapping")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def _require_attention_mask(model_inputs: Any) -> Any:
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None or attention_mask.ndim != 2 or int(attention_mask.shape[0]) != 1:
        raise ValueError("InternVL extraction requires one two-dimensional attention_mask")
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
            f"Expected InternVL trajectory shape {(layer_count, hidden_dim)}, "
            f"got {tuple(trajectory.shape)}"
        )
    if not torch.isfinite(trajectory).all().item():
        raise ValueError("InternVL trajectory contains non-finite values")
    return trajectory


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required provenance file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
