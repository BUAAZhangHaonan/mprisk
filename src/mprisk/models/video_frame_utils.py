"""Deterministic video-to-frame preparation shared by frame-based VT wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mprisk.models.base_wrapper import PrefillRequest


def request_text_and_frames(
    request: PrefillRequest,
    *,
    video_num_segments: int,
) -> tuple[str, list[Any], dict[str, Any]]:
    """Resolve ordered text and visual items without changing the request semantics."""
    from PIL import Image

    if not 1 <= video_num_segments <= 64:
        raise ValueError("video_num_segments must be in [1, 64]")
    text_parts: list[str] = []
    images: list[Image.Image] = []
    video_sources: list[str] = []
    visual_input_types: list[str] = []
    for message in request.messages:
        if str(message.get("role")) != "user":
            raise ValueError("Frame-based VT wrappers support one or more user messages only")
        for item in message.get("content", []):
            if not isinstance(item, Mapping):
                raise TypeError("Multimodal content items must be mappings")
            item_type = str(item.get("type"))
            if item_type == "text":
                text_parts.append(str(item.get("text", "")))
            elif item_type == "image":
                path = required_media_path(item.get("image"), "image")
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
                visual_input_types.append("image")
            elif item_type == "video":
                path = required_media_path(item.get("video"), "video")
                frames = uniform_video_frames(path, video_num_segments)
                images.extend(frames)
                video_sources.append(path)
                visual_input_types.extend("video_frame" for _ in frames)
            else:
                raise ValueError(f"Unsupported frame-based VT content type: {item_type!r}")
    text = "\n".join(part for part in text_parts if part).strip()
    if not text:
        raise ValueError("Frame-based VT request has no text instruction")
    return text, images, {
        "visual_input_types": visual_input_types,
        "video_sampling_method": (
            "uniform_midpoint_decord_v1" if video_sources else None
        ),
        "video_frame_count": len(images),
        "video_num_segments": video_num_segments,
        "video_sources": video_sources,
    }


def required_media_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} content requires a local path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def uniform_video_frames(path: str, count: int) -> list[Any]:
    import decord
    from PIL import Image

    reader = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
    length = len(reader)
    if length <= 0:
        raise ValueError(f"Video has no frames: {path}")
    indices = [
        min(length - 1, int((index + 0.5) * length / count))
        for index in range(count)
    ]
    array = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in array]
