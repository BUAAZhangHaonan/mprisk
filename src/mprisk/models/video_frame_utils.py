"""Deterministic, fail-closed video sampling shared by model wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mprisk.models.base_wrapper import PrefillRequest

UNIFORM_VIDEO_SAMPLING = "uniform_midpoint_decord_v1"


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
    video_metadata: list[dict[str, Any]] = []
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
                frames, metadata = uniform_video_sample(path, video_num_segments)
                images.extend(frames)
                video_sources.append(path)
                video_metadata.append(metadata)
                visual_input_types.extend("video_frame" for _ in frames)
            else:
                raise ValueError(f"Unsupported frame-based VT content type: {item_type!r}")
    text = "\n".join(part for part in text_parts if part).strip()
    if not text:
        raise ValueError("Frame-based VT request has no text instruction")
    return text, images, {
        "visual_input_types": visual_input_types,
        "video_sampling_method": UNIFORM_VIDEO_SAMPLING if video_sources else None,
        "video_frame_count": len(images),
        "video_num_segments": video_num_segments,
        "video_sources": video_sources,
        "requested_frames": video_num_segments * len(video_sources),
        "actual_frames": video_num_segments * len(video_sources),
        "video_frame_indices": [row["frames_indices"] for row in video_metadata],
        "video_source_total_frames": [
            row["total_num_frames"] for row in video_metadata
        ],
    }


def required_media_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} content requires a local path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def uniform_video_frames(path: str, count: int) -> list[Any]:
    frames, _ = uniform_video_sample(path, count)
    return frames


def uniform_video_sample(path: str, count: int) -> tuple[list[Any], dict[str, Any]]:
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
    frames = [Image.fromarray(frame).convert("RGB") for frame in array]
    if len(frames) != count:
        raise ValueError(f"Requested {count} video frames, decoded {len(frames)}")
    fps = float(reader.get_avg_fps())
    if fps <= 0:
        raise ValueError(f"Video has invalid frame rate: {path}")
    return frames, {
        "total_num_frames": length,
        "fps": fps,
        "width": int(array.shape[2]),
        "height": int(array.shape[1]),
        "duration": length / fps,
        "video_backend": "decord",
        "frames_indices": indices,
    }


def request_messages_with_uniform_video(
    request: PrefillRequest,
    *,
    requested_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace each video path with an exact uniformly sampled frame sequence."""
    if not 1 <= requested_frames <= 64:
        raise ValueError("requested_frames must be in [1, 64]")
    messages: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for message in request.messages:
        content: list[dict[str, Any]] = []
        for raw_item in message.get("content", []):
            if not isinstance(raw_item, Mapping):
                raise TypeError("Multimodal content items must be mappings")
            item = dict(raw_item)
            if item.get("type") == "video":
                path = required_media_path(item.get("video"), "video")
                frames, metadata = uniform_video_sample(path, requested_frames)
                item["video"] = frames
                item.pop("fps", None)
                item.pop("num_frames", None)
                item.pop("nframes", None)
                metadata_rows.append(metadata)
            content.append(item)
        messages.append({**dict(message), "content": content})
    actual_frames = requested_frames * len(metadata_rows)
    return messages, {
        "video_sampling_method": UNIFORM_VIDEO_SAMPLING if metadata_rows else None,
        "requested_frames": actual_frames,
        "actual_frames": actual_frames,
        "video_frame_indices": [row["frames_indices"] for row in metadata_rows],
        "video_source_total_frames": [row["total_num_frames"] for row in metadata_rows],
        "video_metadata": metadata_rows,
    }


def validate_video_grid_frames(
    model_inputs: Mapping[str, Any],
    *,
    processor: Any,
    requested_frames: int,
    family: str,
) -> int:
    """Assert that the processor retained the requested physical frame count."""
    grid = model_inputs.get("video_grid_thw")
    if grid is None:
        raise ValueError(f"{family} video request produced no video_grid_thw")
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None or not hasattr(video_processor, "temporal_patch_size"):
        raise ValueError(f"{family} processor does not expose temporal_patch_size")
    temporal_patch_size = int(video_processor.temporal_patch_size)
    if temporal_patch_size <= 0:
        raise ValueError(f"{family} processor has invalid temporal_patch_size")
    temporal_grid = grid[:, 0] if hasattr(grid, "ndim") else [row[0] for row in grid]
    actual_frames = sum(int(value) for value in temporal_grid) * temporal_patch_size
    if actual_frames != requested_frames:
        raise ValueError(
            f"{family} requested {requested_frames} video frames but processor retained "
            f"{actual_frames}"
        )
    return actual_frames
