"""Command-line orchestration for single-sample prefill cache extraction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mprisk.cache.prefill_writer import write_prefill_result
from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.qwen_omni import build_condition_request
from mprisk.models.wrapper_registry import get_wrapper

WrapperFactory = Callable[..., Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract first-reply-token Qwen2.5-Omni Thinker trajectories."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sample-id", required=True, action="append")
    parser.add_argument("--protocol", required=True, choices=("vt", "va", "vta"))
    parser.add_argument("--conditions", nargs="+", default=("M1", "M2", "M12"))
    parser.add_argument("--task-prompt", required=True)
    parser.add_argument(
        "--joint-audio-mode",
        choices=("embedded_video", "separate_file"),
        default="embedded_video",
    )
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--model-key", default="qwen2_5_omni_7b")
    parser.add_argument("--family", default="qwen_omni", choices=("qwen_omni",))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "eager"))
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    wrapper_factory: WrapperFactory | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    requests = _build_requests(args)
    if args.dry_run:
        print(
            json.dumps(
                {"status": "dry_run", "requests": [_request_payload(item) for item in requests]},
                ensure_ascii=False,
            )
        )
        return 0

    factory = wrapper_factory or get_wrapper(args.family)
    wrapper = factory(
        model_key=args.model_key,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    artifacts = []
    try:
        wrapper.load()
        for request in requests:
            result = wrapper.extract_prefill(request)
            artifact = write_prefill_result(
                result,
                output_root=args.output_root,
                manifest_path=args.cache_manifest,
                overwrite=args.overwrite,
            )
            artifacts.append(
                {
                    "sample_id": request.sample_id,
                    "condition": request.condition,
                    "shape": [result.layer_count, result.hidden_dim],
                    "t0_token_index": result.t0_token_index,
                    "token_count": result.token_count,
                    "shard_path": str(artifact.shard_path),
                    "sidecar_path": str(artifact.sidecar_path),
                    "checksum": artifact.checksum,
                    "elapsed_seconds": result.provenance.get("elapsed_seconds"),
                    "peak_gpu_memory_bytes": result.provenance.get("peak_gpu_memory_bytes"),
                }
            )
    finally:
        wrapper.close()
    print(json.dumps({"status": "ok", "artifacts": artifacts}, ensure_ascii=False))
    return 0


def _build_requests(args: argparse.Namespace) -> list[PrefillRequest]:
    rows = _read_jsonl(args.manifest)
    requested_ids = list(args.sample_id)
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("--sample-id values must be unique")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id in requested_ids and str(row.get("protocol", "")).lower() == args.protocol:
            if sample_id in by_id:
                raise ValueError(f"Manifest contains duplicate sample/protocol row: {sample_id}")
            by_id[sample_id] = row
    missing = [sample_id for sample_id in requested_ids if sample_id not in by_id]
    if missing:
        raise KeyError(f"Samples are missing for protocol {args.protocol}: {missing}")

    conditions = [str(condition).upper() for condition in args.conditions]
    if len(set(conditions)) != len(conditions):
        raise ValueError("--conditions values must be unique")
    requests = []
    for sample_id in requested_ids:
        row = by_id[sample_id]
        media_paths = row.get("media_paths")
        if not isinstance(media_paths, dict):
            raise ValueError(f"Manifest row has invalid media_paths: {sample_id}")
        for condition in conditions:
            request = build_condition_request(
                sample_id=sample_id,
                model_key=args.model_key,
                protocol=args.protocol,
                condition=condition,
                dataset_key=str(row.get("source_dataset", "")),
                split=str(row.get("split", "unspecified")),
                media_paths={str(key): str(value) for key, value in media_paths.items()},
                transcript=_optional_string(row.get("text_content")),
                task_prompt=args.task_prompt,
                joint_audio_mode=args.joint_audio_mode,
                video_fps=args.video_fps,
            )
            _validate_local_media(request)
            requests.append(request)
    return requests


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Manifest line {line_number} must be a JSON object")
            rows.append(payload)
    return rows


def _validate_local_media(request: PrefillRequest) -> None:
    for path in request.media_paths.values():
        media = Path(path).expanduser()
        if not media.is_file():
            raise FileNotFoundError(f"Request media file does not exist: {media}")


def _request_payload(request: PrefillRequest) -> dict[str, Any]:
    return {
        "sample_id": request.sample_id,
        "model_key": request.model_key,
        "protocol": request.protocol,
        "condition": request.condition,
        "dataset_key": request.dataset_key,
        "split": request.split,
        "messages": list(request.messages),
        "media_paths": dict(request.media_paths),
        "use_audio_in_video": request.use_audio_in_video,
    }


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
