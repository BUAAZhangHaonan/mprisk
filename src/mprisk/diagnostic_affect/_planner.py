"""Diagnostic Affect Description plan construction from manifest rows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mprisk.models.base_wrapper import GenerationRequest
from mprisk.utils.io import (
    canonical_json as _canonical_json,
    hash_text as _hash_text,
    read_jsonl as _read_jsonl,
    sha256_file as _sha256,
)

from mprisk.diagnostic_affect._plan import (
    CANONICAL_DIAGNOSTIC_AFFECT_PROMPT,
    CONFIG_SCHEMA,
    DiagnosticAffectDescriptionPlan,
    DiagnosticAffectDescriptionTask,
    SIGNATURE_SCHEMA,
    _SUPPORTED_CONDITIONS,
    _SUPPORTED_PROTOCOLS,
)


def build_diagnostic_affect_description_plan(
    *,
    schema_name: str,
    run_id: str,
    manifest_path: Path,
    subject_model_key: str,
    model_family: str,
    model_path: Path,
    protocol: str,
    condition: str,
    dataset: str,
    split: str,
    max_new_tokens: int,
    video_fps: float = 1.0,
    asset_config_sha256: str = "test-asset-config",
    config_sha256: str = "test-config",
    selected_sample_ids: Iterable[str] | None = None,
) -> DiagnosticAffectDescriptionPlan:
    """Build a strict sample-level plan from one explicit dataset/protocol/split."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    protocol = protocol.upper()
    condition = condition.upper()
    if protocol not in _SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported Diagnostic Affect Description protocol: {protocol!r}")
    if condition not in _SUPPORTED_CONDITIONS:
        raise ValueError(f"Unsupported Diagnostic Affect Description condition: {condition!r}")
    if schema_name != CONFIG_SCHEMA:
        raise ValueError(f"Unsupported Diagnostic Affect Description schema: {schema_name!r}")
    for field_name, value in (
        ("run_id", run_id),
        ("subject_model_key", subject_model_key),
        ("model_family", model_family),
        ("dataset", dataset),
        ("split", split),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    selected = None if selected_sample_ids is None else set(selected_sample_ids)
    source_rows = _read_jsonl(manifest_path)
    rows = [
        row
        for row in source_rows
        if str(row.get("source_dataset", "")) == dataset
        and str(row.get("split", "")) == split
        and str(row.get("protocol", "")).upper() == protocol
    ]
    if not rows:
        raise ValueError(
            "Manifest contains no rows for "
            f"dataset={dataset!r}, split={split!r}, protocol={protocol!r}"
        )
    tasks: list[DiagnosticAffectDescriptionTask] = []
    for row in rows:
        sample_id = _required_string(row, "sample_id")
        if selected is not None and sample_id not in selected:
            continue
        media_paths = _required_media_paths(row, protocol=protocol)
        media_hashes = {name: _sha256(path) for name, path in media_paths.items()}
        vision_path = media_paths["vision"]
        content: list[dict[str, Any]] = [
            {"type": "video", "video": str(vision_path), "fps": video_fps}
        ]
        if protocol == "VT":
            dialogue = _required_string(row, "text_content")
            content.append(
                {
                    "type": "text",
                    "text": f"{dialogue}\n\n{CANONICAL_DIAGNOSTIC_AFFECT_PROMPT}",
                }
            )
            use_audio_in_video = False
        else:
            content.append({"type": "text", "text": CANONICAL_DIAGNOSTIC_AFFECT_PROMPT})
            use_audio_in_video = True
        request = GenerationRequest(
            sample_id=sample_id,
            model_key=subject_model_key,
            protocol=protocol.lower(),
            condition=condition,
            messages=({"role": "user", "content": content},),
            media_paths={name: str(path) for name, path in media_paths.items()},
            use_audio_in_video=use_audio_in_video,
            generation_kwargs={
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
            },
        )
        input_payload = {
            "sample_id": sample_id,
            "subject_model_key": subject_model_key,
            "protocol": protocol,
            "condition": condition,
            "dataset": dataset,
            "split": split,
            "messages": list(request.messages),
            "media_sha256": media_hashes,
        }
        input_sha256 = _hash_text(_canonical_json(input_payload))
        task_id = _hash_text(
            _canonical_json({"sample_id": sample_id, "input_sha256": input_sha256})
        )
        tasks.append(
            DiagnosticAffectDescriptionTask(
                task_id=task_id,
                request=request,
                input_sha256=input_sha256,
                media_sha256=_hash_text(_canonical_json(media_hashes)),
                prompt_sha256=_hash_text(CANONICAL_DIAGNOSTIC_AFFECT_PROMPT),
            )
        )
    if selected is not None and {task.request.sample_id for task in tasks} != selected:
        raise ValueError(
            "One or more selected sample IDs are absent from the selected manifest rows"
        )
    if len({task.request.sample_id for task in tasks}) != len(tasks):
        raise ValueError("Selected manifest rows contain duplicate sample_id values")
    counts = {protocol: len(tasks)}
    model_path = model_path.expanduser().resolve()
    signature = {
        "schema_name": SIGNATURE_SCHEMA,
        "run_id": run_id,
        "manifest_sha256": _sha256(manifest_path),
        "asset_config_sha256": asset_config_sha256,
        "subject_model_key": subject_model_key,
        "model_family": model_family,
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "model_weight_map_sha256": _model_weight_map_sha256(model_path),
        "protocol": protocol,
        "condition": condition,
        "dataset": dataset,
        "split": split,
        "prompt_sha256": _hash_text(CANONICAL_DIAGNOSTIC_AFFECT_PROMPT),
        "config_sha256": config_sha256,
        "max_new_tokens": max_new_tokens,
        "video_fps": video_fps,
        "generation_policy": {"do_sample": False, "num_beams": 1},
        "task_count": len(tasks),
        "counts": counts,
    }
    return DiagnosticAffectDescriptionPlan(tasks=tasks, signature=signature, counts=counts)


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest row has no non-empty {key}: {row.get('sample_id')}")
    return value


def _required_media_paths(row: dict[str, Any], *, protocol: str) -> dict[str, Path]:
    raw = row.get("media_paths")
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest row has no media_paths object: {row.get('sample_id')}")
    required_modalities = ("vision",) if protocol == "VT" else ("vision", "audio")
    paths: dict[str, Path] = {}
    for modality in required_modalities:
        value = raw.get(modality)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Manifest row has no {modality} media path: {row.get('sample_id')}"
            )
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model input media is missing: {path}")
        paths[modality] = path
    return paths


def _model_weight_map_sha256(model_path: Path) -> str:
    index_files = sorted(model_path.glob("*.index.json"))
    if index_files:
        entries = {path.name: _sha256(path) for path in index_files}
    else:
        weight_files = sorted(model_path.glob("*.safetensors")) + sorted(
            model_path.glob("*.bin")
        )
        if not weight_files:
            raise FileNotFoundError(f"No model weight files or index found in {model_path}")
        entries = {path.name: _sha256(path) for path in weight_files}
    return _hash_text(_canonical_json(entries))


def _select_smoke_sample_ids(
    manifest_path: Path,
    *,
    dataset: str,
    split: str,
    protocol: str,
) -> list[str]:
    selected: dict[str, str] = {}
    for row in _read_jsonl(manifest_path):
        if (
            str(row.get("source_dataset", "")) != dataset
            or str(row.get("split", "")) != split
            or str(row.get("protocol", "")).upper() != protocol.upper()
        ):
            continue
        sample_type = _required_string(row, "sample_type")
        if sample_type in {"Conflict", "Aligned"} and sample_type not in selected:
            selected[sample_type] = _required_string(row, "sample_id")
    if set(selected) != {"Conflict", "Aligned"}:
        raise ValueError("Smoke selection requires one Conflict and one Aligned sample")
    return [selected["Conflict"], selected["Aligned"]]
