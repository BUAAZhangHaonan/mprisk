"""Per-task execution, recovery, dry-run estimation, and result materialization."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from mprisk.cache._batch_ledger import BatchLedger
from mprisk.cache._batch_plan import CONDITIONS, BatchPlan, RecoveredArtifact, BatchTask
from mprisk.cache.prefill_writer import (
    prefill_artifact_paths,
    write_full_cache_manifest,
)
from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.qwen_omni import build_condition_request
from mprisk.utils.io import (
    atomic_write_text as _atomic_text,
    canonical_json as _canonical_json,
    read_json_object as _read_json,
    sha256_file as _sha256,
)


def _request_for_task(args: argparse.Namespace, task: BatchTask) -> PrefillRequest:
    if task.prompt_text is None:
        raise ValueError(f"Task {task.task_id} has an unresolved prompt")
    media = task.row["media_paths"]
    return build_condition_request(
        sample_id=task.sample_id,
        model_key=args.model_key,
        protocol=args.protocol,
        condition=task.condition,
        dataset_key=str(task.row["source_dataset"]),
        split=str(task.row["split"]),
        media_paths={str(key): str(value) for key, value in media.items()},
        transcript=None if task.row.get("text_content") is None else str(task.row["text_content"]),
        task_prompt=task.prompt_text,
        prompt_set_key=task.prompt_set_key,
        prompt_id=task.prompt_id,
        joint_audio_mode=args.joint_audio_mode,
        video_fps=args.video_fps,
    )


def _recover_entry(request: PrefillRequest, prompt_root: Path) -> RecoveredArtifact | None:
    paths = prefill_artifact_paths(request, output_root=prompt_root)
    shard_exists = paths.shard_path.is_file()
    sidecar_exists = paths.sidecar_path.is_file()
    if not shard_exists and not sidecar_exists:
        return None
    # Shard without sidecar = crashed mid-write; delete the orphan and re-extract.
    if shard_exists and not sidecar_exists:
        paths.shard_path.unlink(missing_ok=True)
        return None
    if not shard_exists and sidecar_exists:
        raise RuntimeError(f"Incomplete cache artifact pair for {request.sample_id}")
    payload = _read_json(paths.sidecar_path)
    if payload.get("schema") != "mprisk_prefill_cache_sidecar_v1":
        raise ValueError(f"Unsupported sidecar schema: {paths.sidecar_path}")
    expected_request = {
        "sample_id": request.sample_id,
        "model_key": request.model_key,
        "protocol": request.protocol,
        "condition": request.condition,
        "prompt_set_key": request.prompt_set_key,
        "prompt_id": request.prompt_id,
        "dataset_key": request.dataset_key,
        "split": request.split,
        "messages": list(request.messages),
        "media_paths": dict(request.media_paths),
        "use_audio_in_video": request.use_audio_in_video,
    }
    if payload.get("request") != expected_request:
        raise ValueError(f"Existing sidecar request mismatch: {paths.sidecar_path}")
    entry = payload.get("entry")
    if not isinstance(entry, dict) or entry.get("checksum") != _sha256(paths.shard_path):
        raise ValueError(f"Existing cache checksum mismatch: {paths.shard_path}")
    tensors = load_file(paths.shard_path)
    hidden = tensors.get("hidden_states")
    if hidden is None or list(hidden.shape) != [entry.get("layer_count"), entry.get("hidden_dim")]:
        raise ValueError(f"Existing cache tensor shape mismatch: {paths.shard_path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Existing sidecar provenance is invalid: {paths.sidecar_path}")
    return RecoveredArtifact(entry=entry, provenance=provenance)


def _materialize_outputs(ledger: BatchLedger, root: Path, prompt_ids: Sequence[str]) -> None:
    for prompt_id in prompt_ids:
        manifest = root / "prompts" / prompt_id / "manifests" / "unified_full_cache_manifest.json"
        write_full_cache_manifest(ledger.completed_entries(prompt_id), manifest)
    entries = ledger.completed_entries_all()
    _atomic_text(root / "manifest.jsonl", "".join(_canonical_json(row) + "\n" for row in entries))
    _materialize_failures(ledger, root)
    _atomic_text(
        root / "batch_summary.json",
        json.dumps(ledger.summary(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _materialize_failures(ledger: BatchLedger, root: Path) -> None:
    lines = "".join(_canonical_json(row) + "\n" for row in ledger.failures())
    _atomic_text(root / "failures.jsonl", lines)


def _dry_run_payload(args: argparse.Namespace, plan: BatchPlan) -> dict[str, Any]:
    rows = plan.rows
    payload: dict[str, Any] = {
        "status": "dry_run",
        "ready": not plan.unresolved_prompt_variables,
        "unresolved_prompt_variables": plan.unresolved_prompt_variables,
        "sample_count": len(rows),
        "prompt_count": len(plan.prompt_ids),
        "prompt_ids": plan.prompt_ids,
        "condition_count": len(CONDITIONS),
        "conditions": CONDITIONS,
        "task_count": len(plan.tasks),
        "sample_type_counts": dict(Counter(str(row.get("sample_type")) for row in rows)),
        "use_in_main_counts": dict(
            Counter(str(bool(row.get("use_in_main"))).lower() for row in rows)
        ),
        "annotation_count_counts": dict(Counter(str(row.get("annotation_count")) for row in rows)),
        "split_counts": dict(Counter(str(row.get("split")) for row in rows)),
        "source_dataset_counts": dict(Counter(str(row.get("source_dataset")) for row in rows)),
        "writes_performed": 0,
    }
    durations = _probe_durations(rows, args.ffprobe_workers) if args.probe_media else None
    if durations is not None:
        payload["media_duration_seconds"] = _duration_summary(durations)
    smoke = _parse_condition_seconds(args.smoke_condition_seconds)
    if smoke:
        if set(smoke) != set(CONDITIONS):
            raise ValueError("Smoke timing requires M1, M2, and M12 values")
        triplet = sum(smoke.values())
        overhead = max(0.0, (args.smoke_wall_seconds or triplet) - triplet)
        payload["gpu_time_estimate"] = {
            "basis_condition_seconds": smoke,
            "model_load_overhead_seconds": overhead,
            "constant_sample_total_seconds": triplet * len(rows) * len(plan.prompt_ids) + overhead,
        }
        if durations is not None and args.smoke_media_seconds:
            payload["gpu_time_estimate"]["linear_duration_total_seconds"] = (
                triplet * sum(durations) / args.smoke_media_seconds * len(plan.prompt_ids)
                + overhead
            )
    if args.trajectory_shape:
        layers, hidden = args.trajectory_shape
        if layers <= 0 or hidden <= 0:
            raise ValueError("--trajectory-shape values must be positive")
        raw_per_task = layers * hidden * 4
        payload["storage_estimate"] = {
            "trajectory_bytes_per_task": raw_per_task,
            "trajectory_total_bytes": raw_per_task * len(plan.tasks),
        }
        if args.smoke_artifact_bytes_per_task is not None:
            payload["storage_estimate"]["smoke_artifact_total_bytes"] = (
                args.smoke_artifact_bytes_per_task * len(plan.tasks)
            )
    if args.gpu_index is not None:
        payload["gpu"] = _gpu_status(args.gpu_index)
    return payload


def _probe_durations(rows: list[dict[str, Any]], workers: int) -> list[float]:
    if workers <= 0:
        raise ValueError("--ffprobe-workers must be positive")
    paths = sorted({str(path) for row in rows for path in row["media_paths"].values()})

    def probe(path: str) -> float:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value = float(completed.stdout.strip())
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid media duration for {path}: {value}")
        return value

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(probe, paths))


def _duration_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total": float(sum(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(max(values)),
    }


def _gpu_status(index: int) -> dict[str, Any]:
    query = "index,name,memory.used,memory.total,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", str(index)],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    used, total, utilization = int(values[2]), int(values[3]), int(values[4])
    return {
        "index": int(values[0]),
        "name": values[1],
        "memory_used_mib": used,
        "memory_total_mib": total,
        "memory_fraction": used / total,
        "utilization_percent": utilization,
        "under_90_percent": used / total < 0.9 and utilization < 90,
    }


def _parse_condition_seconds(items: Sequence[str]) -> dict[str, float]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Smoke timing must be CONDITION=SECONDS: {item!r}")
        key, raw = item.split("=", 1)
        key = key.upper()
        if key in parsed or key not in CONDITIONS:
            raise ValueError(f"Invalid or duplicate smoke condition: {key}")
        value = float(raw)
        if value <= 0:
            raise ValueError("Smoke condition seconds must be positive")
        parsed[key] = value
    return parsed
