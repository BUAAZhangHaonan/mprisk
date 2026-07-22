"""Run and validate the 1-Conflict/1-Aligned cache smoke matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from mprisk.cache.cache_matrix_queue import (
    CONDITIONS,
    SMOKE_SCHEMA,
    CacheJob,
    MatrixConfig,
    build_asset_signature,
    build_job_environment,
    load_matrix_config,
    model_runtime_library_path,
    normalize_manifest,
)


@dataclass(frozen=True)
class SmokePaths:
    root: Path
    manifest: Path
    cache: Path
    log: Path
    evidence: Path
    failure: Path


def build_smoke_manifest(job: CacheJob, paths: SmokePaths) -> tuple[list[dict[str, Any]], str]:
    rows, _ = normalize_manifest(job.domain)
    selected: list[dict[str, Any]] = []
    for sample_type in ("Aligned", "Conflict"):
        matches = [row for row in rows if row["sample_type"] == sample_type]
        if not matches:
            raise ValueError(f"{job.job_id} has no {sample_type} sample")
        selected.append(min(matches, key=lambda row: str(row["sample_id"])))
    text = "".join(_canonical_json(row) + "\n" for row in selected)
    _atomic_text(paths.manifest, text)
    return selected, hashlib.sha256(text.encode()).hexdigest()


def run_smoke_job(
    config: MatrixConfig, job: CacheJob, *, physical_gpu: int | None = None
) -> dict[str, Any]:
    paths = smoke_paths(job)
    rows, manifest_sha256 = build_smoke_manifest(job, paths)
    if paths.evidence.is_file():
        existing = _read_json(paths.evidence)
        if _evidence_matches(config, job, existing, manifest_sha256):
            return {"job_id": job.job_id, "status": "PASS", "resumed": True}
        raise ValueError(f"Stale smoke evidence must not be reused: {paths.evidence}")

    execution_gpu = job.model.gpu_lane if physical_gpu is None else physical_gpu
    _require_gpu_capacity(execution_gpu, config.max_gpu_memory_fraction)
    command = _smoke_command(config, job, paths)
    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=config.repo_root,
            env=build_job_environment(config, job, execution_gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        payload = {
            "schema": "mprisk_cache_smoke_failure_v1",
            "status": "FAIL",
            "job_id": job.job_id,
            "return_code": completed.returncode,
            "command": command,
            "execution_gpu": execution_gpu,
            "log_path": str(paths.log),
        }
        _atomic_json(paths.failure, payload)
        return payload

    evidence = validate_smoke(
        config, job, paths, rows, manifest_sha256, execution_gpu=execution_gpu
    )
    _atomic_json(paths.evidence, evidence)
    paths.failure.unlink(missing_ok=True)
    return {"job_id": job.job_id, "status": "PASS", "resumed": False}


def validate_smoke(
    config: MatrixConfig,
    job: CacheJob,
    paths: SmokePaths,
    rows: list[dict[str, Any]],
    manifest_sha256: str,
    *,
    execution_gpu: int,
) -> dict[str, Any]:
    summary = _read_json(paths.cache / "batch_summary.json")
    expected_tasks = 2 * 8 * len(CONDITIONS)
    expected_summary = {
        "total": expected_tasks,
        "completed": expected_tasks,
        "failed": 0,
        "pending": 0,
        "running": 0,
    }
    if summary != expected_summary:
        raise ValueError(f"Incomplete smoke batch for {job.job_id}: {summary}")

    ledger_counts, ledger_rows = _read_ledger(paths.cache / "batch_state.sqlite3")
    if ledger_counts != {"completed": expected_tasks}:
        raise ValueError(f"Invalid smoke ledger for {job.job_id}: {ledger_counts}")
    sample_type_counts = Counter(str(row["sample_type"]) for row in ledger_rows)
    if sample_type_counts != {"Aligned": 24, "Conflict": 24}:
        raise ValueError(f"Invalid sample types for {job.job_id}: {sample_type_counts}")

    entries = _read_jsonl(paths.cache / "manifest.jsonl")
    if len(entries) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} cache entries, got {len(entries)}")
    keys = {
        (
            str(entry["sample_id"]),
            str(entry["prompt_id"]),
            str(entry["condition"]),
        )
        for entry in entries
    }
    if len(keys) != expected_tasks:
        raise ValueError(f"Duplicate cache identities for {job.job_id}")
    prompt_ids = sorted({str(entry["prompt_id"]) for entry in entries})
    conditions = Counter(str(entry["condition"]) for entry in entries)
    if len(prompt_ids) != 8 or conditions != {condition: 16 for condition in CONDITIONS}:
        raise ValueError(
            f"Invalid prompt/condition coverage for {job.job_id}: {prompt_ids}, {conditions}"
        )

    token_counts: dict[str, list[int]] = {condition: [] for condition in CONDITIONS}
    media_checks: Counter[str] = Counter()
    requested_frames = job.model.requested_frames
    asset_signature = build_asset_signature(config, job.model)
    actual_frames: dict[str, list[int]] = {condition: [] for condition in CONDITIONS}
    provenance_contracts: set[str] = set()
    peak_gpu_memory = 0
    for entry in entries:
        shape = [int(entry["layer_count"]), int(entry["hidden_dim"])]
        if shape != list(job.model.trajectory_shape):
            raise ValueError(f"Unexpected trajectory shape for {job.job_id}: {shape}")
        token_count = int(entry["token_count"])
        t0 = int(entry["t0_token_index"])
        if token_count <= 0 or t0 != token_count - 1:
            raise ValueError(f"Invalid t0 for {job.job_id}: tokens={token_count}, t0={t0}")
        token_counts[str(entry["condition"])].append(token_count)
        peak_gpu_memory = max(peak_gpu_memory, int(entry.get("peak_gpu_memory_bytes") or 0))

        root = Path(str(entry["cache_root"]))
        shard = root / str(entry["shard_path"])
        sidecar_rel = str(entry["metadata"]["sidecar_path"])
        sidecar = root / sidecar_rel
        if _sha256(shard) != str(entry["checksum"]):
            raise ValueError(f"Checksum mismatch: {shard}")
        hidden = load_file(shard).get("hidden_states")
        if hidden is None or list(hidden.shape) != shape or not np.isfinite(hidden).all():
            raise ValueError(f"Invalid hidden-state tensor: {shard}")
        sidecar_payload = _read_json(sidecar)
        request = sidecar_payload.get("request")
        provenance = sidecar_payload.get("provenance")
        if not isinstance(request, dict) or not isinstance(provenance, dict):
            raise ValueError(f"Invalid sidecar contract: {sidecar}")
        media_contract, contains_video = _validate_media_contract(job.model.protocol, request)
        media_checks[media_contract] += 1
        actual_frames[str(entry["condition"])].append(
            _validate_frame_contract(
                provenance,
                condition=str(entry["condition"]),
                contains_video=contains_video,
                expected_frames=requested_frames,
                expected_method=job.model.video_sampling_method,
            )
        )
        provenance_contracts.add(
            _canonical_json(
                {
                    key: provenance.get(key)
                    for key in (
                        "model_path",
                        "model_class",
                        "processor_class",
                        "transformers_version",
                        "num_hidden_layers",
                        "hidden_size",
                        "video_num_segments",
                        "num_frames",
                        "requested_frames",
                        "actual_frames",
                        "video_sampling_method",
                        "video_frame_indices",
                        "video_source_total_frames",
                    )
                    if provenance.get(key) is not None
                }
            )
        )

    return {
        "schema": SMOKE_SCHEMA,
        "status": "PASS",
        "model_key": job.model.model_key,
        "family": job.model.family,
        "protocol": job.model.protocol,
        "domain": job.domain.domain,
        "expected_tasks": expected_tasks,
        "completed_tasks": expected_tasks,
        "failed_tasks": 0,
        "environment_python": str(job.model.python),
        "python_no_user_site": job.model.python_no_user_site,
        "env_isolation": job.model.env_isolation,
        "runtime_library_path": str(model_runtime_library_path(job.model)),
        "execution_gpu": execution_gpu,
        "prompt_set_sha256": _sha256(config.prompt_sets[job.model.protocol]),
        "asset_config_sha256": _sha256(config.asset_config),
        "smoke_manifest_sha256": manifest_sha256,
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "sample_types": [str(row["sample_type"]) for row in rows],
        "prompt_ids": prompt_ids,
        "conditions": dict(conditions),
        "trajectory_shape": list(job.model.trajectory_shape),
        "extra_args": list(job.model.extra_args),
        "dtype": job.model.dtype,
        "requested_frames": requested_frames,
        "frame_protocol": job.model.frame_protocol,
        "video_sampling_method": job.model.video_sampling_method,
        "asset_signature": asset_signature,
        "actual_frames": {
            condition: {
                "unique": sorted(set(values)),
                "entries": len(values),
            }
            for condition, values in actual_frames.items()
        },
        "token_count": {
            condition: {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for condition, values in token_counts.items()
        },
        "media_contract_counts": dict(media_checks),
        "provenance_contracts": sorted(provenance_contracts),
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "log_path": str(paths.log),
        "cache_root": str(paths.cache),
    }


def smoke_paths(job: CacheJob) -> SmokePaths:
    root = job.smoke_evidence.parent
    return SmokePaths(
        root=root,
        manifest=root / "smoke_manifest.jsonl",
        cache=root / "cache",
        log=root / "runtime.log",
        evidence=job.smoke_evidence,
        failure=root / "SMOKE_FAILURE.json",
    )


def _smoke_command(config: MatrixConfig, job: CacheJob, paths: SmokePaths) -> list[str]:
    return [
        str(job.model.python),
        str(config.job_runner),
        "--gpu-memory-fraction",
        str(config.max_gpu_memory_fraction),
        "--",
        "--manifest",
        str(paths.manifest),
        "--prompt-set",
        str(config.prompt_sets[job.model.protocol]),
        "--protocol",
        job.model.protocol,
        "--model-key",
        job.model.model_key,
        "--asset-config",
        str(config.asset_config),
        "--device",
        "cuda:0",
        "--dtype",
        job.model.dtype,
        "--prefill-strategy",
        "full_prefill",
        "--output-root",
        str(paths.cache),
        "--materialize-every",
        "48",
        "--fail-fast",
        "--video-num-segments",
        str(job.model.requested_frames),
        *job.model.extra_args,
    ]


def _require_gpu_capacity(lane: int, fraction: float) -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            str(lane),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used, total = (float(item.strip()) for item in completed.stdout.split(","))
    if used / total >= fraction:
        raise RuntimeError(f"GPU {lane} memory is already {used / total:.1%} utilized")


def _validate_media_contract(protocol: str, request: dict[str, Any]) -> tuple[str, bool]:
    condition = str(request.get("condition"))
    media = request.get("media_paths")
    if not isinstance(media, dict):
        raise ValueError("Smoke sidecar media_paths must be a mapping")
    for value in media.values():
        if not Path(str(value)).is_file():
            raise FileNotFoundError(value)
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Smoke sidecar messages must be a list")
    content_types = [
        str(item.get("type"))
        for message in messages
        if isinstance(message, dict)
        for item in message.get("content", [])
        if isinstance(item, dict)
    ]
    media_types = [item for item in content_types if item in {"image", "video", "audio"}]
    use_audio_in_video = bool(request.get("use_audio_in_video"))
    if protocol == "vt":
        expected = ["video"] if condition in {"M1", "M12"} else []
    else:
        if condition == "M1":
            expected = ["video"]
        elif condition == "M2":
            expected = ["audio"]
        else:
            expected = ["video"] if use_audio_in_video else ["video", "audio"]
    if media_types != expected:
        raise ValueError(
            f"{protocol}/{condition} expected message media {expected}; got {media_types}"
        )
    return (
        f"{protocol}:{condition}:{'+'.join(media_types) or 'none'}:"
        f"embedded_audio={str(use_audio_in_video).lower()}",
        "video" in media_types or "image" in media_types,
    )


def _validate_frame_contract(
    provenance: dict[str, Any],
    *,
    condition: str,
    contains_video: bool,
    expected_frames: int,
    expected_method: str,
) -> int:
    requested = provenance.get("requested_frames")
    actual = provenance.get("actual_frames")
    if not isinstance(requested, int) or isinstance(requested, bool):
        raise ValueError(f"{condition} provenance requested_frames must be an integer")
    if not isinstance(actual, int) or isinstance(actual, bool):
        raise ValueError(f"{condition} provenance actual_frames must be an integer")
    expected = expected_frames if contains_video else 0
    if requested != expected or actual != expected:
        raise ValueError(
            f"{condition} frame contract mismatch: expected={expected}, "
            f"requested={requested}, actual={actual}"
        )
    if contains_video:
        method = provenance.get("video_sampling_method")
        indices_by_video = provenance.get("video_frame_indices")
        source_totals = provenance.get("video_source_total_frames")
        if method != expected_method:
            raise ValueError(
                f"{condition} expected video_sampling_method={expected_method!r}; "
                f"got {method!r}"
            )
        if indices_by_video is None or source_totals is None:
            raise ValueError(
                f"{condition} must provide both frame indices and source frame totals"
            )
        if isinstance(indices_by_video, list) and all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in indices_by_video
        ):
            indices_by_video = [indices_by_video]
        if isinstance(source_totals, int) and not isinstance(source_totals, bool):
            source_totals = [source_totals]
        if (
            not isinstance(indices_by_video, list)
            or len(indices_by_video) != 1
            or not isinstance(indices_by_video[0], list)
        ):
            raise ValueError(
                f"{condition} requires one nested video_frame_indices list; "
                f"got {indices_by_video!r}"
            )
        indices = indices_by_video[0]
        if (
            len(indices) != actual
            or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError(f"{condition} has invalid video_frame_indices={indices!r}")
        if (
            not isinstance(source_totals, list)
            or len(source_totals) != 1
            or not isinstance(source_totals[0], int)
            or isinstance(source_totals[0], bool)
            or source_totals[0] <= 0
        ):
            raise ValueError(
                f"{condition} has invalid video_source_total_frames={source_totals!r}"
            )
        source_total = source_totals[0]
        if any(index < 0 or index >= source_total for index in indices):
            raise ValueError(
                f"{condition} video_frame_indices exceed source frame count {source_total}"
            )
    return actual


def _read_ledger(path: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        counts = {
            str(status): int(count)
            for status, count in connection.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status"
            ).fetchall()
        }
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT sample_id, sample_type, prompt_id, condition, status FROM tasks"
            ).fetchall()
        ]
    finally:
        connection.close()
    return counts, rows


def _evidence_matches(
    config: MatrixConfig, job: CacheJob, value: dict[str, Any], manifest_sha256: str
) -> bool:
    expected = {
        "schema": SMOKE_SCHEMA,
        "status": "PASS",
        "model_key": job.model.model_key,
        "family": job.model.family,
        "protocol": job.model.protocol,
        "domain": job.domain.domain,
        "expected_tasks": 48,
        "completed_tasks": 48,
        "failed_tasks": 0,
        "environment_python": str(job.model.python),
        "python_no_user_site": job.model.python_no_user_site,
        "env_isolation": job.model.env_isolation,
        "runtime_library_path": str(model_runtime_library_path(job.model)),
        "prompt_set_sha256": _sha256(config.prompt_sets[job.model.protocol]),
        "asset_config_sha256": _sha256(config.asset_config),
        "smoke_manifest_sha256": manifest_sha256,
        "trajectory_shape": list(job.model.trajectory_shape),
        "extra_args": list(job.model.extra_args),
        "dtype": job.model.dtype,
        "requested_frames": job.model.requested_frames,
        "frame_protocol": job.model.frame_protocol,
        "video_sampling_method": job.model.video_sampling_method,
        "asset_signature": build_asset_signature(config, job.model),
    }
    return all(value.get(key) == expected_value for key, expected_value in expected.items())


def execute(
    config: MatrixConfig,
    domain: str,
    model_keys: set[str] | None,
    *,
    physical_gpu: int | None = None,
) -> dict[str, Any]:
    jobs = [
        job
        for job in config.jobs
        if job.domain.domain == domain
        and (model_keys is None or job.model.model_key in model_keys)
    ]
    if model_keys is not None:
        missing = model_keys - {job.model.model_key for job in jobs}
        if missing:
            raise ValueError(f"Unknown model keys: {sorted(missing)}")
    if physical_gpu is not None and physical_gpu not in {0, 1}:
        raise ValueError("physical_gpu must be 0 or 1")

    def run_lane(lane: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for job in jobs:
            if job.model.gpu_lane != lane:
                continue
            try:
                results.append(run_smoke_job(config, job, physical_gpu=physical_gpu))
            except Exception as exc:  # Failures are explicit and do not hide later model evidence.
                payload = {
                    "schema": "mprisk_cache_smoke_failure_v1",
                    "status": "FAIL",
                    "job_id": job.job_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                _atomic_json(smoke_paths(job).failure, payload)
                results.append(payload)
        return results

    if physical_gpu is None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_lane, lane) for lane in (0, 1)]
            results = [item for future in futures for item in future.result()]
    else:
        results = []
        for lane in (0, 1):
            results.extend(run_lane(lane))
    status = "PASS" if all(result["status"] == "PASS" for result in results) else "FAIL"
    return {
        "schema": "mprisk_cache_smoke_matrix_run_v1",
        "status": status,
        "domain": domain,
        "results": sorted(results, key=lambda item: str(item["job_id"])),
    }


def launch_tmux(
    config: MatrixConfig,
    domain: str,
    model_keys: list[str],
    *,
    session_name: str | None = None,
    physical_gpu: int | None = None,
) -> str:
    session = session_name or f"{config.tmux_session}-smoke-{domain}"
    if subprocess.run(["tmux", "has-session", "-t", session], check=False).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    command = [
        "env",
        f"PYTHONPATH={config.repo_root / 'src'}",
        sys.executable,
        str(config.repo_root / "scripts" / "run_cache_smoke_matrix.py"),
        "--config",
        str(config.source_path),
        "--domain",
        domain,
        "--execute",
    ]
    for model_key in model_keys:
        command.extend(["--model", model_key])
    if physical_gpu is not None:
        command.extend(["--physical-gpu", str(physical_gpu)])
    subprocess.run(["tmux", "new-session", "-d", "-s", session, *command], check=True)
    return session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--domain", default="target", choices=("source", "target"))
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--tmux-session")
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--launch", action="store_true")
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_matrix_config(args.config)
    models = set(str(value) for value in args.model) or None
    if args.launch:
        session = launch_tmux(
            config,
            args.domain,
            sorted(models or []),
            session_name=args.tmux_session,
            physical_gpu=args.physical_gpu,
        )
        payload = {"status": "launched", "domain": args.domain, "tmux_session": session}
    else:
        payload = execute(
            config, args.domain, models, physical_gpu=args.physical_gpu
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"PASS", "launched"} else 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(cli())
