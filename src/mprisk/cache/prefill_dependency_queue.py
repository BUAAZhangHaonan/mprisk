"""Event-driven dependent queue for sequential prefill-cache batches."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sqlite3
import struct
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mprisk.config.loader import load_yaml
from mprisk.viz.runtime_records import (
    load_run_records,
    snapshot_cache_summary,
    utc_now,
    write_run_records,
)

QUEUE_SCHEMA = "mprisk_prefill_dependency_queue_v1"
CLASS_CODE = {"A": "Conflict", "C": "Aligned"}
CLASS_CODE_SEMANTICS = {"A": "sample_type.Conflict", "C": "sample_type.Aligned"}


class GateFailure(RuntimeError):
    """Raised when a dependency gate can no longer succeed without intervention."""


class QueueExecutionError(RuntimeError):
    """Raised when a queued extraction job fails its runtime contract."""


class CapacityFailure(GateFailure):
    """Raised when projected cache artifacts would exceed the capacity limit."""


@dataclass(frozen=True)
class GateJob:
    model_key: str
    ledger: Path
    expected_tasks: int
    runtime_cache_key: str


@dataclass(frozen=True)
class MainGate:
    runtime_record: Path
    jobs: tuple[GateJob, ...]


@dataclass(frozen=True)
class FollowupJob:
    job_id: str
    seed: int
    model_key: str
    protocol: str
    manifest: Path
    prompt_set: Path
    output_root: Path
    log_path: Path
    expected_tasks: int
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class CapacityOutput:
    output_root: Path
    expected_tasks: int


@dataclass(frozen=True)
class CapacityModel:
    model_key: str
    calibration_root: Path
    outputs: tuple[CapacityOutput, ...]


@dataclass(frozen=True)
class CapacityGate:
    filesystem_path: Path
    max_projected_utilization: float
    models: tuple[CapacityModel, ...]


@dataclass(frozen=True)
class CapacityStatus:
    safe: bool
    filesystem_path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int
    projected_bytes: int
    projected_used_bytes: int
    projected_utilization: float
    total_inodes: int
    free_inodes: int
    projected_inodes: int
    projected_inode_utilization: float
    max_projected_utilization: float
    models: tuple[dict[str, Any], ...]

    def require_safe(self) -> None:
        if self.safe:
            return
        raise CapacityFailure(
            "Projected cache utilization is "
            f"{self.projected_utilization:.2%} bytes and "
            f"{self.projected_inode_utilization:.2%} inodes; "
            f"limit is {self.max_projected_utilization:.2%}"
        )


@dataclass(frozen=True)
class QueueManifest:
    source_path: Path
    physical_gpu: int
    device: str
    python: Path
    extract_script: Path
    runtime_record: Path
    capacity_gate: CapacityGate
    main_gate: MainGate
    followup_jobs: tuple[FollowupJob, ...]


@dataclass(frozen=True)
class GateStatus:
    ready: bool
    reasons: tuple[str, ...]


class EventWatcher(Protocol):
    def wait(self) -> None: ...

    def close(self) -> None: ...


JobExecutor = Callable[..., None]
WatcherFactory = Callable[[Sequence[Path]], EventWatcher]


def load_queue_manifest(path: str | Path) -> QueueManifest:
    source_path = Path(path).expanduser()
    data = load_yaml(source_path)
    if data.get("schema") != QUEUE_SCHEMA:
        raise ValueError(f"Queue manifest schema must be {QUEUE_SCHEMA}")
    physical_gpu = _positive_int(data, "physical_gpu", allow_zero=True)
    device = _required_str(data, "device")
    if device != "cuda:0":
        raise ValueError("Dependent queue requires process-local device cuda:0")
    main_raw = _required_mapping(data, "main_gate")
    capacity_raw = _required_mapping(data, "capacity_gate")
    gate_jobs = tuple(_load_gate_job(item) for item in _required_list(main_raw, "jobs"))
    jobs = tuple(_load_followup_job(item) for item in _required_list(data, "followup_jobs"))
    if not gate_jobs or not jobs:
        raise ValueError("Dependent queue requires main-gate and follow-up jobs")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Follow-up job IDs must be unique")
    if len({job.output_root for job in jobs}) != len(jobs):
        raise ValueError("Follow-up output roots must be unique")
    return QueueManifest(
        source_path=source_path.resolve(),
        physical_gpu=physical_gpu,
        device=device,
        python=Path(_required_str(data, "python")).expanduser(),
        extract_script=Path(_required_str(data, "extract_script")).expanduser(),
        runtime_record=Path(_required_str(data, "runtime_record")).expanduser(),
        capacity_gate=_load_capacity_gate(capacity_raw),
        main_gate=MainGate(
            runtime_record=Path(_required_str(main_raw, "runtime_record")).expanduser(),
            jobs=gate_jobs,
        ),
        followup_jobs=jobs,
    )


def evaluate_main_gate(queue: QueueManifest) -> GateStatus:
    reasons: list[str] = []
    for job in queue.main_gate.jobs:
        counts = _ledger_counts(job.ledger)
        if counts is None:
            reasons.append(f"{job.model_key} ledger is missing")
            continue
        _validate_ledger_counts(job.model_key, counts, job.expected_tasks, GateFailure)
        if counts.get("completed", 0) != job.expected_tasks:
            reasons.append(f"{job.model_key} ledger is incomplete")

    runtime_path = queue.main_gate.runtime_record
    if not runtime_path.is_file():
        reasons.append("runtime record is missing")
    else:
        runtime = _read_json(runtime_path)
        caches = {
            str(row.get("cache_key")): row
            for row in runtime.get("caches", [])
            if isinstance(row, dict)
        }
        for job in queue.main_gate.jobs:
            row = caches.get(job.runtime_cache_key)
            if row is None:
                reasons.append(f"{job.runtime_cache_key} runtime cache is missing")
                continue
            status = str(row.get("status", ""))
            if status == "failure":
                raise GateFailure(f"{job.runtime_cache_key} runtime cache reports failure")
            if status != "complete":
                reasons.append(f"{job.runtime_cache_key} runtime cache is {status or 'unknown'}")
    return GateStatus(ready=not reasons, reasons=tuple(reasons))


def evaluate_capacity(
    queue: QueueManifest,
    *,
    statvfs_fn: Callable[[Path], Any] = os.statvfs,
) -> CapacityStatus:
    gate = queue.capacity_gate
    filesystem = statvfs_fn(gate.filesystem_path)
    block_size = int(filesystem.f_frsize)
    total_bytes = int(filesystem.f_blocks) * block_size
    used_bytes = (int(filesystem.f_blocks) - int(filesystem.f_bfree)) * block_size
    free_bytes = int(filesystem.f_bavail) * block_size
    total_inodes = int(filesystem.f_files)
    free_inodes = int(filesystem.f_favail)
    used_inodes = total_inodes - int(filesystem.f_ffree)
    projected_bytes = 0
    projected_inodes = 0
    model_records: list[dict[str, Any]] = []
    for model in gate.models:
        calibration_bytes, _, calibration_tasks = _artifact_stats(model.calibration_root)
        if calibration_tasks <= 0:
            raise CapacityFailure(f"{model.model_key} calibration has no cache artifacts")
        average_bytes = calibration_bytes / calibration_tasks
        expected_tasks = sum(output.expected_tasks for output in model.outputs)
        current_bytes = 0
        current_files = 0
        current_tasks = 0
        for output in model.outputs:
            artifact_bytes, artifact_files, artifact_tasks = _artifact_stats(output.output_root)
            current_bytes += artifact_bytes
            current_files += artifact_files
            current_tasks += artifact_tasks
        projected_final_bytes = math.ceil(average_bytes * expected_tasks)
        additional_bytes = max(projected_final_bytes - current_bytes, 0)
        additional_files = max(expected_tasks * 2 - current_files, 0)
        projected_bytes += additional_bytes
        projected_inodes += additional_files
        model_records.append(
            {
                "model_key": model.model_key,
                "calibration_root": str(model.calibration_root),
                "calibration_tasks": calibration_tasks,
                "average_artifact_bytes_per_task": average_bytes,
                "expected_tasks": expected_tasks,
                "current_tasks": current_tasks,
                "current_artifact_bytes": current_bytes,
                "projected_additional_bytes": additional_bytes,
                "projected_additional_inodes": additional_files,
            }
        )
    projected_used_bytes = used_bytes + projected_bytes
    df_capacity_bytes = used_bytes + free_bytes
    projected_utilization = (
        projected_used_bytes / df_capacity_bytes if df_capacity_bytes else 1.0
    )
    df_capacity_inodes = used_inodes + free_inodes
    projected_inode_utilization = (
        (used_inodes + projected_inodes) / df_capacity_inodes
        if df_capacity_inodes
        else 1.0
    )
    safe = (
        projected_utilization < gate.max_projected_utilization
        and projected_inode_utilization < gate.max_projected_utilization
    )
    return CapacityStatus(
        safe=safe,
        filesystem_path=gate.filesystem_path,
        total_bytes=total_bytes,
        used_bytes=used_bytes,
        free_bytes=free_bytes,
        projected_bytes=projected_bytes,
        projected_used_bytes=projected_used_bytes,
        projected_utilization=projected_utilization,
        total_inodes=total_inodes,
        free_inodes=free_inodes,
        projected_inodes=projected_inodes,
        projected_inode_utilization=projected_inode_utilization,
        max_projected_utilization=gate.max_projected_utilization,
        models=tuple(model_records),
    )


def wait_for_main_gate(
    queue: QueueManifest,
    *,
    watcher_factory: WatcherFactory = lambda paths: InotifyArtifactWatcher(paths),
) -> None:
    watcher = watcher_factory(_gate_artifacts(queue))
    try:
        while True:
            status = evaluate_main_gate(queue)
            _write_queue_runtime(
                queue,
                status="waiting" if not status.ready else "ready",
                gate=status,
            )
            if status.ready:
                return
            watcher.wait()
    finally:
        watcher.close()


def run_dependency_queue(
    queue: QueueManifest,
    *,
    watcher_factory: WatcherFactory = lambda paths: InotifyArtifactWatcher(paths),
    job_executor: JobExecutor | None = None,
    retry_failed: bool = False,
) -> None:
    capacity = evaluate_capacity(queue)
    _write_queue_runtime(
        queue,
        status="capacity_ready" if capacity.safe else "blocked_capacity",
        gate=evaluate_main_gate(queue),
        capacity=capacity,
    )
    capacity.require_safe()
    wait_for_main_gate(queue, watcher_factory=watcher_factory)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(queue.physical_gpu)
    environment["PYTHONNOUSERSITE"] = "1"
    executor = job_executor or (
        lambda job, *, environment: _execute_job(
            queue,
            job,
            environment=environment,
            retry_failed=retry_failed,
        )
    )
    capacity = evaluate_capacity(queue)
    capacity.require_safe()
    _write_queue_runtime(
        queue,
        status="running",
        gate=GateStatus(True, ()),
        capacity=capacity,
    )
    try:
        for job in queue.followup_jobs:
            capacity = evaluate_capacity(queue)
            capacity.require_safe()
            counts = _ledger_counts(job.output_root / "batch_state.sqlite3")
            if counts is not None:
                _validate_ledger_counts(
                    job.job_id,
                    counts,
                    job.expected_tasks,
                    QueueExecutionError,
                    allow_failed=retry_failed,
                )
            if counts is None or counts.get("completed", 0) != job.expected_tasks:
                _write_queue_runtime(
                    queue,
                    status="running",
                    gate=GateStatus(True, ()),
                    active_job=job.job_id,
                    capacity=capacity,
                )
                executor(job, environment=environment)
                counts = _ledger_counts(job.output_root / "batch_state.sqlite3")
                if counts is None:
                    raise QueueExecutionError(f"{job.job_id} did not create a ledger")
                _validate_ledger_counts(
                    job.job_id,
                    counts,
                    job.expected_tasks,
                    QueueExecutionError,
                )
                if counts.get("completed", 0) != job.expected_tasks:
                    raise QueueExecutionError(f"{job.job_id} ended before exact completion")
            summary = job.output_root / "batch_summary.json"
            if not summary.is_file():
                raise QueueExecutionError(f"{job.job_id} did not create batch_summary.json")
            snapshot_cache_summary(
                queue.runtime_record,
                cache_key=job.job_id,
                summary_path=summary,
            )
            _write_queue_runtime(queue, status="running", gate=GateStatus(True, ()))
    except Exception as exc:
        _write_queue_runtime(
            queue,
            status="failure",
            gate=GateStatus(True, ()),
            error=f"{type(exc).__name__}: {exc}",
            capacity=capacity,
        )
        raise
    _write_queue_runtime(
        queue,
        status="complete",
        gate=GateStatus(True, ()),
        capacity=evaluate_capacity(queue),
    )


class InotifyArtifactWatcher:
    """Block on relevant artifact changes without time-based polling."""

    _EVENT = struct.Struct("iIII")
    _MASK = 0x00000002 | 0x00000004 | 0x00000008 | 0x00000080 | 0x00000100

    def __init__(self, paths: Sequence[Path]) -> None:
        self.targets = {path.expanduser().resolve() for path in paths}
        for target in self.targets:
            target.parent.mkdir(parents=True, exist_ok=True)
        libc = ctypes.CDLL(None, use_errno=True)
        self._close = libc.close
        self.fd = int(libc.inotify_init1(os.O_CLOEXEC))
        if self.fd < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        self.watch_dirs: dict[int, Path] = {}
        for parent in sorted({path.parent for path in self.targets}):
            wd = int(libc.inotify_add_watch(self.fd, os.fsencode(parent), self._MASK))
            if wd < 0:
                errno = ctypes.get_errno()
                self.close()
                raise OSError(errno, os.strerror(errno), parent)
            self.watch_dirs[wd] = parent

    def wait(self) -> None:
        while True:
            data = os.read(self.fd, 65536)
            offset = 0
            while offset < len(data):
                wd, _, _, name_length = self._EVENT.unpack_from(data, offset)
                offset += self._EVENT.size
                raw_name = data[offset : offset + name_length]
                offset += name_length
                name = os.fsdecode(raw_name.split(b"\0", 1)[0])
                if name and (self.watch_dirs[wd] / name).resolve() in self.targets:
                    return

    def close(self) -> None:
        if getattr(self, "fd", -1) >= 0:
            self._close(self.fd)
            self.fd = -1


def build_job_argv(queue: QueueManifest, job: FollowupJob, *, retry_failed: bool) -> list[str]:
    argv = [
        str(queue.python),
        str(queue.extract_script),
        "--manifest",
        str(job.manifest),
        "--prompt-set",
        str(job.prompt_set),
        "--protocol",
        job.protocol,
        "--model-key",
        job.model_key,
        "--device",
        queue.device,
        "--output-root",
        str(job.output_root),
        "--fail-fast",
        "--materialize-every",
        "100",
        *job.extra_args,
    ]
    if retry_failed:
        argv.append("--retry-failed")
    return argv


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an event-gated prefill dependency queue.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    queue = load_queue_manifest(args.config)
    run_dependency_queue(queue, retry_failed=args.retry_failed)
    return 0


def _execute_job(
    queue: QueueManifest,
    job: FollowupJob,
    *,
    environment: dict[str, str],
    retry_failed: bool,
) -> None:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_root.parent.mkdir(parents=True, exist_ok=True)
    argv = build_job_argv(queue, job, retry_failed=retry_failed)
    with job.log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            argv,
            cwd=Path.cwd(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise QueueExecutionError(f"{job.job_id} exited with code {completed.returncode}")


def _write_queue_runtime(
    queue: QueueManifest,
    *,
    status: str,
    gate: GateStatus,
    active_job: str | None = None,
    error: str = "",
    capacity: CapacityStatus | None = None,
) -> None:
    payload = load_run_records(queue.runtime_record)
    payload["class_code"] = CLASS_CODE
    payload["class_code_semantics"] = CLASS_CODE_SEMANTICS
    cache_status = {
        str(row.get("cache_key")): str(row.get("status"))
        for row in payload.get("caches", [])
        if isinstance(row, dict)
    }
    existing_queue = payload.get("dependency_queue", {})
    capacity_payload = (
        _capacity_payload(capacity)
        if capacity is not None
        else existing_queue.get("capacity")
        if isinstance(existing_queue, dict)
        else None
    )
    payload["dependency_queue"] = {
        "schema": QUEUE_SCHEMA,
        "manifest_path": str(queue.source_path),
        "manifest_sha256": hashlib.sha256(queue.source_path.read_bytes()).hexdigest(),
        "status": status,
        "active_job": active_job,
        "error": error,
        "physical_gpu": queue.physical_gpu,
        "device": queue.device,
        "capacity": capacity_payload,
        "gate": {
            "ready": gate.ready,
            "reasons": list(gate.reasons),
            "runtime_record": str(queue.main_gate.runtime_record),
            "jobs": [
                {
                    "model_key": job.model_key,
                    "ledger": str(job.ledger),
                    "expected_tasks": job.expected_tasks,
                    "runtime_cache_key": job.runtime_cache_key,
                }
                for job in queue.main_gate.jobs
            ],
        },
        "jobs": [
            {
                "job_id": job.job_id,
                "seed": job.seed,
                "model_key": job.model_key,
                "expected_tasks": job.expected_tasks,
                "output_root": str(job.output_root),
                "log_path": str(job.log_path),
                "status": "complete" if cache_status.get(job.job_id) == "complete" else "pending",
            }
            for job in queue.followup_jobs
        ],
        "recorded_at": utc_now(),
    }
    write_run_records(queue.runtime_record, payload)


def _gate_artifacts(queue: QueueManifest) -> tuple[Path, ...]:
    paths = [queue.main_gate.runtime_record]
    for job in queue.main_gate.jobs:
        paths.extend(
            (
                job.ledger.parent / "batch_summary.json",
                job.ledger.parent / "failures.jsonl",
            )
        )
    return tuple(paths)


def _artifact_stats(root: Path) -> tuple[int, int, int]:
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        return 0, 0, 0
    total_bytes = 0
    file_count = 0
    task_count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cache_root = Path(str(entry["cache_root"]))
        shard = cache_root / str(entry["shard_path"])
        sidecar = cache_root / str(entry["metadata"]["sidecar_path"])
        for artifact in (shard, sidecar):
            if not artifact.is_file():
                raise CapacityFailure(f"Capacity artifact is missing: {artifact}")
            total_bytes += artifact.stat().st_size
            file_count += 1
        task_count += 1
    return total_bytes, file_count, task_count


def _capacity_payload(status: CapacityStatus) -> dict[str, Any]:
    return {
        "safe": status.safe,
        "filesystem_path": str(status.filesystem_path),
        "total_bytes": status.total_bytes,
        "used_bytes": status.used_bytes,
        "free_bytes": status.free_bytes,
        "projected_bytes": status.projected_bytes,
        "projected_used_bytes": status.projected_used_bytes,
        "projected_utilization": status.projected_utilization,
        "total_inodes": status.total_inodes,
        "free_inodes": status.free_inodes,
        "projected_inodes": status.projected_inodes,
        "projected_inode_utilization": status.projected_inode_utilization,
        "max_projected_utilization": status.max_projected_utilization,
        "models": list(status.models),
        "recorded_at": utc_now(),
    }


def _ledger_counts(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        rows = connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status").fetchall()
    except sqlite3.OperationalError:
        return None
    finally:
        if "connection" in locals():
            connection.close()
    return {str(status): int(count) for status, count in rows}


def _validate_ledger_counts(
    key: str,
    counts: dict[str, int],
    expected_tasks: int,
    error_type: type[RuntimeError],
    *,
    allow_failed: bool = False,
) -> None:
    total = sum(counts.values())
    if total != expected_tasks:
        raise error_type(f"{key} expected {expected_tasks} tasks, found {total}")
    failed = counts.get("failed", 0)
    if failed and not allow_failed:
        raise error_type(f"{key} failed={failed}")


def _load_gate_job(data: Any) -> GateJob:
    if not isinstance(data, dict):
        raise ValueError("Main-gate jobs must be mappings")
    return GateJob(
        model_key=_required_str(data, "model_key"),
        ledger=Path(_required_str(data, "ledger")).expanduser(),
        expected_tasks=_positive_int(data, "expected_tasks"),
        runtime_cache_key=_required_str(data, "runtime_cache_key"),
    )


def _load_capacity_gate(data: dict[str, Any]) -> CapacityGate:
    maximum = data.get("max_projected_utilization")
    if not isinstance(maximum, int | float) or isinstance(maximum, bool):
        raise ValueError("max_projected_utilization must be numeric")
    if not 0 < float(maximum) < 1:
        raise ValueError("max_projected_utilization must be between zero and one")
    models = tuple(_load_capacity_model(item) for item in _required_list(data, "models"))
    if not models:
        raise ValueError("capacity_gate.models must not be empty")
    return CapacityGate(
        filesystem_path=Path(_required_str(data, "filesystem_path")).expanduser(),
        max_projected_utilization=float(maximum),
        models=models,
    )


def _load_capacity_model(data: Any) -> CapacityModel:
    if not isinstance(data, dict):
        raise ValueError("Capacity models must be mappings")
    outputs = tuple(_load_capacity_output(item) for item in _required_list(data, "outputs"))
    if not outputs:
        raise ValueError("Capacity model outputs must not be empty")
    return CapacityModel(
        model_key=_required_str(data, "model_key"),
        calibration_root=Path(_required_str(data, "calibration_root")).expanduser(),
        outputs=outputs,
    )


def _load_capacity_output(data: Any) -> CapacityOutput:
    if not isinstance(data, dict):
        raise ValueError("Capacity outputs must be mappings")
    return CapacityOutput(
        output_root=Path(_required_str(data, "output_root")).expanduser(),
        expected_tasks=_positive_int(data, "expected_tasks"),
    )


def _load_followup_job(data: Any) -> FollowupJob:
    if not isinstance(data, dict):
        raise ValueError("Follow-up jobs must be mappings")
    extra_args = data.get("extra_args", [])
    if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
        raise ValueError("extra_args must be a list of strings")
    protocol = _required_str(data, "protocol")
    if protocol not in {"vt", "va"}:
        raise ValueError(f"Unsupported follow-up protocol: {protocol}")
    return FollowupJob(
        job_id=_required_str(data, "job_id"),
        seed=_positive_int(data, "seed"),
        model_key=_required_str(data, "model_key"),
        protocol=protocol,
        manifest=Path(_required_str(data, "manifest")).expanduser(),
        prompt_set=Path(_required_str(data, "prompt_set")).expanduser(),
        output_root=Path(_required_str(data, "output_root")).expanduser(),
        log_path=Path(_required_str(data, "log_path")).expanduser(),
        expected_tasks=_positive_int(data, "expected_tasks"),
        extra_args=tuple(extra_args),
    )


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _positive_int(data: dict[str, Any], key: str, *, allow_zero: bool = False) -> int:
    value = data.get(key)
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GateFailure(f"Runtime record must contain an object: {path}")
    return data
