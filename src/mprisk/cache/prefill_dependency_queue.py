"""Event-driven dependent queue for sequential prefill-cache batches.

This module is the public entry point. Most of the implementation now lives in
focused submodules:

- :mod:`mprisk.cache._queue_manifest` -- dataclasses, exceptions, loaders
- :mod:`mprisk.cache._queue_capacity` -- disk/inode capacity gate evaluation
- :mod:`mprisk.cache._queue_watcher` -- inotify artifact watcher

The runner, gate evaluation and lock remain here so that tests that monkey-patch
``dependency_queue.evaluate_capacity`` / ``dependency_queue.subprocess`` /
``dependency_queue.time`` continue to work without changes.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mprisk.cache._queue_capacity import (
    _capacity_payload,
    evaluate_capacity,
)
from mprisk.cache._queue_manifest import (
    CLASS_CODE,
    CLASS_CODE_SEMANTICS,
    QUEUE_SCHEMA,
    CapacityFailure,
    CapacityGate,
    CapacityModel,
    CapacityOutput,
    CapacityStatus,
    EventWatcher,
    FollowupJob,
    GateFailure,
    GateJob,
    GateStatus,
    JobExecutor,
    MainGate,
    QueueExecutionError,
    QueueLockError,
    QueueManifest,
    UpstreamChecker,
    UpstreamConfig,
    UpstreamStatus,
    WatcherFactory,
    _read_json,
    load_queue_manifest,
)
from mprisk.cache._queue_watcher import InotifyArtifactWatcher
from mprisk.viz.runtime_records import (
    load_run_records,
    snapshot_cache_summary,
    utc_now,
    write_run_records,
)

__all__ = [
    "CLASS_CODE",
    "CLASS_CODE_SEMANTICS",
    "QUEUE_SCHEMA",
    "CapacityFailure",
    "CapacityGate",
    "CapacityModel",
    "CapacityOutput",
    "CapacityStatus",
    "EventWatcher",
    "FollowupJob",
    "GateFailure",
    "GateJob",
    "GateStatus",
    "InotifyArtifactWatcher",
    "JobExecutor",
    "MainGate",
    "QueueExecutionError",
    "QueueLockError",
    "QueueManifest",
    "QueueScopeLock",
    "UpstreamChecker",
    "UpstreamConfig",
    "UpstreamStatus",
    "WatcherFactory",
    "build_job_argv",
    "cli",
    "evaluate_capacity",
    "evaluate_main_gate",
    "evaluate_upstream_activity",
    "load_queue_manifest",
    "run_dependency_queue",
    "wait_for_main_gate",
]


class QueueScopeLock:
    """Hold a process-level exclusive lock for the complete queue lifecycle."""

    def __init__(self, queue: QueueManifest) -> None:
        self.queue = queue
        self.handle: Any | None = None

    def __enter__(self) -> QueueScopeLock:
        path = self.queue.lock_path
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise QueueLockError(f"Dependent queue scope is already locked: {path}") from exc
        try:
            metadata = {
                "pid": os.getpid(),
                "manifest_path": str(self.queue.source_path),
                "runtime_record": str(self.queue.runtime_record),
                "output_roots": [str(job.output_root) for job in self.queue.followup_jobs],
                "acquired_at": utc_now(),
            }
            handle.seek(0)
            handle.truncate()
            json.dump(metadata, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            raise
        self.handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


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


def evaluate_upstream_activity(queue: QueueManifest) -> UpstreamStatus:
    config = queue.main_gate.upstream
    try:
        completed = subprocess.run(
            ["tmux", "list-panes", "-t", config.tmux_session, "-F", "#{pane_pid}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return UpstreamStatus(False, f"tmux check failed: {exc}", None, 0)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "tmux session is missing"
        return UpstreamStatus(False, detail, None, 0)
    pane_pids = tuple(
        int(line)
        for line in completed.stdout.splitlines()
        if line.strip().isdigit() and int(line) > 0
    )
    if not pane_pids:
        return UpstreamStatus(False, "tmux session has no live pane PID", None, 0)
    process_pid = config.pid if config.pid is not None else pane_pids[0]
    if process_pid not in pane_pids:
        return UpstreamStatus(
            False,
            f"configured PID {process_pid} is not in tmux session {config.tmux_session}",
            None,
            0,
            process_pid,
        )
    if not _pid_is_alive(process_pid):
        return UpstreamStatus(False, f"PID {process_pid} is not running", None, 0, process_pid)

    heartbeat_times = []
    for path in config.heartbeat_paths:
        try:
            heartbeat_times.append(path.stat().st_mtime)
        except FileNotFoundError:
            continue
    if not heartbeat_times:
        return UpstreamStatus(
            False,
            "no upstream heartbeat artifact exists",
            None,
            0,
            process_pid,
        )
    heartbeat_age = max(time.time() - max(heartbeat_times), 0.0)
    seconds_until_stale = max(config.heartbeat_max_age_seconds - heartbeat_age, 0.0)
    if seconds_until_stale <= 0:
        return UpstreamStatus(
            False,
            f"upstream heartbeat is stale by {heartbeat_age:.1f} seconds",
            heartbeat_age,
            0,
            process_pid,
        )
    return UpstreamStatus(
        True,
        "",
        heartbeat_age,
        seconds_until_stale,
        process_pid,
    )


def wait_for_main_gate(
    queue: QueueManifest,
    *,
    watcher_factory: WatcherFactory = lambda paths, pid: InotifyArtifactWatcher(paths, pid),
    upstream_checker: UpstreamChecker = lambda queue: evaluate_upstream_activity(queue),
) -> None:
    watcher: EventWatcher | None = None
    watched_pid: int | None = None
    try:
        while True:
            status = evaluate_main_gate(queue)
            if status.ready:
                _write_queue_runtime(queue, status="ready", gate=status)
                return
            upstream = upstream_checker(queue)
            if not upstream.running:
                raise GateFailure(f"upstream_not_running: {upstream.reason}")
            if watcher is None or watched_pid != upstream.process_pid:
                if watcher is not None:
                    watcher.close()
                watcher = watcher_factory(_gate_artifacts(queue), upstream.process_pid)
                watched_pid = upstream.process_pid
                # Close the event-registration race before blocking.
                status = evaluate_main_gate(queue)
                if status.ready:
                    _write_queue_runtime(
                        queue,
                        status="ready",
                        gate=status,
                        upstream=upstream,
                    )
                    return
            _write_queue_runtime(
                queue,
                status="waiting",
                gate=status,
                upstream=upstream,
            )
            watcher.wait(upstream.seconds_until_stale)
    finally:
        if watcher is not None:
            watcher.close()


def run_dependency_queue(
    queue: QueueManifest,
    *,
    watcher_factory: WatcherFactory = lambda paths, pid: InotifyArtifactWatcher(paths, pid),
    upstream_checker: UpstreamChecker = lambda queue: evaluate_upstream_activity(queue),
    job_executor: JobExecutor | None = None,
    retry_failed: bool = False,
) -> None:
    capacity: CapacityStatus | None = None
    gate = GateStatus(False, ("gate not evaluated",))
    try:
        with QueueScopeLock(queue):
            gate = evaluate_main_gate(queue)
            capacity = evaluate_capacity(queue)
            _write_queue_runtime(
                queue,
                status="capacity_ready" if capacity.safe else "blocked_capacity",
                gate=gate,
                capacity=capacity,
            )
            capacity.require_safe()
            wait_for_main_gate(
                queue,
                watcher_factory=watcher_factory,
                upstream_checker=upstream_checker,
            )
            gate = GateStatus(True, ())
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
                gate=gate,
                capacity=capacity,
            )
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
                        gate=gate,
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
                _write_queue_runtime(queue, status="running", gate=gate)
            _write_queue_runtime(
                queue,
                status="complete",
                gate=gate,
                capacity=evaluate_capacity(queue),
            )
    except Exception as exc:
        failure_code = _failure_code(exc)
        upstream = None
        if failure_code == "upstream_not_running":
            try:
                upstream = upstream_checker(queue)
            except Exception:
                upstream = None
        _write_queue_runtime(
            queue,
            status="failure",
            gate=gate,
            error=f"{type(exc).__name__}: {exc}",
            failure_code=failure_code,
            capacity=capacity,
            upstream=upstream,
        )
        raise


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
    failure_code: str = "",
    capacity: CapacityStatus | None = None,
    upstream: UpstreamStatus | None = None,
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
    upstream_payload = (
        _upstream_payload(upstream)
        if upstream is not None
        else existing_queue.get("upstream")
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
        "failure_code": failure_code,
        "physical_gpu": queue.physical_gpu,
        "device": queue.device,
        "lock": _lock_payload(queue),
        "upstream": upstream_payload,
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


def _upstream_payload(status: UpstreamStatus) -> dict[str, Any]:
    return {
        "running": status.running,
        "reason": status.reason,
        "process_pid": status.process_pid,
        "heartbeat_age_seconds": status.heartbeat_age_seconds,
        "seconds_until_stale": status.seconds_until_stale,
        "recorded_at": utc_now(),
    }


def _lock_payload(queue: QueueManifest) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if queue.lock_path.is_file():
        try:
            loaded = json.loads(queue.lock_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        except (OSError, json.JSONDecodeError):
            metadata = {}
    return {"path": str(queue.lock_path), **metadata}


def _failure_code(exc: Exception) -> str:
    if "upstream_not_running" in str(exc):
        return "upstream_not_running"
    if isinstance(exc, QueueLockError):
        return "lock_unavailable"
    if isinstance(exc, CapacityFailure):
        return "capacity_gate"
    if isinstance(exc, GateFailure):
        return "gate_failure"
    return "queue_execution"


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
