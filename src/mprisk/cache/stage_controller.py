"""Durable, fail-closed source-to-target cache stage controller."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mprisk.cache.cache_matrix_queue import (
    MatrixConfig,
    _ledger_status,
    _scoped_execution_paths,
    audit_matrix,
    load_matrix_config,
)

CONTROLLER_SCHEMA = "mprisk_cache_stage_controller_v1"
AUDIT_SCHEMA = "mprisk_complete_cache_matrix_audit_v1"
NONBLOCKING_STATUSES = frozenset({"ready", "complete", "accepted_bundle"})
TERMINAL_STATUSES = frozenset({"complete", "accepted_bundle"})
DEFAULT_LANE_STARTUP_TIMEOUT_SECONDS = 900.0
LANE_STARTUP_POLL_SECONDS = 0.25
MANAGER_LOG_TAIL_LINES = 20


@dataclass(frozen=True)
class ControllerPaths:
    output_dir: Path
    status_json: Path
    run_status: Path
    event_log: Path
    lock_path: Path
    extraction_launch_audit: Path
    source_audit: Path
    final_audit: Path


def build_controller_paths(output_dir: Path) -> ControllerPaths:
    root = output_dir.expanduser().resolve()
    return ControllerPaths(
        output_dir=root,
        status_json=root / "status.json",
        run_status=root / "RUN_STATUS.md",
        event_log=root / "controller.log",
        lock_path=root / "controller.lock",
        extraction_launch_audit=root / "EXTRACTION_LAUNCH_AUDIT.json",
        source_audit=root / "SOURCE_COMPLETE_AUDIT.json",
        final_audit=root / "FINAL_CACHE_AUDIT.json",
    )


def summarize_stage(
    audit: dict[str, Any],
    *,
    stage: str,
    expected_jobs: int,
    expected_accepted: int,
) -> dict[str, Any]:
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("Unexpected cache audit schema")
    records = [record for record in audit.get("job_records", []) if record.get("domain") == stage]
    if len(records) != expected_jobs:
        raise ValueError(f"{stage} audit has {len(records)} jobs; expected {expected_jobs}")
    job_ids = [str(record.get("job_id")) for record in records]
    if len(set(job_ids)) != expected_jobs:
        raise ValueError(f"{stage} audit contains duplicate job IDs")
    counts = Counter(str(record.get("status")) for record in records)
    blocked = sorted(
        f"{record['job_id']}={record.get('status')}"
        for record in records
        if record.get("status") not in NONBLOCKING_STATUSES
    )
    signature_mismatches = sorted(
        str(record["job_id"])
        for record in records
        if not bool(record.get("asset_signature", {}).get("passed"))
        or (
            "cache_asset_signature" in record
            and not bool(record["cache_asset_signature"].get("passed"))
        )
    )
    missing = 0
    for record in records:
        if record.get("status") in TERMINAL_STATUSES:
            continue
        missing += int(record.get("ledger", {}).get("missing", record.get("expected_tasks", 0)))
    expected_complete = expected_jobs - expected_accepted
    strict_complete = (
        counts.get("complete", 0) == expected_complete
        and counts.get("accepted_bundle", 0) == expected_accepted
        and sum(counts.values()) == expected_jobs
        and missing == 0
        and not blocked
        and not signature_mismatches
    )
    return {
        "stage": stage,
        "expected_jobs": expected_jobs,
        "expected_complete": expected_complete,
        "expected_accepted": expected_accepted,
        "status_counts": dict(sorted(counts.items())),
        "missing_tasks": missing,
        "blocked": blocked,
        "signature_mismatches": signature_mismatches,
        "strict_complete": strict_complete,
        "records": records,
    }


def expected_accepted_jobs(config: MatrixConfig, stage: str) -> int:
    return sum(
        1
        for model in config.models
        if stage in model.accepted_bundle_domains and stage not in model.invalidated_domains
    )


def read_stage_progress(config: MatrixConfig, stage: str) -> dict[str, Any]:
    """Read only the canonical ledgers while a stage is still running."""
    records: list[dict[str, Any]] = []
    for job in config.jobs:
        if job.domain.domain != stage:
            continue
        accepted = (
            stage in job.model.accepted_bundle_domains
            and stage not in job.model.invalidated_domains
        )
        if accepted:
            status = "accepted_bundle"
            ledger = {"status": "accepted_bundle", "missing": 0}
        else:
            ledger = _ledger_status(job.output_root, job.domain.expected_tasks)
            ledger_status = str(ledger["status"])
            status = "ready" if ledger_status in {"absent", "incomplete"} else ledger_status
        records.append(
            {
                "job_id": job.job_id,
                "domain": stage,
                "gpu_lane": job.model.gpu_lane,
                "status": status,
                "expected_tasks": job.domain.expected_tasks,
                "ledger": ledger,
            }
        )
    counts = Counter(str(record["status"]) for record in records)
    expected_accepted = expected_accepted_jobs(config, stage)
    missing = sum(
        int(record["ledger"].get("missing", record["expected_tasks"]))
        for record in records
        if record["status"] not in TERMINAL_STATUSES
    )
    blocked = sorted(
        f"{record['job_id']}={record['status']}"
        for record in records
        if record["status"] not in NONBLOCKING_STATUSES
    )
    strict_complete = (
        len(records) == 16
        and counts.get("complete", 0) == 16 - expected_accepted
        and counts.get("accepted_bundle", 0) == expected_accepted
        and sum(counts.values()) == 16
        and missing == 0
        and not blocked
    )
    return {
        "stage": stage,
        "expected_jobs": 16,
        "expected_complete": 16 - expected_accepted,
        "expected_accepted": expected_accepted,
        "status_counts": dict(sorted(counts.items())),
        "missing_tasks": missing,
        "blocked": blocked,
        "signature_mismatches": [],
        "strict_complete": strict_complete,
        "audit_level": "ledger_progress",
        "records": records,
    }


def tmux_session_exists(session: str) -> bool:
    completed = subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def lane_supervisor_status(
    config: MatrixConfig,
    *,
    stage: str,
    lane: int,
    session: str,
) -> dict[str, Any]:
    lock_path, runtime_record = _scoped_execution_paths(config, stage=stage, lane=lane)
    lock_pid: int | None = None
    lock_error: str | None = None
    if lock_path.is_file():
        try:
            lock_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            lock_error = str(exc)
    pid_alive = False
    if lock_pid is not None:
        try:
            os.kill(lock_pid, 0)
            pid_alive = True
        except ProcessLookupError:
            pid_alive = False
        except PermissionError:
            pid_alive = True
    session_exists = tmux_session_exists(session)
    return {
        "stage": stage,
        "lane": lane,
        "session": session,
        "session_exists": session_exists,
        "lock_path": str(lock_path),
        "lock_exists": lock_path.is_file(),
        "lock_pid": lock_pid,
        "lock_pid_alive": pid_alive,
        "lock_error": lock_error,
        "runtime_record": str(runtime_record),
        "runtime_record_exists": runtime_record.is_file(),
        "active": session_exists and lock_path.is_file() and pid_alive,
    }


def validate_active_lanes(
    config: MatrixConfig,
    summary: dict[str, Any],
    sessions: dict[int, str],
) -> list[dict[str, Any]]:
    pending_lanes = {
        int(record["gpu_lane"])
        for record in summary["records"]
        if record.get("status") not in TERMINAL_STATUSES
    }
    statuses = [
        lane_supervisor_status(
            config,
            stage=str(summary["stage"]),
            lane=lane,
            session=sessions[lane],
        )
        for lane in sorted(pending_lanes)
    ]
    inactive = [status for status in statuses if not status["active"]]
    if inactive:
        detail = ", ".join(
            f"lane{status['lane']} session={status['session_exists']} "
            f"lock={status['lock_exists']} pid_alive={status['lock_pid_alive']}"
            for status in inactive
        )
        raise RuntimeError(
            f"{summary['stage']} is incomplete but its supervisor is inactive: {detail}"
        )
    return statuses


def stage_is_finalized(
    config: MatrixConfig, *, stage: str, sessions: dict[int, str]
) -> tuple[bool, list[dict[str, Any]]]:
    statuses = [
        lane_supervisor_status(config, stage=stage, lane=lane, session=sessions[lane])
        for lane in (0, 1)
    ]
    finalized = all(
        not status["session_exists"] and not status["lock_exists"] for status in statuses
    )
    return finalized, statuses


def stage_lane_statuses(
    config: MatrixConfig, *, stage: str, sessions: dict[int, str]
) -> list[dict[str, Any]]:
    return [
        lane_supervisor_status(config, stage=stage, lane=lane, session=sessions[lane])
        for lane in (0, 1)
    ]


def plan_target_lane_launches(
    *,
    source_statuses: list[dict[str, Any]],
    target_statuses: list[dict[str, Any]],
    pending_target_lanes: set[int],
    allow_launch: bool,
) -> dict[str, list[int]]:
    source_owned = {
        int(status["lane"])
        for status in source_statuses
        if status["session_exists"] or status["lock_exists"]
    }
    target_by_lane = {
        int(status["lane"]): status
        for status in target_statuses
        if int(status["lane"]) in pending_target_lanes
    }
    missing = sorted(pending_target_lanes.difference(target_by_lane))
    if missing:
        raise ValueError(f"Missing target supervisor status for lanes {missing}")
    target_active = {lane for lane, status in target_by_lane.items() if status["active"]}
    overlaps = sorted(source_owned.intersection(target_active))
    if overlaps:
        raise RuntimeError(
            "Source and target supervisors claim the same GPU lanes: "
            + ", ".join(str(lane) for lane in overlaps)
        )
    inconsistent = sorted(
        lane
        for lane, status in target_by_lane.items()
        if not status["active"] and (status["session_exists"] or status["lock_exists"])
    )
    if inconsistent:
        raise RuntimeError(
            "Target lane ownership markers exist without an active supervisor: "
            + ", ".join(str(lane) for lane in inconsistent)
        )
    waiting = pending_target_lanes.intersection(source_owned)
    launchable = (
        pending_target_lanes.difference(source_owned, target_active) if allow_launch else set()
    )
    if not allow_launch and target_active:
        raise RuntimeError("Target supervisors are active before the serial source gate passed")
    return {
        "source_owned": sorted(source_owned),
        "target_active": sorted(target_active),
        "target_waiting": sorted(waiting),
        "target_launchable": sorted(launchable),
    }


def build_stage_lane_command(
    config: MatrixConfig,
    *,
    python: Path,
    stage: str,
    lane: int,
) -> list[str]:
    if stage not in {"source", "target"}:
        raise ValueError(f"Unsupported cache stage: {stage!r}")
    if lane not in {0, 1}:
        raise ValueError(f"Unsupported GPU lane: {lane!r}")
    return [
        "env",
        f"PYTHONPATH={config.repo_root / 'src'}",
        str(python),
        str(config.repo_root / "scripts" / "run_cache_matrix_queue.py"),
        "--config",
        str(config.source_path),
        "--execute",
        "--stage",
        stage,
        "--lane",
        str(lane),
        "--wait-for-gpu",
    ]


def _manager_log_tail(path: Path) -> str:
    if not path.is_file():
        return "<manager log not created>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-MANAGER_LOG_TAIL_LINES:])


def wait_for_lane_startup(
    config: MatrixConfig,
    *,
    stage: str,
    lane: int,
    session: str,
    manager_log: Path,
    timeout_seconds: float,
    poll_seconds: float = LANE_STARTUP_POLL_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("Lane startup timeout must be positive")
    if poll_seconds <= 0:
        raise ValueError("Lane startup poll interval must be positive")
    deadline = monotonic_fn() + timeout_seconds
    while True:
        status = lane_supervisor_status(config, stage=stage, lane=lane, session=session)
        if status["active"]:
            return status
        if not status["session_exists"]:
            raise RuntimeError(
                f"{stage} lane {lane} startup failed before acquiring its live lock; "
                f"manager log tail:\n{_manager_log_tail(manager_log)}"
            )
        now = monotonic_fn()
        if now >= deadline:
            subprocess.run(
                ["tmux", "kill-session", "-t", session],
                check=False,
                capture_output=True,
            )
            raise TimeoutError(
                f"{stage} lane {lane} did not acquire its live lock within "
                f"{timeout_seconds:.1f}s; manager log tail:\n"
                f"{_manager_log_tail(manager_log)}"
            )
        sleep_fn(min(poll_seconds, deadline - now))


def launch_target_lanes(
    config: MatrixConfig,
    *,
    source_sessions: dict[int, str],
    sessions: dict[int, str],
    manager_logs: dict[int, Path],
    python: Path,
    lanes: tuple[int, ...] = (0, 1),
    startup_timeout_seconds: float = DEFAULT_LANE_STARTUP_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    selected_lanes = tuple(sorted(set(lanes)))
    if not selected_lanes or any(lane not in {0, 1} for lane in selected_lanes):
        raise ValueError("Target lanes must be a non-empty subset of {0, 1}")
    for lane in selected_lanes:
        source_status = lane_supervisor_status(
            config,
            stage="source",
            lane=lane,
            session=source_sessions[lane],
        )
        if source_status["session_exists"] or source_status["lock_exists"]:
            raise RuntimeError(f"Source lane {lane} still owns GPU {lane}; refusing target launch")
        lock_path, _ = _scoped_execution_paths(config, stage="target", lane=lane)
        if lock_path.exists():
            raise RuntimeError(f"Target lane {lane} lock already exists: {lock_path}")
        if tmux_session_exists(sessions[lane]):
            raise RuntimeError(f"Target lane {lane} tmux session already exists: {sessions[lane]}")
    launched: list[dict[str, Any]] = []
    for lane in selected_lanes:
        manager_logs[lane].parent.mkdir(parents=True, exist_ok=True)
        command = build_stage_lane_command(config, python=python, stage="target", lane=lane)
        shell_command = (
            "set -o pipefail; "
            + shlex.join(command)
            + " 2>&1 | tee -a "
            + shlex.quote(str(manager_logs[lane]))
        )
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                sessions[lane],
                "-c",
                str(config.repo_root),
                shell_command,
            ],
            check=True,
        )
        pane = subprocess.run(
            [
                "tmux",
                "list-panes",
                "-t",
                sessions[lane],
                "-F",
                "#{pane_pid}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        startup_status = wait_for_lane_startup(
            config,
            stage="target",
            lane=lane,
            session=sessions[lane],
            manager_log=manager_logs[lane],
            timeout_seconds=startup_timeout_seconds,
        )
        launched.append(
            {
                "lane": lane,
                "session": sessions[lane],
                "pane_pid": int(pane.stdout.strip()),
                "manager_log": str(manager_logs[lane]),
                "command": command,
                "startup_status": startup_status,
            }
        )
    return launched


class StageController:
    def __init__(
        self,
        config: MatrixConfig,
        *,
        paths: ControllerPaths,
        poll_interval_seconds: float,
        source_sessions: dict[int, str],
        target_sessions: dict[int, str],
        lane_startup_timeout_seconds: float = DEFAULT_LANE_STARTUP_TIMEOUT_SECONDS,
        audit_fn: Callable[[MatrixConfig], dict[str, Any]] = audit_matrix,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if lane_startup_timeout_seconds <= 0:
            raise ValueError("lane_startup_timeout_seconds must be positive")
        self.config = config
        self.paths = paths
        self.poll_interval_seconds = poll_interval_seconds
        self.source_sessions = source_sessions
        self.target_sessions = target_sessions
        self.lane_startup_timeout_seconds = lane_startup_timeout_seconds
        self.audit_fn = audit_fn
        self.sleep_fn = sleep_fn
        self.target_launches: list[dict[str, Any]] = []
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, message: str) -> None:
        line = f"{_timestamp()} {message}"
        print(line, flush=True)
        with self.paths.event_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_status(
        self,
        status: str,
        *,
        source: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
        supervisors: list[dict[str, Any]] | None = None,
        lane_ownership: dict[str, list[int]] | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "schema": CONTROLLER_SCHEMA,
            "status": status,
            "updated_at": _timestamp(),
            "pid": os.getpid(),
            "config": str(self.config.source_path),
            "git_head": _git_head(self.config.repo_root),
            "poll_interval_seconds": self.poll_interval_seconds,
            "allow_parallel_domain_extraction": (self.config.allow_parallel_domain_extraction),
            "source": _compact_summary(source),
            "target": _compact_summary(target),
            "supervisors": supervisors or [],
            "lane_ownership": lane_ownership or {},
            "target_launches": self.target_launches,
            "error": error,
            "extraction_launch_audit": str(self.paths.extraction_launch_audit),
            "source_audit": str(self.paths.source_audit),
            "final_audit": str(self.paths.final_audit),
            "event_log": str(self.paths.event_log),
        }
        _atomic_json(self.paths.status_json, payload)
        _atomic_text(self.paths.run_status, _status_markdown(payload))

    def run(self) -> int:
        lock_fd = os.open(self.paths.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        os.close(lock_fd)
        source: dict[str, Any] | None = None
        target: dict[str, Any] | None = None
        source_gate_passed = False
        extraction_launch_audit_passed = False
        try:
            self.emit("controller_started")
            while True:
                cycle_audit: dict[str, Any] | None = None
                source = read_stage_progress(self.config, "source")
                target = read_stage_progress(self.config, "target")
                for summary in (source, target):
                    if not summary["blocked"] and not summary["signature_mismatches"]:
                        continue
                    raise RuntimeError(
                        f"{str(summary['stage']).title()} audit failed: "
                        + json.dumps(
                            {
                                "blocked": summary["blocked"],
                                "signature_mismatches": summary["signature_mismatches"],
                            },
                            sort_keys=True,
                        )
                    )
                source_finalized = False
                if source["strict_complete"]:
                    source_finalized, _ = stage_is_finalized(
                        self.config,
                        stage="source",
                        sessions=self.source_sessions,
                    )
                else:
                    validate_active_lanes(self.config, source, self.source_sessions)
                if (
                    self.config.allow_parallel_domain_extraction
                    and not target["strict_complete"]
                    and not extraction_launch_audit_passed
                ):
                    launch_audit = self.audit_fn(self.config)
                    cycle_audit = launch_audit
                    audited_source = summarize_stage(
                        launch_audit,
                        stage="source",
                        expected_jobs=16,
                        expected_accepted=expected_accepted_jobs(self.config, "source"),
                    )
                    audited_target = summarize_stage(
                        launch_audit,
                        stage="target",
                        expected_jobs=16,
                        expected_accepted=expected_accepted_jobs(self.config, "target"),
                    )
                    launch_blockers = (
                        audited_source["blocked"]
                        + audited_source["signature_mismatches"]
                        + audited_target["blocked"]
                        + audited_target["signature_mismatches"]
                    )
                    if launch_blockers or not launch_audit.get("ready_to_launch"):
                        raise RuntimeError(
                            "Parallel extraction launch audit failed: "
                            + json.dumps(
                                {
                                    "blockers": launch_blockers,
                                    "ready_to_launch": launch_audit.get("ready_to_launch"),
                                },
                                sort_keys=True,
                            )
                        )
                    _atomic_json(self.paths.extraction_launch_audit, launch_audit)
                    extraction_launch_audit_passed = True
                    self.emit("parallel_extraction_launch_audit_complete")
                if source["strict_complete"] and source_finalized and not source_gate_passed:
                    audit = cycle_audit if cycle_audit is not None else self.audit_fn(self.config)
                    source = summarize_stage(
                        audit,
                        stage="source",
                        expected_jobs=16,
                        expected_accepted=expected_accepted_jobs(self.config, "source"),
                    )
                    target = summarize_stage(
                        audit,
                        stage="target",
                        expected_jobs=16,
                        expected_accepted=expected_accepted_jobs(self.config, "target"),
                    )
                    if not source["strict_complete"]:
                        raise RuntimeError("Source ledger candidate failed the full strict audit")
                    if not audit.get("ready_to_launch") and not target["strict_complete"]:
                        raise RuntimeError(
                            "Full strict audit is not launchable after source completion"
                        )
                    _atomic_json(self.paths.source_audit, audit)
                    self.emit("source_audit_complete")
                    if target["blocked"] or target["signature_mismatches"]:
                        raise RuntimeError(
                            "Target prelaunch audit failed: "
                            + json.dumps(
                                {
                                    "blocked": target["blocked"],
                                    "signature_mismatches": target["signature_mismatches"],
                                },
                                sort_keys=True,
                            )
                        )
                    source_gate_passed = True
                source_lane_status = stage_lane_statuses(
                    self.config,
                    stage="source",
                    sessions=self.source_sessions,
                )
                pending_target_lanes = {
                    int(record["gpu_lane"])
                    for record in target["records"]
                    if record.get("status") not in TERMINAL_STATUSES
                }
                target_lane_status = stage_lane_statuses(
                    self.config,
                    stage="target",
                    sessions=self.target_sessions,
                )
                allow_target_launch = (
                    self.config.allow_parallel_domain_extraction or source_gate_passed
                )
                lane_ownership = plan_target_lane_launches(
                    source_statuses=source_lane_status,
                    target_statuses=target_lane_status,
                    pending_target_lanes=pending_target_lanes,
                    allow_launch=allow_target_launch,
                )
                launchable = tuple(lane_ownership["target_launchable"])
                if launchable:
                    manager_logs = {
                        lane: self.paths.output_dir / f"target_gpu{lane}.manager.log"
                        for lane in launchable
                    }
                    for lane in launchable:
                        launches = launch_target_lanes(
                            self.config,
                            source_sessions=self.source_sessions,
                            sessions=self.target_sessions,
                            manager_logs={lane: manager_logs[lane]},
                            python=Path(sys.executable).resolve(),
                            lanes=(lane,),
                            startup_timeout_seconds=(self.lane_startup_timeout_seconds),
                        )
                        self.target_launches.extend(launches)
                        self.emit(f"target_lane_launched lane={lane}")
                    target_lane_status = stage_lane_statuses(
                        self.config,
                        stage="target",
                        sessions=self.target_sessions,
                    )
                    lane_ownership = plan_target_lane_launches(
                        source_statuses=source_lane_status,
                        target_statuses=target_lane_status,
                        pending_target_lanes=pending_target_lanes,
                        allow_launch=allow_target_launch,
                    )
                    inactive_launched = sorted(
                        set(launchable).difference(lane_ownership["target_active"])
                    )
                    if inactive_launched:
                        raise RuntimeError(
                            "Launched target lanes have no active supervisor: "
                            + ", ".join(str(lane) for lane in inactive_launched)
                        )
                if source_gate_passed and source["strict_complete"] and target["strict_complete"]:
                    audit = self.audit_fn(self.config)
                    source = summarize_stage(
                        audit,
                        stage="source",
                        expected_jobs=16,
                        expected_accepted=expected_accepted_jobs(self.config, "source"),
                    )
                    target = summarize_stage(
                        audit,
                        stage="target",
                        expected_jobs=16,
                        expected_accepted=expected_accepted_jobs(self.config, "target"),
                    )
                    if not source["strict_complete"] or not target["strict_complete"]:
                        raise RuntimeError("Final ledger candidate failed the full strict audit")
                    target_finalized, target_supervisors = stage_is_finalized(
                        self.config,
                        stage="target",
                        sessions=self.target_sessions,
                    )
                    if not target_finalized:
                        self.write_status(
                            "target_finalizing",
                            source=source,
                            target=target,
                            supervisors=source_lane_status + target_supervisors,
                            lane_ownership=lane_ownership,
                        )
                        self.sleep_fn(self.poll_interval_seconds)
                        continue
                    _atomic_json(self.paths.final_audit, audit)
                    self.write_status("complete", source=source, target=target)
                    self.emit("cache_matrix_complete")
                    return 0
                if not source["strict_complete"]:
                    status = (
                        "monitoring_parallel_extraction"
                        if lane_ownership["target_active"]
                        else "monitoring_source"
                    )
                elif not source_finalized:
                    status = "source_finalizing"
                elif not target["strict_complete"]:
                    status = "monitoring_target"
                else:
                    status = "awaiting_final_audit"
                self.write_status(
                    status,
                    source=source,
                    target=target,
                    supervisors=source_lane_status + target_lane_status,
                    lane_ownership=lane_ownership,
                )
                self.sleep_fn(self.poll_interval_seconds)
        except Exception as exc:
            self.emit(f"controller_failed error={type(exc).__name__}: {exc}")
            self.write_status(
                "failed",
                source=source,
                target=target,
                error=f"{type(exc).__name__}: {exc}",
            )
            return 1
        finally:
            self.paths.lock_path.unlink(missing_ok=True)


def _compact_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {key: value for key, value in summary.items() if key != "records"}


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _status_markdown(payload: dict[str, Any]) -> str:
    source = payload.get("source") or {}
    target = payload.get("target") or {}
    lines = [
        "# Cache Matrix Stage Controller",
        "",
        f"- Status: `{payload['status']}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- PID: `{payload['pid']}`",
        f"- Git HEAD: `{payload['git_head']}`",
        f"- Config: `{payload['config']}`",
        (f"- Parallel source/target extraction: `{payload['allow_parallel_domain_extraction']}`"),
        "- API/Misread actions: `disabled`",
        "",
        "## Source",
        "",
        f"- Strict complete: `{source.get('strict_complete', False)}`",
        f"- Status counts: `{json.dumps(source.get('status_counts', {}), sort_keys=True)}`",
        f"- Missing tasks: `{source.get('missing_tasks', 'N/A')}`",
        "",
        "## Target",
        "",
        f"- Strict complete: `{target.get('strict_complete', False)}`",
        f"- Status counts: `{json.dumps(target.get('status_counts', {}), sort_keys=True)}`",
        f"- Missing tasks: `{target.get('missing_tasks', 'N/A')}`",
        "",
        "## Runtime",
        "",
        f"- Lane ownership: `{json.dumps(payload.get('lane_ownership', {}), sort_keys=True)}`",
        f"- Target launches: `{json.dumps(payload.get('target_launches', []), sort_keys=True)}`",
        f"- Extraction launch audit: `{payload['extraction_launch_audit']}`",
        f"- Event log: `{payload['event_log']}`",
    ]
    if payload.get("error"):
        lines.extend(["", "## Failure", "", f"`{payload['error']}`"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--source-session-prefix", default="mprisk-cache-source-gpu")
    parser.add_argument("--target-session-prefix", default="mprisk-cache-target-gpu")
    parser.add_argument(
        "--lane-startup-timeout-seconds",
        type=float,
        default=DEFAULT_LANE_STARTUP_TIMEOUT_SECONDS,
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_matrix_config(args.config)
    controller = StageController(
        config,
        paths=build_controller_paths(args.output_dir),
        poll_interval_seconds=args.poll_interval_seconds,
        source_sessions={lane: f"{args.source_session_prefix}{lane}" for lane in (0, 1)},
        target_sessions={lane: f"{args.target_session_prefix}{lane}" for lane in (0, 1)},
        lane_startup_timeout_seconds=args.lane_startup_timeout_seconds,
    )
    return controller.run()
