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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mprisk.cache.cache_matrix_queue import (
    MatrixConfig,
    _scoped_execution_paths,
    audit_matrix,
    load_matrix_config,
)

CONTROLLER_SCHEMA = "mprisk_cache_stage_controller_v1"
AUDIT_SCHEMA = "mprisk_complete_cache_matrix_audit_v1"
NONBLOCKING_STATUSES = frozenset({"ready", "complete", "accepted_bundle"})
TERMINAL_STATUSES = frozenset({"complete", "accepted_bundle"})


@dataclass(frozen=True)
class ControllerPaths:
    output_dir: Path
    status_json: Path
    run_status: Path
    event_log: Path
    lock_path: Path
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
    records = [
        record
        for record in audit.get("job_records", [])
        if record.get("domain") == stage
    ]
    if len(records) != expected_jobs:
        raise ValueError(
            f"{stage} audit has {len(records)} jobs; expected {expected_jobs}"
        )
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
        missing += int(
            record.get("ledger", {}).get(
                "missing", record.get("expected_tasks", 0)
            )
        )
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
        if stage in model.accepted_bundle_domains
        and stage not in model.invalidated_domains
    )


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
    lock_path, runtime_record = _scoped_execution_paths(
        config, stage=stage, lane=lane
    )
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
        "active": session_exists and (not lock_path.is_file() or pid_alive),
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
        lane_supervisor_status(
            config, stage=stage, lane=lane, session=sessions[lane]
        )
        for lane in (0, 1)
    ]
    finalized = all(
        not status["session_exists"] and not status["lock_exists"]
        for status in statuses
    )
    return finalized, statuses


def launch_target_lanes(
    config: MatrixConfig,
    *,
    sessions: dict[int, str],
    manager_logs: dict[int, Path],
    python: Path,
) -> list[dict[str, Any]]:
    for lane in (0, 1):
        lock_path, _ = _scoped_execution_paths(config, stage="target", lane=lane)
        if lock_path.exists():
            raise RuntimeError(f"Target lane {lane} lock already exists: {lock_path}")
        if tmux_session_exists(sessions[lane]):
            raise RuntimeError(
                f"Target lane {lane} tmux session already exists: {sessions[lane]}"
            )
    launched: list[dict[str, Any]] = []
    for lane in (0, 1):
        manager_logs[lane].parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            str(config.repo_root / "scripts" / "run_cache_matrix_queue.py"),
            "--config",
            str(config.source_path),
            "--execute",
            "--stage",
            "target",
            "--lane",
            str(lane),
            "--wait-for-gpu",
        ]
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
        launched.append(
            {
                "lane": lane,
                "session": sessions[lane],
                "pane_pid": int(pane.stdout.strip()),
                "manager_log": str(manager_logs[lane]),
                "command": command,
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
        audit_fn: Callable[[MatrixConfig], dict[str, Any]] = audit_matrix,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.config = config
        self.paths = paths
        self.poll_interval_seconds = poll_interval_seconds
        self.source_sessions = source_sessions
        self.target_sessions = target_sessions
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
            "source": _compact_summary(source),
            "target": _compact_summary(target),
            "supervisors": supervisors or [],
            "target_launches": self.target_launches,
            "error": error,
            "source_audit": str(self.paths.source_audit),
            "final_audit": str(self.paths.final_audit),
            "event_log": str(self.paths.event_log),
        }
        _atomic_json(self.paths.status_json, payload)
        _atomic_text(self.paths.run_status, _status_markdown(payload))

    def run(self) -> int:
        lock_fd = os.open(
            self.paths.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
        )
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        os.close(lock_fd)
        source: dict[str, Any] | None = None
        target: dict[str, Any] | None = None
        try:
            self.emit("controller_started")
            while True:
                audit = self.audit_fn(self.config)
                source = summarize_stage(
                    audit,
                    stage="source",
                    expected_jobs=16,
                    expected_accepted=expected_accepted_jobs(
                        self.config, "source"
                    ),
                )
                target = summarize_stage(
                    audit,
                    stage="target",
                    expected_jobs=16,
                    expected_accepted=expected_accepted_jobs(
                        self.config, "target"
                    ),
                )
                if source["blocked"] or source["signature_mismatches"]:
                    raise RuntimeError(
                        "Source audit failed: "
                        + json.dumps(
                            {
                                "blocked": source["blocked"],
                                "signature_mismatches": source[
                                    "signature_mismatches"
                                ],
                            },
                            sort_keys=True,
                        )
                    )
                if not source["strict_complete"]:
                    supervisors = validate_active_lanes(
                        self.config, source, self.source_sessions
                    )
                    self.write_status(
                        "monitoring_source",
                        source=source,
                        target=target,
                        supervisors=supervisors,
                    )
                    self.sleep_fn(self.poll_interval_seconds)
                    continue
                source_finalized, source_supervisors = stage_is_finalized(
                    self.config,
                    stage="source",
                    sessions=self.source_sessions,
                )
                if not source_finalized:
                    self.write_status(
                        "source_finalizing",
                        source=source,
                        target=target,
                        supervisors=source_supervisors,
                    )
                    self.sleep_fn(self.poll_interval_seconds)
                    continue
                if not audit.get("ready_to_launch"):
                    raise RuntimeError(
                        "Full strict audit is not launchable after source completion"
                    )
                if not self.paths.source_audit.is_file():
                    _atomic_json(self.paths.source_audit, audit)
                    self.emit("source_audit_complete")
                if target["blocked"] or target["signature_mismatches"]:
                    raise RuntimeError(
                        "Target prelaunch audit failed: "
                        + json.dumps(
                            {
                                "blocked": target["blocked"],
                                "signature_mismatches": target[
                                    "signature_mismatches"
                                ],
                            },
                            sort_keys=True,
                        )
                    )
                if target["strict_complete"]:
                    finalized, supervisors = stage_is_finalized(
                        self.config,
                        stage="target",
                        sessions=self.target_sessions,
                    )
                    if not finalized:
                        self.write_status(
                            "target_finalizing",
                            source=source,
                            target=target,
                            supervisors=supervisors,
                        )
                        self.sleep_fn(self.poll_interval_seconds)
                        continue
                    _atomic_json(self.paths.final_audit, audit)
                    self.write_status(
                        "complete", source=source, target=target
                    )
                    self.emit("cache_matrix_complete")
                    return 0
                if not self.target_launches:
                    manager_logs = {
                        lane: self.paths.output_dir
                        / f"target_gpu{lane}.manager.log"
                        for lane in (0, 1)
                    }
                    self.target_launches = launch_target_lanes(
                        self.config,
                        sessions=self.target_sessions,
                        manager_logs=manager_logs,
                        python=Path(sys.executable).resolve(),
                    )
                    self.emit("target_lanes_launched")
                supervisors = validate_active_lanes(
                    self.config, target, self.target_sessions
                )
                self.write_status(
                    "monitoring_target",
                    source=source,
                    target=target,
                    supervisors=supervisors,
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
    return datetime.now(UTC).isoformat()


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
        f"- Target launches: `{json.dumps(payload.get('target_launches', []), sort_keys=True)}`",
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
    parser.add_argument(
        "--source-session-prefix", default="mprisk-cache-source-gpu"
    )
    parser.add_argument(
        "--target-session-prefix", default="mprisk-cache-target-gpu"
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_matrix_config(args.config)
    controller = StageController(
        config,
        paths=build_controller_paths(args.output_dir),
        poll_interval_seconds=args.poll_interval_seconds,
        source_sessions={
            lane: f"{args.source_session_prefix}{lane}" for lane in (0, 1)
        },
        target_sessions={
            lane: f"{args.target_session_prefix}{lane}" for lane in (0, 1)
        },
    )
    return controller.run()
