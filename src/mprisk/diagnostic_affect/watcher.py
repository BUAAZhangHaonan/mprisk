"""Fail-closed supervision for resumable diagnostic-description generation."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

WATCHER_STATUS_SCHEMA = "mprisk_diagnostic_description_watcher_status_v1"


def read_ledger_summary(path: Path) -> dict[str, int] | None:
    """Read task counts without creating or mutating a missing ledger."""
    if not path.is_file():
        return None
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5.0) as connection:
        rows = connection.execute(
            "SELECT status,COUNT(*) FROM tasks GROUP BY status"
        ).fetchall()
    counts = {str(status): int(count) for status, count in rows}
    return {
        "total": sum(counts.values()),
        **{
            status: counts.get(status, 0)
            for status in ("pending", "running", "completed", "failed")
        },
    }


def watch_description_generation(
    *,
    config_path: Path,
    python_executable: Path,
    stall_timeout_seconds: float,
    poll_interval_seconds: float,
    terminate_grace_seconds: float,
    retry_failed: bool = False,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Launch one full-plan child and supervise progress from its durable ledger."""
    for name, value in (
        ("stall_timeout_seconds", stall_timeout_seconds),
        ("poll_interval_seconds", poll_interval_seconds),
        ("terminate_grace_seconds", terminate_grace_seconds),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    config = _read_config(config_path)
    output_root = Path(config["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "batch_state.sqlite3"
    status_path = output_root / "watcher_status.json"
    before = read_ledger_summary(ledger_path)
    if before is not None and _strictly_complete(before):
        _write_status(
            status_path,
            state="completed",
            child_pid=None,
            child_returncode=0,
            ledger=before,
            detail="ledger was already strictly complete; no child launched",
        )
        return 0

    repository_root = Path(__file__).resolve().parents[3]
    command = [
        str(python_executable),
        str(repository_root / "scripts" / "generate_diagnostic_affect_descriptions.py"),
        "--config",
        str(config_path.expanduser().resolve()),
    ]
    if retry_failed:
        command.append("--retry-failed")
    try:
        process = popen_factory(command, cwd=repository_root)
    except OSError as error:
        _write_status(
            status_path,
            state="failed",
            child_pid=None,
            child_returncode=None,
            ledger=before,
            detail=f"child launch failed: {type(error).__name__}: {error}",
        )
        return 1
    last_summary = before
    last_activity = monotonic_fn()
    _write_status(
        status_path,
        state="running",
        child_pid=int(process.pid),
        child_returncode=None,
        ledger=before,
        detail="full immutable plan launched; ledger resumes missing tasks only",
    )
    while True:
        summary = read_ledger_summary(ledger_path)
        now = monotonic_fn()
        if summary != last_summary:
            last_summary = summary
            last_activity = now
            _write_status(
                status_path,
                state="running",
                child_pid=int(process.pid),
                child_returncode=None,
                ledger=summary,
                detail="ledger heartbeat advanced",
            )
        if summary is not None and summary["failed"] > 0:
            _stop_child(process, terminate_grace_seconds)
            _write_status(
                status_path,
                state="failed",
                child_pid=int(process.pid),
                child_returncode=process.poll(),
                ledger=summary,
                detail="ledger contains failed tasks",
            )
            return 1
        returncode = process.poll()
        if returncode is not None:
            if returncode == 0 and summary is not None and _strictly_complete(summary):
                _write_status(
                    status_path,
                    state="completed",
                    child_pid=int(process.pid),
                    child_returncode=0,
                    ledger=summary,
                    detail="child exited zero and ledger is strictly complete",
                )
                return 0
            _write_status(
                status_path,
                state="failed",
                child_pid=int(process.pid),
                child_returncode=int(returncode),
                ledger=summary,
                detail="child exited before a strictly complete ledger",
            )
            return 1
        if now - last_activity >= stall_timeout_seconds:
            _stop_child(process, terminate_grace_seconds)
            _write_status(
                status_path,
                state="timed_out",
                child_pid=int(process.pid),
                child_returncode=process.poll(),
                ledger=summary,
                detail=(
                    f"no ledger heartbeat for {stall_timeout_seconds:.1f} seconds "
                    "while child remained alive"
                ),
            )
            return 1
        sleep_fn(poll_interval_seconds)


def _strictly_complete(summary: dict[str, int]) -> bool:
    return (
        summary["total"] > 0
        and summary["completed"] == summary["total"]
        and summary["pending"] == 0
        and summary["running"] == 0
        and summary["failed"] == 0
    )


def _stop_child(process: Any, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Diagnostic description config must be a mapping")
    output_root = value.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("Diagnostic description config has no output_root")
    return value


def _write_status(
    path: Path,
    *,
    state: str,
    child_pid: int | None,
    child_returncode: int | None,
    ledger: dict[str, int] | None,
    detail: str,
) -> None:
    payload = {
        "schema_name": WATCHER_STATUS_SCHEMA,
        "state": state,
        "child_pid": child_pid,
        "child_returncode": child_returncode,
        "ledger": ledger,
        "detail": detail,
        "updated_unix_seconds": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise resumable diagnostic-description generation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stall-timeout-seconds", type=float, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=30.0)
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return watch_description_generation(
        config_path=args.config,
        python_executable=args.python,
        stall_timeout_seconds=args.stall_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        terminate_grace_seconds=args.terminate_grace_seconds,
        retry_failed=args.retry_failed,
    )
