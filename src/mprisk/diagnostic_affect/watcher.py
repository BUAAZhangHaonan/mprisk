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
from collections.abc import Callable, Mapping, Sequence
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
    python_environment: Mapping[str, str] | None = None,
    runtime_contract: Mapping[str, Any] | None = None,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    run_factory: Callable[..., Any] = subprocess.run,
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

    child_environment = _child_environment(python_environment)
    if runtime_contract is not None:
        runtime_evidence = _verify_runtime_contract(
            python_executable,
            environment=child_environment,
            contract=runtime_contract,
            run_factory=run_factory,
        )
        _write_json(
            output_root / "runtime_contract_receipt.json",
            {
                "schema_name": "mprisk_description_runtime_contract_receipt_v1",
                "status": "PASS",
                "expected": dict(runtime_contract),
                "observed": runtime_evidence,
            },
        )

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
        process = popen_factory(command, cwd=repository_root, env=child_environment)
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
    awaiting_retry_reset = bool(
        retry_failed and before is not None and before["failed"] > 0
    )
    _write_status(
        status_path,
        state="running",
        child_pid=int(process.pid),
        child_returncode=None,
        ledger=before,
        detail=(
            "full immutable plan launched; awaiting atomic failed-task retry reset"
            if awaiting_retry_reset
            else "full immutable plan launched; ledger resumes missing tasks only"
        ),
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
        if awaiting_retry_reset and summary != before:
            if summary is None or summary["failed"] > 0:
                _stop_child(process, terminate_grace_seconds)
                _write_status(
                    status_path,
                    state="failed",
                    child_pid=int(process.pid),
                    child_returncode=process.poll(),
                    ledger=summary,
                    detail="invalid failed-task retry reset transition",
                )
                return 1
            awaiting_retry_reset = False
            _write_status(
                status_path,
                state="running",
                child_pid=int(process.pid),
                child_returncode=None,
                ledger=summary,
                detail="atomic failed-task retry reset observed; strict supervision resumed",
            )
        if (
            not awaiting_retry_reset
            and summary is not None
            and summary["failed"] > 0
        ):
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


def _child_environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
    environment = os.environ.copy()
    if overrides is None:
        return environment
    for key, value in overrides.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Description child environment keys must be non-empty text")
        if not isinstance(value, str):
            raise ValueError("Description child environment values must be text")
        environment[key] = value
    return environment


def _verify_runtime_contract(
    python_executable: Path,
    *,
    environment: Mapping[str, str],
    contract: Mapping[str, Any],
    run_factory: Callable[..., Any],
) -> dict[str, Any]:
    expected_keys = {
        "python_executable",
        "python_prefix",
        "python_version",
        "user_site_enabled",
        "torch_cuda_version",
        "package_versions",
    }
    if set(contract) != expected_keys:
        raise ValueError("Description runtime contract fields are not exact")
    text_fields = (
        "python_executable",
        "python_prefix",
        "python_version",
        "torch_cuda_version",
    )
    if any(
        not isinstance(contract[field], str) or not contract[field]
        for field in text_fields
    ):
        raise ValueError("Description runtime scalar fields must be non-empty text")
    if not isinstance(contract["user_site_enabled"], bool):
        raise ValueError("Description runtime user_site_enabled must be boolean")
    package_versions = contract["package_versions"]
    if not isinstance(package_versions, Mapping) or not package_versions:
        raise ValueError("Description runtime package_versions must be a non-empty mapping")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in package_versions.items()
    ):
        raise ValueError("Description runtime package versions must be non-empty text")
    probe = (
        "import importlib.metadata,json,site,sys,torch;"
        "names=json.loads(sys.argv[1]);"
        "print(json.dumps({'python_executable':sys.executable,"
        "'python_prefix':sys.prefix,'python_version':sys.version.split()[0],"
        "'user_site_enabled':site.ENABLE_USER_SITE,"
        "'torch_cuda_version':torch.version.cuda,"
        "'package_versions':{name:importlib.metadata.version(name) for name in names}},"
        "sort_keys=True))"
    )
    result = run_factory(
        [
            str(python_executable),
            "-c",
            probe,
            json.dumps(sorted(package_versions)),
        ],
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Description runtime probe failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Description runtime probe returned invalid JSON") from exc
    expected = {
        "python_executable": contract["python_executable"],
        "python_prefix": contract["python_prefix"],
        "python_version": contract["python_version"],
        "user_site_enabled": contract["user_site_enabled"],
        "torch_cuda_version": contract["torch_cuda_version"],
        "package_versions": dict(package_versions),
    }
    if observed != expected:
        raise RuntimeError(
            f"Description runtime contract mismatch: expected={expected}, observed={observed}"
        )
    return observed


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
    _write_json(path, payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
