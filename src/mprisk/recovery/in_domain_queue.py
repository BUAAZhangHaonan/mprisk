"""Fail-closed, resumable orchestration for in-domain cache recovery."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "mprisk_in_domain_recovery_queue_v1"
STATE_SCHEMA = "mprisk_in_domain_recovery_state_v1"


def load_queue(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_name") != SCHEMA:
        raise ValueError(f"Unsupported recovery queue schema: {path}")
    root = Path(_text(value, "repository_root")).expanduser().resolve()
    output_root = Path(_text(value, "output_root")).expanduser().resolve()
    gate = value.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("Recovery queue gate must be a mapping")
    status_path = Path(_text(gate, "controller_status_path")).expanduser().resolve()
    sessions = gate.get("target_tmux_sessions")
    if not isinstance(sessions, list) or len(sessions) != 2 or not all(
        isinstance(item, str) and item for item in sessions
    ):
        raise ValueError("Exactly two target tmux sessions are required")
    workers = gate.get("forbidden_cuda_worker_patterns")
    if not isinstance(workers, list) or not workers or not all(
        isinstance(item, str) and item for item in workers
    ):
        raise ValueError("forbidden_cuda_worker_patterns must be non-empty")
    steps = value.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Recovery queue must contain steps")
    seen: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} must be a mapping")
        step_id = _text(step, "id")
        if step_id in seen:
            raise ValueError(f"Duplicate recovery step: {step_id}")
        dependencies = step.get("requires", [])
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) and item in seen for item in dependencies
        ):
            raise ValueError(f"Step dependencies must precede {step_id}")
        command = step.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError(f"Step command must be a non-empty argv list: {step_id}")
        dry_run_command = step.get("dry_run_command")
        if not isinstance(dry_run_command, list) or not dry_run_command or not all(
            isinstance(item, str) and item for item in dry_run_command
        ):
            raise ValueError(
                f"Step dry_run_command must be a non-empty argv list: {step_id}"
            )
        completion = step.get("completion")
        if not isinstance(completion, list) or not completion:
            raise ValueError(f"Step completion contract is missing: {step_id}")
        for contract in completion:
            _validate_contract(contract, output_root)
        seen.add(step_id)
    value["_resolved_repository_root"] = root
    value["_resolved_output_root"] = output_root
    value["_resolved_status_path"] = status_path
    return value


def dry_run_commands(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse and preflight every stage command without starting GPU/API work."""
    results: list[dict[str, Any]] = []
    for step in config["steps"]:
        command = step["dry_run_command"]
        completed = subprocess.run(
            command,
            cwd=config["_resolved_repository_root"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Recovery dry-run failed for {step['id']}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            payload = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Recovery dry-run did not return JSON for {step['id']}"
            ) from error
        if not isinstance(payload, dict) or payload.get("would_start_gpu") is not False:
            raise ValueError(f"Recovery dry-run GPU contract failed: {step['id']}")
        if payload.get("would_issue_api_requests") is not False:
            raise ValueError(f"Recovery dry-run API contract failed: {step['id']}")
        results.append({"id": step["id"], **payload})
    return results


def gate_evidence(config: dict[str, Any]) -> dict[str, Any]:
    path = config["_resolved_status_path"]
    status = json.loads(path.read_text(encoding="utf-8"))
    target = status.get("target")
    target_complete = isinstance(target, dict) and target.get("strict_complete") is True
    target_supervisors = [
        row
        for row in status.get("supervisors", [])
        if isinstance(row, dict) and row.get("stage") == "target"
    ]
    lanes_idle = (
        len(target_supervisors) == 2
        and {row.get("lane") for row in target_supervisors} == {0, 1}
        and all(
            row.get("active") is False
            and row.get("lock_exists") is False
            and row.get("lock_pid_alive") is False
            and row.get("session_exists") is False
            for row in target_supervisors
        )
    )
    tmux_sessions = config["gate"]["target_tmux_sessions"]
    live_sessions = _live_tmux_sessions(tmux_sessions)
    cuda_workers = _matching_cuda_workers(
        config["gate"]["forbidden_cuda_worker_patterns"]
    )
    ready = target_complete and lanes_idle and not live_sessions and not cuda_workers
    return {
        "ready": ready,
        "target_strict_complete": target_complete,
        "target_supervisors_idle": lanes_idle,
        "live_target_tmux_sessions": live_sessions,
        "matching_cuda_workers": cuda_workers,
        "controller_status_path": str(path),
    }


def run_queue(config: dict[str, Any], *, poll_interval_seconds: float) -> int:
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    while True:
        evidence = gate_evidence(config)
        if evidence["ready"]:
            break
        _write_state(config, state="waiting_for_cross_domain", gate=evidence, steps={})
        time.sleep(poll_interval_seconds)
    completed: dict[str, str] = {}
    for step in config["steps"]:
        step_id = step["id"]
        if _contracts_pass(step["completion"]):
            completed[step_id] = "already_complete"
            continue
        log_path = config["_resolved_output_root"] / "logs" / f"{step_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _write_state(config, state="running", gate=evidence, steps=completed | {step_id: "running"})
        with log_path.open("ab", buffering=0) as log:
            result = subprocess.run(
                step["command"],
                cwd=config["_resolved_repository_root"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0 or not _contracts_pass(step["completion"]):
            _write_state(
                config,
                state="failed",
                gate=evidence,
                steps=completed | {step_id: f"failed:{result.returncode}"},
            )
            return 1
        completed[step_id] = "completed"
    _write_state(config, state="completed", gate=evidence, steps=completed)
    return 0


def _contracts_pass(contracts: list[dict[str, Any]]) -> bool:
    try:
        for contract in contracts:
            path = Path(contract["path"]).expanduser().resolve()
            if not path.is_file():
                return False
            kind = contract["kind"]
            if kind == "file":
                continue
            if kind == "jsonl_count":
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                if len(rows) != int(contract["expected_rows"]):
                    return False
                identity_fields = contract.get("identity_fields", ["sample_id"])
                identities = [
                    tuple(row[field] for field in identity_fields) for row in rows
                ]
                if len(identities) != len(set(identities)):
                    return False
            elif kind == "json_field":
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get(contract["field"]) != contract["equals"]:
                    return False
            else:
                raise ValueError(f"Unknown completion contract: {kind}")
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def _validate_contract(contract: Any, output_root: Path) -> None:
    if not isinstance(contract, dict):
        raise ValueError("Completion contract must be a mapping")
    kind = _text(contract, "kind")
    if kind not in {"file", "jsonl_count", "json_field"}:
        raise ValueError(f"Unsupported completion contract: {kind}")
    path = Path(_text(contract, "path")).expanduser().resolve()
    if output_root != path and output_root not in path.parents:
        raise ValueError(f"Completion artifact escapes recovery output root: {path}")
    if kind == "jsonl_count" and (
        not isinstance(contract.get("expected_rows"), int)
        or contract["expected_rows"] <= 0
    ):
        raise ValueError("jsonl_count requires positive expected_rows")
    if kind == "jsonl_count":
        identity_fields = contract.get("identity_fields", ["sample_id"])
        if (
            not isinstance(identity_fields, list)
            or not identity_fields
            or not all(isinstance(field, str) and field for field in identity_fields)
        ):
            raise ValueError("jsonl_count identity_fields must be non-empty strings")
    if kind == "json_field" and (
        not isinstance(contract.get("field"), str) or "equals" not in contract
    ):
        raise ValueError("json_field requires field and equals")


def _live_tmux_sessions(names: list[str]) -> list[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"tmux inspection failed: {result.stderr.strip()}")
    observed = set(result.stdout.splitlines())
    return sorted(set(names) & observed)


def _matching_cuda_workers(patterns: list[str]) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi inspection failed: {result.stderr.strip()}")
    matches: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        pid_text, _, process_name = line.partition(",")
        try:
            pid = int(pid_text.strip())
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="strict"
            )
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if any(pattern in argv for pattern in patterns):
            matches.append(
                {"pid": pid, "process_name": process_name.strip(), "argv": argv.strip()}
            )
    return matches


def _write_state(
    config: dict[str, Any],
    *,
    state: str,
    gate: dict[str, Any],
    steps: dict[str, str],
) -> None:
    path = config["_resolved_output_root"] / "queue_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": STATE_SCHEMA,
        "state": state,
        "gate": gate,
        "steps": steps,
        "updated_unix_seconds": time.time(),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def _text(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be non-empty text")
    return item


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run gated in-domain recovery queue.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    config = load_queue(args.config)
    evidence = gate_evidence(config)
    summary = {
        "mode": "execute" if args.execute else "dry_run",
        "gate": evidence,
        "steps": [step["id"] for step in config["steps"]],
    }
    if not args.execute:
        summary["stage_preflights"] = dry_run_commands(config)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    if not args.execute:
        return 0
    return run_queue(config, poll_interval_seconds=args.poll_interval_seconds)
