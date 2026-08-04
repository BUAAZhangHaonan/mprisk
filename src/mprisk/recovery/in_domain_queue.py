"""Fail-closed, resumable orchestration for in-domain cache recovery."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "mprisk_in_domain_recovery_queue_v1"
STATE_SCHEMA = "mprisk_in_domain_recovery_state_v1"
RUNTIME_SCHEMA = "mprisk_in_domain_recovery_queue_runtime_v1"
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
    required_environment = value.get("required_environment", [])
    if not isinstance(required_environment, list):
        raise ValueError("required_environment must be a list")
    environment_bindings: list[dict[str, str]] = []
    for binding in required_environment:
        if not isinstance(binding, dict):
            raise ValueError("required_environment entries must be mappings")
        name = binding.get("name")
        source = binding.get("source")
        if not isinstance(name, str) or not ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError("required_environment contains an invalid name")
        if source != "tmux_global":
            raise ValueError("required_environment contains an invalid source")
        environment_bindings.append({"name": name, "source": source})
    names = [binding["name"] for binding in environment_bindings]
    if len(names) != len(set(names)):
        raise ValueError("required_environment names must be unique")
    resume = value.get("resume_contract")
    if resume is not None:
        _validate_resume_contract(resume, output_root)
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
    value["_config_path"] = path.expanduser().resolve()
    value["_environment_bindings"] = environment_bindings
    return value


def environment_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return a non-secret receipt for required inherited environment variables."""
    bindings = list(config.get("_environment_bindings", []))
    entries: list[dict[str, Any]] = []
    for binding in bindings:
        name = binding["name"]
        process_value = os.environ.get(name)
        source_value = _read_tmux_global_environment(name)
        process_valid = bool(process_value) and process_value == process_value.strip()
        source_valid = bool(source_value) and source_value == source_value.strip()
        matches = bool(
            process_valid
            and source_valid
            and hmac.compare_digest(process_value, source_value)
        )
        entries.append(
            {
                "name": name,
                "source": binding["source"],
                "present": bool(process_value),
                "value_format_valid": process_valid,
                "source_present": bool(source_value),
                "source_value_format_valid": source_valid,
                "source_matches_process": matches,
            }
        )
    missing = [entry["name"] for entry in entries if not entry["source_matches_process"]]
    return {
        "required": entries,
        "missing_names": missing,
        "secret_values_recorded": False,
    }


def write_runtime_receipt(
    config: dict[str, Any], contract: dict[str, Any], resume: dict[str, Any] | None
) -> Path:
    """Bind an execution to code/config while recording no environment values."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config["_resolved_repository_root"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to bind recovery runtime: {completed.stderr.strip()}")
    config_path = config["_config_path"]
    receipt = {
        "schema_name": RUNTIME_SCHEMA,
        "repository_root": str(config["_resolved_repository_root"]),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_commit": completed.stdout.strip(),
        "python_executable": sys.executable,
        "environment_contract": contract,
        "resume_contract": resume,
    }
    path = config["_resolved_output_root"] / "queue_runtime_receipt.json"
    _write_json_atomic(path, receipt)
    return path


def resume_contract(config: dict[str, Any]) -> dict[str, Any] | None:
    """Audit an immutable paid-stage resume point before starting any queue step."""
    expected = config.get("resume_contract")
    if expected is None:
        return None
    ledger_path = Path(expected["ledger_path"]).expanduser().resolve()
    sums_path = Path(expected["frozen_sha256s_path"]).expanduser().resolve()
    observed_ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    observed_sums_sha = hashlib.sha256(sums_path.read_bytes()).hexdigest()
    with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as database:
        columns = {row[1] for row in database.execute("pragma table_info(calls)")}
        zero_fields = list(expected["prestate"]["zero_nonnull_fields"])
        missing_columns = sorted(set(zero_fields) - columns)
        nonnull = {
            field: database.execute(
                f"select count(*) from calls where {field} is not null"
            ).fetchone()[0]
            for field in zero_fields
            if field in columns
        }
        attempts_total, attempts_max = database.execute(
            "select coalesce(sum(attempts),0),coalesce(max(attempts),0) from calls"
        ).fetchone()
        observed_prestate = {
            "calls": database.execute("select count(*) from calls").fetchone()[0],
            "distinct_call_ids": database.execute(
                "select count(distinct call_id) from calls"
            ).fetchone()[0],
            "distinct_sample_ids": database.execute(
                "select count(distinct sample_id) from calls"
            ).fetchone()[0],
            "statuses": dict(
                database.execute("select status,count(*) from calls group by status")
            ),
            "attempts_total": attempts_total,
            "attempts_max": attempts_max,
            "final_rows": database.execute("select count(*) from final").fetchone()[0],
            "nonnull_fields": nonnull,
            "missing_columns": missing_columns,
        }
    expected_prestate = expected["prestate"]
    prestate_matches = (
        observed_prestate["calls"] == expected_prestate["calls"]
        and observed_prestate["distinct_call_ids"]
        == expected_prestate["distinct_call_ids"]
        and observed_prestate["distinct_sample_ids"]
        == expected_prestate["distinct_sample_ids"]
        and observed_prestate["statuses"] == expected_prestate["statuses"]
        and observed_prestate["attempts_total"] == expected_prestate["attempts_total"]
        and observed_prestate["attempts_max"] == expected_prestate["attempts_max"]
        and observed_prestate["final_rows"] == expected_prestate["final_rows"]
        and not missing_columns
        and all(value == 0 for value in nonnull.values())
    )
    matches = bool(
        observed_ledger_sha == expected["ledger_sha256"]
        and observed_sums_sha == expected["frozen_sha256s_sha256"]
        and prestate_matches
    )
    return {
        "ledger_path": str(ledger_path),
        "expected_ledger_sha256": expected["ledger_sha256"],
        "observed_ledger_sha256": observed_ledger_sha,
        "frozen_sha256s_path": str(sums_path),
        "expected_frozen_sha256s_sha256": expected["frozen_sha256s_sha256"],
        "observed_frozen_sha256s_sha256": observed_sums_sha,
        "expected_prestate": expected_prestate,
        "observed_prestate": observed_prestate,
        "matches": matches,
    }


def _read_tmux_global_environment(name: str) -> str | None:
    completed = subprocess.run(
        ["tmux", "show-environment", "-g", name],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    prefix = f"{name}="
    output = completed.stdout.removesuffix("\n")
    if not output.startswith(prefix):
        return None
    return output[len(prefix) :]


def _validate_resume_contract(value: Any, output_root: Path) -> None:
    if not isinstance(value, dict):
        raise ValueError("resume_contract must be a mapping")
    for field in ("ledger_path", "frozen_sha256s_path"):
        path = Path(_text(value, field)).expanduser().resolve()
        if output_root != path and output_root not in path.parents:
            raise ValueError(f"resume_contract {field} escapes output root")
    for field in ("ledger_sha256", "frozen_sha256s_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"resume_contract has invalid {field}")
    prestate = value.get("prestate")
    if not isinstance(prestate, dict):
        raise ValueError("resume_contract prestate must be a mapping")
    for field in (
        "calls",
        "distinct_call_ids",
        "distinct_sample_ids",
        "attempts_total",
        "attempts_max",
        "final_rows",
    ):
        if not isinstance(prestate.get(field), int) or prestate[field] < 0:
            raise ValueError(f"resume_contract prestate has invalid {field}")
    statuses = prestate.get("statuses")
    if not isinstance(statuses, dict) or not statuses or not all(
        isinstance(key, str) and key and isinstance(count, int) and count >= 0
        for key, count in statuses.items()
    ):
        raise ValueError("resume_contract prestate has invalid statuses")
    zero_fields = prestate.get("zero_nonnull_fields")
    if not isinstance(zero_fields, list) or not zero_fields or not all(
        isinstance(field, str) and ENVIRONMENT_NAME.fullmatch(field)
        for field in zero_fields
    ):
        raise ValueError("resume_contract prestate has invalid zero_nonnull_fields")


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
    _write_json_atomic(path, payload)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "required_environment": list(config.get("_environment_bindings", [])),
    }
    if not args.execute:
        summary["stage_preflights"] = dry_run_commands(config)
    else:
        contract = environment_contract(config)
        resume = resume_contract(config)
        summary["environment_contract"] = contract
        summary["resume_contract"] = resume
        summary["runtime_receipt_path"] = str(
            write_runtime_receipt(config, contract, resume)
        )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    if not args.execute:
        return 0
    if contract["missing_names"] or (resume is not None and not resume["matches"]):
        return 2
    return run_queue(config, poll_interval_seconds=args.poll_interval_seconds)
