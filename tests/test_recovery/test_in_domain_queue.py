from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import mprisk.recovery.in_domain_queue as in_domain_queue
from mprisk.recovery.in_domain_queue import (
    _contracts_pass,
    environment_contract,
    load_queue,
    resume_contract,
    write_runtime_receipt,
)


def test_load_queue_rejects_dependency_that_does_not_precede_step(tmp_path: Path) -> None:
    config = {
        "schema_name": "mprisk_in_domain_recovery_queue_v1",
        "repository_root": str(tmp_path),
        "output_root": str(tmp_path / "out"),
        "gate": {
            "controller_status_path": str(tmp_path / "status.json"),
            "target_tmux_sessions": ["lane0", "lane1"],
            "forbidden_cuda_worker_patterns": ["worker.py"],
        },
        "steps": [
            {
                "id": "a",
                "requires": ["future"],
                "command": ["true"],
                "dry_run_command": ["true"],
                "completion": [
                    {"kind": "file", "path": str(tmp_path / "out" / "done")}
                ],
            }
        ],
    }
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="dependencies must precede"):
        load_queue(path)


def test_load_queue_accepts_ordered_fail_closed_contract(tmp_path: Path) -> None:
    output = tmp_path / "out"
    config = {
        "schema_name": "mprisk_in_domain_recovery_queue_v1",
        "repository_root": str(tmp_path),
        "output_root": str(output),
        "gate": {
            "controller_status_path": str(tmp_path / "status.json"),
            "target_tmux_sessions": ["lane0", "lane1"],
            "forbidden_cuda_worker_patterns": ["worker.py"],
        },
        "steps": [
            {
                "id": "a",
                "requires": [],
                "command": ["true"],
                "dry_run_command": ["true"],
                "completion": [
                    {
                        "kind": "jsonl_count",
                        "path": str(output / "rows.jsonl"),
                        "expected_rows": 2,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert load_queue(path)["steps"][0]["id"] == "a"


def test_jsonl_contract_can_use_relation_row_identity(tmp_path: Path) -> None:
    path = tmp_path / "relation.jsonl"
    path.write_text(
        '{"row_id":"sample-a:p1","sample_id":"sample-a"}\n'
        '{"row_id":"sample-a:p2","sample_id":"sample-a"}\n',
        encoding="utf-8",
    )
    assert _contracts_pass(
        [
            {
                "kind": "jsonl_count",
                "path": str(path),
                "expected_rows": 2,
                "identity_fields": ["row_id"],
            }
        ]
    )


def test_gate_accepts_complete_controller_status_with_idle_target_supervisors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "target": {"strict_complete": True},
                "supervisors": [
                    {
                        "stage": "target",
                        "lane": lane,
                        "active": False,
                        "lock_exists": False,
                        "lock_pid_alive": False,
                        "session_exists": False,
                    }
                    for lane in (0, 1)
                ],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "_resolved_status_path": status_path,
        "gate": {
            "target_tmux_sessions": ["target0", "target1"],
            "forbidden_cuda_worker_patterns": ["worker.py"],
        },
    }
    monkeypatch.setattr(in_domain_queue, "_live_tmux_sessions", lambda _: [])
    monkeypatch.setattr(in_domain_queue, "_matching_cuda_workers", lambda _: [])

    evidence = in_domain_queue.gate_evidence(config)

    assert evidence == {
        "ready": True,
        "target_strict_complete": True,
        "target_supervisors_idle": True,
        "live_target_tmux_sessions": [],
        "matching_cuda_workers": [],
        "controller_status_path": str(status_path),
    }


def test_load_queue_rejects_invalid_required_environment_name(tmp_path: Path) -> None:
    config = {
        "schema_name": "mprisk_in_domain_recovery_queue_v1",
        "repository_root": str(tmp_path),
        "output_root": str(tmp_path / "out"),
        "required_environment": [
            {"name": "DEEPSEEK_API_KEY", "source": "tmux_global"},
            {"name": "invalid-name", "source": "tmux_global"},
        ],
        "gate": {
            "controller_status_path": str(tmp_path / "status.json"),
            "target_tmux_sessions": ["lane0", "lane1"],
            "forbidden_cuda_worker_patterns": ["worker.py"],
        },
        "steps": [
            {
                "id": "a",
                "requires": [],
                "command": ["true"],
                "dry_run_command": ["true"],
                "completion": [{"kind": "file", "path": str(tmp_path / "out" / "done")}],
            }
        ],
    }
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="required_environment"):
        load_queue(path)


def test_environment_contract_never_records_secret_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "must-never-enter-receipt"
    monkeypatch.setenv("DEEPSEEK_API_KEY", sentinel)
    monkeypatch.setattr(
        in_domain_queue, "_read_tmux_global_environment", lambda _: sentinel
    )
    contract = environment_contract(
        {
            "_environment_bindings": [
                {"name": "DEEPSEEK_API_KEY", "source": "tmux_global"}
            ]
        }
    )
    assert contract == {
        "required": [
            {
                "name": "DEEPSEEK_API_KEY",
                "source": "tmux_global",
                "present": True,
                "value_format_valid": True,
                "source_present": True,
                "source_value_format_valid": True,
                "source_matches_process": True,
            }
        ],
        "missing_names": [],
        "secret_values_recorded": False,
    }
    assert sentinel not in json.dumps(contract)


def test_environment_contract_fails_closed_on_blank_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(
        in_domain_queue, "_read_tmux_global_environment", lambda _: "accepted"
    )
    contract = environment_contract(
        {
            "_environment_bindings": [
                {"name": "DEEPSEEK_API_KEY", "source": "tmux_global"}
            ]
        }
    )
    assert contract["required"][0]["present"] is False
    assert contract["required"][0]["source_matches_process"] is False
    assert contract["missing_names"] == ["DEEPSEEK_API_KEY"]


@pytest.mark.parametrize(
    ("process_value", "source_value"),
    [("accepted", "different"), (" accepted", " accepted"), ("accepted ", "accepted ")],
)
def test_environment_contract_rejects_source_mismatch_and_whitespace(
    process_value: str, source_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", process_value)
    monkeypatch.setattr(
        in_domain_queue,
        "_read_tmux_global_environment",
        lambda _: source_value,
    )
    contract = environment_contract(
        {
            "_environment_bindings": [
                {"name": "DEEPSEEK_API_KEY", "source": "tmux_global"}
            ]
        }
    )
    assert contract["required"][0]["source_matches_process"] is False
    assert contract["missing_names"] == ["DEEPSEEK_API_KEY"]


def test_runtime_receipt_binds_code_and_config_without_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "queue.yaml"
    config_path.write_text("schema_name: test\n", encoding="utf-8")
    sentinel = "must-never-enter-runtime-receipt"
    monkeypatch.setenv("DEEPSEEK_API_KEY", sentinel)
    monkeypatch.setattr(
        in_domain_queue, "_read_tmux_global_environment", lambda _: sentinel
    )
    monkeypatch.setattr(
        in_domain_queue.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="a" * 40 + "\n", stderr=""
        ),
    )
    config = {
        "_environment_bindings": [
            {"name": "DEEPSEEK_API_KEY", "source": "tmux_global"}
        ],
        "_resolved_repository_root": tmp_path,
        "_resolved_output_root": tmp_path / "out",
        "_config_path": config_path,
    }
    receipt_path = write_runtime_receipt(config, environment_contract(config), None)
    serialized = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(serialized)
    assert receipt["schema_name"] == "mprisk_in_domain_recovery_queue_runtime_v1"
    assert receipt["git_commit"] == "a" * 40
    assert receipt["environment_contract"]["required"][0] == {
        "name": "DEEPSEEK_API_KEY",
        "source": "tmux_global",
        "present": True,
        "value_format_valid": True,
        "source_present": True,
        "source_value_format_valid": True,
        "source_matches_process": True,
    }
    assert sentinel not in serialized


def _resume_config(tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "out"
    ledger = output / "judgments" / "request_ledger.sqlite3"
    ledger.parent.mkdir(parents=True)
    with sqlite3.connect(ledger) as database:
        database.execute(
            "create table calls (call_id text, sample_id text, status text, "
            "attempts integer, request_id text, result_json text, started_at text)"
        )
        database.executemany(
            "insert into calls values (?,?,?,?,?,?,?)",
            [
                ("c1", "s1", "pending", 0, None, None, None),
                ("c2", "s1", "pending", 0, None, None, None),
            ],
        )
        database.execute("create table final (sample_id text)")
    frozen = output / "frozen" / "SHA256SUMS"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("frozen evidence\n", encoding="utf-8")
    return {
        "_resolved_output_root": output,
        "resume_contract": {
            "ledger_path": str(ledger),
            "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            "frozen_sha256s_path": str(frozen),
            "frozen_sha256s_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "prestate": {
                "calls": 2,
                "distinct_call_ids": 2,
                "distinct_sample_ids": 1,
                "statuses": {"pending": 2},
                "attempts_total": 0,
                "attempts_max": 0,
                "final_rows": 0,
                "zero_nonnull_fields": ["request_id", "result_json", "started_at"],
            },
        },
    }


def test_resume_contract_binds_ledger_archive_and_exact_prestate(tmp_path: Path) -> None:
    contract = resume_contract(_resume_config(tmp_path))
    assert contract is not None
    assert contract["matches"] is True
    assert contract["expected_ledger_sha256"] == contract["observed_ledger_sha256"]
    assert (
        contract["expected_frozen_sha256s_sha256"]
        == contract["observed_frozen_sha256s_sha256"]
    )
    assert contract["observed_prestate"]["nonnull_fields"] == {
        "request_id": 0,
        "result_json": 0,
        "started_at": 0,
    }


def test_resume_contract_rejects_mutated_ledger_before_any_step(tmp_path: Path) -> None:
    config = _resume_config(tmp_path)
    ledger = Path(config["resume_contract"]["ledger_path"])
    with sqlite3.connect(ledger) as database:
        database.execute(
            "update calls set status='running', attempts=1, started_at='now' where call_id='c1'"
        )
    contract = resume_contract(config)
    assert contract is not None
    assert contract["matches"] is False
    assert contract["observed_ledger_sha256"] != contract["expected_ledger_sha256"]
    assert contract["observed_prestate"]["attempts_total"] == 1
    assert contract["observed_prestate"]["nonnull_fields"]["started_at"] == 1


@pytest.mark.parametrize(
    ("environment_missing", "resume_matches"),
    [(["DEEPSEEK_API_KEY"], True), ([], False)],
)
def test_execute_stops_before_queue_on_environment_or_resume_mismatch(
    environment_missing: list[str],
    resume_matches: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "steps": [],
        "_environment_bindings": [],
        "_resolved_output_root": tmp_path,
    }
    monkeypatch.setattr(in_domain_queue, "load_queue", lambda _: config)
    monkeypatch.setattr(in_domain_queue, "gate_evidence", lambda _: {"ready": True})
    monkeypatch.setattr(
        in_domain_queue,
        "environment_contract",
        lambda _: {
            "required": [],
            "missing_names": environment_missing,
            "secret_values_recorded": False,
        },
    )
    monkeypatch.setattr(
        in_domain_queue, "resume_contract", lambda _: {"matches": resume_matches}
    )
    monkeypatch.setattr(
        in_domain_queue,
        "write_runtime_receipt",
        lambda *_: tmp_path / "queue_runtime_receipt.json",
    )
    monkeypatch.setattr(
        in_domain_queue,
        "run_queue",
        lambda *args, **kwargs: pytest.fail("queue must not start"),
    )
    assert (
        in_domain_queue.main(
            ["--config", str(tmp_path / "queue.yaml"), "--execute"]
        )
        == 2
    )
