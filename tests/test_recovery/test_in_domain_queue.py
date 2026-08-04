from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import mprisk.recovery.in_domain_queue as in_domain_queue
from mprisk.recovery.in_domain_queue import (
    _contracts_pass,
    environment_contract,
    load_queue,
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
    contract = environment_contract(
        {
            "_environment_bindings": [
                {"name": "DEEPSEEK_API_KEY", "source": "tmux_global"}
            ]
        }
    )
    assert contract["required"][0]["present"] is False
    assert contract["missing_names"] == ["DEEPSEEK_API_KEY"]


def test_runtime_receipt_binds_code_and_config_without_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "queue.yaml"
    config_path.write_text("schema_name: test\n", encoding="utf-8")
    sentinel = "must-never-enter-runtime-receipt"
    monkeypatch.setenv("DEEPSEEK_API_KEY", sentinel)
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
    receipt_path = write_runtime_receipt(config, environment_contract(config))
    serialized = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(serialized)
    assert receipt["schema_name"] == "mprisk_in_domain_recovery_queue_runtime_v1"
    assert receipt["git_commit"] == "a" * 40
    assert receipt["environment_contract"]["required"][0] == {
        "name": "DEEPSEEK_API_KEY",
        "source": "tmux_global",
        "present": True,
    }
    assert sentinel not in serialized
