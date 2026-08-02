from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import mprisk.recovery.in_domain_queue as in_domain_queue
from mprisk.recovery.in_domain_queue import _contracts_pass, load_queue


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
