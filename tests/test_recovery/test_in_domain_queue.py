from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mprisk.recovery.in_domain_queue import load_queue


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
