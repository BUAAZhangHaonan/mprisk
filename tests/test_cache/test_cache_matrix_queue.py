from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from mprisk.cache.cache_matrix_queue import (
    CacheJob,
    DomainProtocol,
    ModelSpec,
    _ledger_status,
    _smoke_status,
    normalize_manifest,
)


def test_normalize_manifest_resolves_media_and_adds_batch_fields(tmp_path: Path) -> None:
    media_root = tmp_path / "bundle"
    media = media_root / "datasets" / "sample.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"video")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "protocol": "VT",
                "sample_type": "Conflict",
                "media_paths": {"vision": "datasets/sample.mp4"},
                "text_content": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    domain = DomainProtocol(
        domain="source",
        protocol="vt",
        source_manifest=source,
        prepared_manifest=tmp_path / "prepared.jsonl",
        media_root=media_root,
        source_dataset="generated_3810",
        split="all",
        expected_samples=1,
    )

    rows, digest = normalize_manifest(domain)

    assert rows[0]["media_paths"] == {"vision": str(media.resolve())}
    assert rows[0]["source_dataset"] == "generated_3810"
    assert rows[0]["split"] == "all"
    assert rows[0]["annotation_count"] == 0
    assert rows[0]["use_in_main"] is True
    assert len(digest) == 64


def test_ledger_status_reports_missing_only_and_refuses_failed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    connection = sqlite3.connect(root / "batch_state.sqlite3")
    connection.execute("CREATE TABLE tasks(status TEXT)")
    connection.executemany(
        "INSERT INTO tasks(status) VALUES(?)", [("completed",), ("pending",), ("pending",)]
    )
    connection.commit()
    connection.close()

    assert _ledger_status(root, 3) == {
        "status": "incomplete",
        "counts": {"completed": 1, "pending": 2},
        "completed": 1,
        "missing": 2,
    }

    connection = sqlite3.connect(root / "batch_state.sqlite3")
    connection.execute("UPDATE tasks SET status='failed' WHERE rowid=2")
    connection.commit()
    connection.close()

    status = _ledger_status(root, 3)
    assert status["status"] == "failed"
    assert status["missing"] == 2


def test_smoke_gate_requires_exact_48_task_contract(tmp_path: Path) -> None:
    prompt_set = tmp_path / "p8.yaml"
    prompt_set.write_text("p8\n", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    model = ModelSpec(
        model_key="model",
        family="family",
        protocol="vt",
        python=python,
        gpu_lane=0,
        trajectory_shape=(32, 2560),
        extra_args=(),
        invalidated_domains={},
        accepted_bundle_domains={},
    )
    domain = DomainProtocol(
        domain="source",
        protocol="vt",
        source_manifest=tmp_path / "source.jsonl",
        prepared_manifest=tmp_path / "prepared.jsonl",
        media_root=tmp_path,
        source_dataset="source",
        split="all",
        expected_samples=1,
    )
    smoke = tmp_path / "SMOKE_COMPLETE.json"
    job = CacheJob(domain, model, tmp_path / "out", smoke)
    config = SimpleNamespace(prompt_sets={"vt": prompt_set})
    payload = {
        "schema": "mprisk_cache_smoke_evidence_v1",
        "status": "PASS",
        "model_key": "model",
        "family": "family",
        "protocol": "vt",
        "domain": "source",
        "expected_tasks": 48,
        "completed_tasks": 48,
        "failed_tasks": 0,
        "prompt_set_sha256": hashlib.sha256(b"p8\n").hexdigest(),
        "environment_python": str(python),
        "trajectory_shape": [32, 2560],
        "prompt_ids": [f"p{i}" for i in range(8)],
    }
    smoke.write_text(json.dumps(payload), encoding="utf-8")

    assert _smoke_status(config, job)["passed"] is True
    payload["completed_tasks"] = 47
    smoke.write_text(json.dumps(payload), encoding="utf-8")
    result = _smoke_status(config, job)
    assert result["passed"] is False
    assert "completed_tasks" in result["mismatches"]
