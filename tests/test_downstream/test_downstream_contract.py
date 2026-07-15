from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mprisk.experiments import downstream
from mprisk.experiments.downstream import (
    CacheJob,
    _retained_conflict_rows,
    official_test_rows,
    validate_completed_cache,
)
from mprisk.representation.training import _read_relation_rows


def _completed_cache(tmp_path: Path) -> CacheJob:
    prompt_path = tmp_path / "prompts.yaml"
    templates = [
        {"prompt_id": f"p{index}", "enabled": True, "template_text": "x"} for index in range(8)
    ]
    prompt_path.write_text(yaml.safe_dump({"key": "p8", "templates": templates}), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    ledger = sqlite3.connect(cache / "batch_state.sqlite3")
    ledger.execute(
        """CREATE TABLE tasks (
        status TEXT, model_key TEXT, protocol TEXT, prompt_set_key TEXT)"""
    )
    ledger.executemany(
        "INSERT INTO tasks VALUES (?,?,?,?)",
        [("completed", "model", "vt", "p8")] * 24,
    )
    ledger.commit()
    ledger.close()
    rows = []
    for prompt in templates:
        for condition in ("M1", "M2", "M12"):
            rows.append(
                {
                    "sample_id": "s1",
                    "model_key": "model",
                    "protocol": "vt",
                    "prompt_set_key": "p8",
                    "prompt_id": prompt["prompt_id"],
                    "condition": condition,
                    "checksum": "a" * 64,
                }
            )
    (cache / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return CacheJob(20260715, "model", "vt", tmp_path / "source", prompt_path, cache, 24)


def test_completed_cache_gate_rejects_seven_prompt_drift(tmp_path: Path) -> None:
    job = _completed_cache(tmp_path)
    report = validate_completed_cache(job, verify_artifacts=False)
    assert report["sample_count"] == 1
    rows = [
        json.loads(line) for line in (job.cache_root / "manifest.jsonl").read_text().splitlines()
    ]
    rows[-1]["prompt_id"] = "unregistered"
    (job.cache_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unregistered prompt"):
        validate_completed_cache(job, verify_artifacts=False)


def test_paper_rows_are_official_test_only_with_split_provenance() -> None:
    base = {
        "split_assignment_key": "registered",
        "split_assignment_sha256": "a" * 64,
    }
    rows = [
        {**base, "sample_id": "train", "representation_split": "relation_train"},
        {**base, "sample_id": "cal", "representation_split": "aligned_calibration"},
        {**base, "sample_id": "test", "representation_split": "official_test"},
    ]
    selected, provenance = official_test_rows(rows, source_name="fixture")
    assert [row["sample_id"] for row in selected] == ["test"]
    assert provenance["source_split_counts"] == {
        "aligned_calibration": 1,
        "official_test": 1,
        "relation_train": 1,
    }
    assert provenance["included_count"] == 1


def test_relation_reader_hard_fails_protocol_and_prompt_sha_drift(tmp_path: Path) -> None:
    row = {
        "schema": "mprisk_relation_sample_v1",
        "row_id": "s:p",
        "sample_id": "s",
        "sample_type": "Conflict",
        "label_id": 1,
        "model_key": "model",
        "protocol": "vt",
        "prompt_set_artifact_sha256": "a" * 64,
        "conditions": {"M1": {}, "M2": {}, "M12": {}},
    }
    path = tmp_path / "relation.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert _read_relation_rows(
        path,
        expected_model_key="model",
        expected_protocol="vt",
        expected_prompt_set_artifact_sha256="a" * 64,
    ) == [row]
    with pytest.raises(ValueError, match="protocol"):
        _read_relation_rows(
            path,
            expected_model_key="model",
            expected_protocol="va",
            expected_prompt_set_artifact_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="artifact SHA"):
        _read_relation_rows(
            path,
            expected_model_key="model",
            expected_protocol="vt",
            expected_prompt_set_artifact_sha256="b" * 64,
        )


def test_conflict_retention_changes_training_groups_only() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "row_id": f"train-{index}",
                "split_group_id": f"conflict-{index}",
                "representation_split": "relation_train",
                "sample_type": "Conflict",
            }
        )
    protected = [
        {
            "row_id": "aligned-train",
            "split_group_id": "aligned-train",
            "representation_split": "relation_train",
            "sample_type": "Aligned",
        },
        {
            "row_id": "calibration",
            "split_group_id": "calibration",
            "representation_split": "aligned_calibration",
            "sample_type": "Aligned",
        },
        {
            "row_id": "test",
            "split_group_id": "test",
            "representation_split": "official_test",
            "sample_type": "Conflict",
        },
    ]
    retained, metadata = _retained_conflict_rows(rows + protected, fraction=0.1, seed=7)
    assert metadata["retained_relation_train_conflict_groups"] == 1
    assert {row["row_id"] for row in protected} <= {row["row_id"] for row in retained}


def test_live_cache_producer_guard_prevents_gpu_check_start_race(monkeypatch) -> None:
    plan = SimpleNamespace(
        producer_tmux_sessions=("producer",),
        producer_command_substrings=("extract_prefill_batch.py",),
    )

    def fake_run(command, **_kwargs):
        assert command[:3] == ["tmux", "has-session", "-t"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(downstream.subprocess, "run", fake_run)
    assert downstream._cache_producer_can_launch_gpu_work(plan) is True
