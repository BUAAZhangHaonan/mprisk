from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from mprisk.diagnostic_affect.matrix import (
    _audit_target_gt_coverage,
    _gt_row,
    _planned_request_records,
    _write_blocked_gt_plan,
)


def test_matrix_request_plan_has_unique_global_call_ids() -> None:
    records = _planned_request_records(
        run_id="run",
        job_id="target_model",
        model_key="model",
        protocol="VT",
        sample_ids=["a", "b"],
        flash_model="deepseek-v4-flash",
        pro_model="deepseek-v4-pro",
    )

    assert len(records) == 8
    assert len({row["call_id"] for row in records}) == 8
    assert Counter(row["role"] for row in records) == {"flash": 6, "pro": 2}
    assert all(row["api_request_issued"] is False for row in records)
    assert all(row["request_sha256"] is None for row in records)
    assert all(row["conditional"] is (row["role"] == "pro") for row in records)
    assert all(
        row["request_materialization_status"] == "awaiting_diagnostic_affect_description"
        for row in records
    )


def test_target_gt_audit_blocks_manifests_without_gt_description(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    vt = bundle / "vt.jsonl"
    va = bundle / "va.jsonl"
    bundle.mkdir()
    vt.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "protocol": "VT"}) + "\n"
            for sample_id in ("vt1", "vt2")
        ),
        encoding="utf-8",
    )
    va.write_text(
        json.dumps({"sample_id": "va1", "protocol": "VA"}) + "\n",
        encoding="utf-8",
    )
    jobs = [
        {
            "job_id": f"target_{protocol.lower()}",
            "model_key": f"model_{protocol.lower()}",
            "domain": "target",
            "protocol": protocol,
            "manifest_path": path.name,
            "expected_count": expected,
            "dataset": "target",
            "split": "train",
        }
        for protocol, path, expected in (("VT", vt, 2), ("VA", va, 1))
    ]

    audit = _audit_target_gt_coverage(
        jobs=jobs,
        bundle_root=bundle,
        canonical_domains={
            "target:vt": {"source_manifest": str(vt), "expected_samples": 2},
            "target:va": {"source_manifest": str(va), "expected_samples": 1},
        },
    )

    assert audit["status"] == "BLOCKED"
    assert audit["nonempty_gt_descriptions"] == 0
    assert audit["missing_gt_descriptions"] == 3
    assert audit["protocols"]["VT"]["duplicate_sample_ids"] == 0


def test_target_gt_row_never_synthesizes_valence() -> None:
    with pytest.raises(ValueError, match="Missing non-empty GT_DESCRIPTION"):
        _gt_row(
            {"sample_id": "a", "views": {"M12": {"label": "positive"}}},
            domain="target",
            dataset="target",
            split="train",
        )


def test_blocked_gt_plan_invalidates_pre_gate_request_artifacts(tmp_path: Path) -> None:
    destination = tmp_path / "plan"
    destination.mkdir()
    (destination / "request_plan_ledger.jsonl").write_text("old\n", encoding="utf-8")
    (destination / "jobs.jsonl").write_text("old\n", encoding="utf-8")
    (destination / "jobs").mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("run: blocked\n", encoding="utf-8")
    coverage_path = destination / "target_gt_coverage_audit.json"
    coverage_path.write_text("{}\n", encoding="utf-8")
    coverage = {
        "nonempty_gt_descriptions": 0,
        "missing_gt_descriptions": 3,
    }

    _write_blocked_gt_plan(
        destination=destination,
        config_path=config,
        run_id="run",
        coverage=coverage,
        coverage_path=coverage_path,
    )

    plan = json.loads((destination / "plan.json").read_text(encoding="utf-8"))
    assert plan["status"] == "blocked_gt_coverage"
    assert plan["api_requests_issued"] == 0
    assert plan["request_plan_ledger_path"] is None
    assert not (destination / "request_plan_ledger.jsonl").exists()
    assert (destination / "request_plan_ledger.jsonl.pre_gt_gate.INVALID").exists()
    assert (destination / "jobs.jsonl.pre_gt_gate.INVALID").exists()
    assert (destination / "jobs.pre_gt_gate.INVALID").is_dir()
