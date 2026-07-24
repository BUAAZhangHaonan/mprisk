from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/packaging/build_taffc_complete_bundle.py"
CONFIG = ROOT / "configs/packaging/complete_bundle_matrix.yaml"
SPEC = importlib.util.spec_from_file_location("taffc_complete_bundle_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["artifact_root"] = str(tmp_path)
    for dataset in payload["datasets"]:
        domain = dataset["domain"]
        protocol = dataset["protocol"]
        expected = dataset["expected_samples"]
        manifest = tmp_path / f"{domain}_{protocol.lower()}.jsonl"
        provenance = tmp_path / f"{domain}_{protocol.lower()}.provenance.json"
        _write_jsonl(
            manifest,
            [
                {
                    "sample_id": f"{domain}:{protocol}:{index:05d}",
                    "protocol": protocol,
                    "sample_type": "Conflict" if index % 2 else "Aligned",
                }
                for index in range(expected)
            ],
        )
        provenance.write_text(
            json.dumps({"domain": domain, "protocol": protocol}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dataset["manifest"] = {
            "status": "ready",
            "path": manifest.name,
            "sha256": _sha(manifest),
            "provenance_path": provenance.name,
            "provenance_sha256": _sha(provenance),
        }
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_canonical_config_is_exact_16_by_2_and_keeps_protocol_specific_target_counts(
    tmp_path: Path,
) -> None:
    contract = builder.load_contract(_fixture_config(tmp_path))

    assert len(contract.jobs) == 32
    assert {job.model_key for job in contract.jobs} == set(builder.MODEL_PROTOCOLS)
    assert {(job.domain, job.protocol, job.expected_samples) for job in contract.jobs} == {
        ("source", "VT", 1876),
        ("source", "VA", 1934),
        ("target", "VT", 2035),
        ("target", "VA", 2190),
    }
    phi4 = next(
        job
        for job in contract.jobs
        if job.domain == "source" and job.model_key == "phi4_multimodal"
    )
    assert phi4.artifacts["misread_labels"].status == "pending"
    assert "Phi-4 source" in phi4.artifacts["misread_labels"].reason


def test_pending_target_gt_is_reported_and_complete_publication_fails_closed(
    tmp_path: Path,
) -> None:
    contract = builder.load_contract(_fixture_config(tmp_path))

    report = builder.audit_contract(contract)

    assert report["status"] == "PENDING"
    assert report["publishable"] is False
    assert report["contract"]["target_intersection_policy"] == ("protocol_specific_no_intersection")
    assert report["counts"] == {
        "pending_artifacts": 128,
        "invalid_checks": 0,
        "ready_artifacts": 0,
        "total_artifacts": 128,
    }
    target_gt = [
        row
        for row in report["pending"]
        if row["job_id"].startswith("target:") and row["artifact"] == "ground_truth"
    ]
    assert len(target_gt) == 16
    assert all("GT_DESCRIPTION is missing" in row["reason"] for row in target_gt)
    with pytest.raises(
        builder.BundleContractError,
        match="complete bundle publication is fail-closed",
    ):
        builder.build_bundle(contract, tmp_path / "forbidden_complete_bundle")


def test_wrong_target_va_intersection_count_is_rejected(tmp_path: Path) -> None:
    path = _fixture_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    target_va = next(
        row for row in payload["datasets"] if row["domain"] == "target" and row["protocol"] == "VA"
    )
    target_va["expected_samples"] = 2035
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(builder.BundleContractError, match="dataset .* count drift"):
        builder.load_contract(path)


def test_ready_artifact_requires_payload_and_provenance_checksums(
    tmp_path: Path,
) -> None:
    path = _fixture_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["jobs"][0]["artifacts"]["cache"] = {
        "status": "ready",
        "path": "missing_cache.json",
        "sha256": "0" * 64,
        "provenance_path": "missing_cache.provenance.json",
        "provenance_sha256": "1" * 64,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = builder.audit_contract(builder.load_contract(path))

    assert report["status"] == "BLOCKED"
    assert report["publishable"] is False
    assert any("missing_cache.json" in error for error in report["failures"])
    assert any("missing_cache.provenance.json" in error for error in report["failures"])


def test_readiness_writer_emits_json_and_markdown(tmp_path: Path) -> None:
    report = builder.audit_contract(builder.load_contract(_fixture_config(tmp_path)))

    json_path, markdown_path = builder.write_readiness(report, tmp_path / "readiness.json")

    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["schema"] == builder.READINESS_SCHEMA
    assert persisted["status"] == "PENDING"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Status: **PENDING**" in markdown
    assert "target:phi4_multimodal" in markdown
