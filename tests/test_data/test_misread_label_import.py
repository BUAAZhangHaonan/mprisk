from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mprisk.data.misread_labels import (
    ModelImportSpec,
    import_single_flash_labels,
    verify_imported_labels,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _manifest_row(sample_id: str, protocol: str, sample_type: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_id": f"source:{sample_id}",
        "protocol": protocol,
        "sample_type": sample_type,
        "split_group_id": f"group:{sample_id}",
        "split": "test",
        "gt_describe": f"Ground truth for {sample_id}.",
    }


def _description(row: dict, model: str, *, error: bool = False) -> dict:
    value = {
        "schema": "mprisk_v2_diagnostic_description_v1",
        "sample_id": row["sample_id"],
        "subject_model_key": model,
        "protocol": row["protocol"],
        "sample_type": row["sample_type"],
        "condition": "M12",
        "source_id": row["source_id"],
        "gt_describe": row["gt_describe"],
        "diagnostic_description": "A concise affect description.",
    }
    if error:
        value["diagnostic_description"] = ""
        value["error"] = "missing audio"
    return value


def _judgment(row: dict, model: str, decision: str, confidence: float) -> dict:
    source_final = "MISREAD" if decision == "UNCERTAIN" else decision
    return {
        "schema": "mprisk_v2_misread_label_v1",
        "sample_id": row["sample_id"],
        "subject_model_key": model,
        "protocol": row["protocol"],
        "final_label": source_final,
        "arbitrator_used": False,
        "agreement_ratio": 1.0,
        "flash": [
            {
                "judge_model": "deepseek-v4-flash",
                "decision": decision,
                "confidence": confidence,
                "rationale": "fixture",
            }
        ],
        "pro_arbitration": None,
    }


def _fixture(tmp_path: Path) -> dict:
    rows = [
        _manifest_row("vt:a", "VT", "Conflict"),
        _manifest_row("vt:c", "VT", "Aligned"),
        _manifest_row("va:a", "VA", "Conflict"),
        _manifest_row("va:c", "VA", "Aligned"),
        _manifest_row("va:blocked", "VA", "Conflict"),
    ]
    manifest = tmp_path / "delivery.jsonl"
    splits = tmp_path / "splits.jsonl"
    invalid = tmp_path / "invalid.jsonl"
    _write(manifest, rows)
    _write(
        splits,
        [
            {
                "schema": "mprisk_representation_split_assignment_v1",
                "split_group_id": row["split_group_id"],
                "sample_ids": [row["sample_id"]],
                "master_split": row["split"],
                "representation_split": "official_test",
            }
            for row in rows
            if row["sample_id"] != "va:blocked"
        ],
    )
    _write(
        invalid,
        [
            {
                "sample_id": "va:blocked",
                "reason": "missing_audio_stream",
                "protocol": "VA",
            }
        ],
    )
    v2 = tmp_path / "v2"
    settings = {
        "qwen": ("VT", True, {"vt:a": ("UNCERTAIN", 0.0), "vt:c": ("NON_MISREAD", 0.9)}),
        "intern": ("VT", True, {"vt:a": ("MISREAD", 0.9), "vt:c": ("NON_MISREAD", 0.9)}),
        "omni": ("VA", True, {"va:a": ("MISREAD", 0.4), "va:c": ("NON_MISREAD", 0.9)}),
        "gemma": ("VA", False, {"va:a": ("MISREAD", 0.9), "va:c": ("NON_MISREAD", 0.9)}),
    }
    specs = []
    for model, (protocol, representative, judgments) in settings.items():
        model_rows = [row for row in rows if row["protocol"] == protocol]
        model_dir = v2 / "outputs/v2/misread" / model
        desc_path = model_dir / "descriptions.jsonl"
        judge_path = model_dir / "judgments_single_flash.jsonl"
        _write(
            desc_path,
            [
                _description(row, model, error=row["sample_id"] == "va:blocked")
                for row in model_rows
            ],
        )
        _write(
            judge_path,
            [
                _judgment(
                    next(row for row in model_rows if row["sample_id"] == sample_id),
                    model,
                    decision,
                    confidence,
                )
                for sample_id, (decision, confidence) in judgments.items()
            ],
        )
        specs.append(
            ModelImportSpec(
                model,
                protocol,
                len(model_rows),
                len(judgments),
                _sha(desc_path),
                _sha(judge_path),
                representative,
            )
        )
    return {
        "v2": v2,
        "manifest": manifest,
        "splits": splits,
        "invalid": invalid,
        "specs": tuple(specs),
        "output": tmp_path / "imported",
    }


def _run(fixture: dict):
    return import_single_flash_labels(
        v2_root=fixture["v2"],
        delivery_manifest=fixture["manifest"],
        split_assignment=fixture["splits"],
        invalid_assets=fixture["invalid"],
        output_root=fixture["output"],
        model_specs=fixture["specs"],
        expected_delivery_sha256=_sha(fixture["manifest"]),
        expected_split_sha256=_sha(fixture["splits"]),
        expected_invalid_assets_sha256=_sha(fixture["invalid"]),
    )


def test_import_is_read_only_fail_closed_and_materializes_verified_marker(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    source_paths = [fixture["manifest"], fixture["splits"], fixture["invalid"]]
    source_paths += [
        fixture["v2"] / "outputs/v2/misread" / spec.model_key / filename
        for spec in fixture["specs"]
        for filename in ("descriptions.jsonl", "judgments_single_flash.jsonl")
    ]
    hashes_before = {path: _sha(path) for path in source_paths}

    result = _run(fixture)

    assert result.total_rows == 10
    assert result.manual_review_rows == 2
    assert result.blocked_rows == 2
    assert result.probe_eligible_rows == 4
    assert {path: _sha(path) for path in source_paths} == hashes_before
    marker = verify_imported_labels(fixture["output"])
    assert marker["schema"] == "mprisk_formal_misread_labels_root_v1"
    assert marker["status"] == "partial_manual_review_required"
    assert marker["eligible_subset_complete"] is True
    assert marker["resolved_count"] == 6
    assert marker["unresolved_count"] == 4
    summary = json.loads((fixture["output"] / "summary.json").read_text())
    assert summary["models"]["intern"]["overall"]["misread_rate"] == 0.5
    assert summary["models"]["qwen"]["overall"]["label_eligible"] == 1
    assert summary["models"]["omni"]["overall"]["blocked"] == 1
    gemma_rows = [
        json.loads(line)
        for line in (fixture["output"] / "labels/gemma.jsonl").read_text().splitlines()
    ]
    assert all(not row["probe_eligible"] for row in gemma_rows)
    qwen_rows = [
        json.loads(line)
        for line in (fixture["output"] / "labels/qwen.jsonl").read_text().splitlines()
    ]
    uncertain = next(row for row in qwen_rows if row["sample_id"] == "vt:a")
    assert uncertain["source_final_label"] == "MISREAD"
    assert uncertain["imported_label"] is None
    assert uncertain["needs_manual_review"] is True
    with pytest.raises(FileExistsError, match="immutable"):
        _run(fixture)


def test_duplicate_judgment_fails_before_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec = fixture["specs"][0]
    path = fixture["v2"] / "outputs/v2/misread" / spec.model_key / "judgments_single_flash.jsonl"
    first = path.read_text().splitlines()[0]
    path.write_text(path.read_text() + first + "\n")
    fixture["specs"] = (
        ModelImportSpec(
            spec.model_key,
            spec.protocol,
            spec.expected_descriptions,
            spec.expected_judgments + 1,
            spec.descriptions_sha256,
            _sha(path),
            spec.representative_probe_model,
        ),
        *fixture["specs"][1:],
    )
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        _run(fixture)
    assert not fixture["output"].exists()


def test_missing_judgment_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec = fixture["specs"][1]
    path = fixture["v2"] / "outputs/v2/misread" / spec.model_key / "judgments_single_flash.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    fixture["specs"] = tuple(
        ModelImportSpec(
            item.model_key,
            item.protocol,
            item.expected_descriptions,
            item.expected_judgments,
            item.descriptions_sha256,
            _sha(path) if item.model_key == spec.model_key else item.judgments_sha256,
            item.representative_probe_model,
        )
        for item in fixture["specs"]
    )
    with pytest.raises(ValueError, match="judgments count"):
        _run(fixture)
    assert not fixture["output"].exists()


def test_protocol_type_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec = fixture["specs"][0]
    path = fixture["v2"] / "outputs/v2/misread" / spec.model_key / "descriptions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["protocol"] = "VA"
    _write(path, rows)
    fixture["specs"] = (
        ModelImportSpec(
            spec.model_key,
            spec.protocol,
            spec.expected_descriptions,
            spec.expected_judgments,
            _sha(path),
            spec.judgments_sha256,
            spec.representative_probe_model,
        ),
        *fixture["specs"][1:],
    )
    with pytest.raises(ValueError, match="Description protocol mismatch"):
        _run(fixture)
    assert not fixture["output"].exists()


def test_pinned_source_sha_mismatch_fails_before_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec = fixture["specs"][0]
    fixture["specs"] = (
        ModelImportSpec(
            spec.model_key,
            spec.protocol,
            spec.expected_descriptions,
            spec.expected_judgments,
            "0" * 64,
            spec.judgments_sha256,
            spec.representative_probe_model,
        ),
        *fixture["specs"][1:],
    )
    with pytest.raises(ValueError, match="descriptions SHA-256 mismatch"):
        _run(fixture)
    assert not fixture["output"].exists()
