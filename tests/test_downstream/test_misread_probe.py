from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from mprisk.evaluation.misread_probe import (
    run_conflict_misread_probe,
    write_pending_conflict_misread_probe,
)

SPLIT_SHA = "a" * 64
FORMAL_ROOT_SCHEMA = "mprisk_formal_misread_labels_root_v1"
FORMAL_ROW_SCHEMA = "mprisk_imported_misread_label_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _canonical_rows_sha256(rows: list[dict]) -> str:
    content = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in sorted(rows, key=lambda row: row["sample_id"])
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _sample_ids_sha256(sample_ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(sample_ids), separators=(",", ":")).encode()
    ).hexdigest()


def _is_eligible_conflict(row: dict) -> bool:
    return (
        row["probe_eligible"] is True
        and row["label_eligible"] is True
        and row["blocked"] is False
        and row["needs_manual_review"] is False
        and row["imported_label"] in {"MISREAD", "NON_MISREAD"}
        and row["sample_type"] == "Conflict"
    )


def _formal_row(
    sample_id: str,
    group: str,
    split: str,
    label: str | None,
    *,
    manual: bool = False,
) -> dict:
    eligible = label is not None
    return {
        "schema": FORMAL_ROW_SCHEMA,
        "sample_id": sample_id,
        "source_id": f"source:{sample_id}",
        "subject_model_key": "fixture-model",
        "protocol": "VT",
        "sample_type": "Conflict",
        "split_group_id": group,
        "master_split": "test",
        "representation_split": split,
        "representative_probe_model": True,
        "judge_model": "deepseek-v4-flash",
        "judge_confidence": None if manual else 0.9,
        "raw_judge_decision": None if manual else label,
        "source_final_label": None if manual else label,
        "imported_label": label,
        "needs_manual_review": manual,
        "manual_review_reasons": ["fixture_manual_review"] if manual else [],
        "blocked": False,
        "blocked_reason": None,
        "label_eligible": eligible,
        "probe_eligible": eligible,
        "diagnostic_description_sha256": "b" * 64,
        "gt_description_sha256": "c" * 64,
    }


def _refresh_formal_root(payload: dict, *, update_eligible_lock: bool = True) -> None:
    root = Path(payload["labels"]["root"])
    labels = root / "labels/fixture-model.jsonl"
    checksums = root / "artifact_checksums.json"
    relative = "labels/fixture-model.jsonl"
    _json(
        checksums,
        {
            "schema": "mprisk_artifact_checksums_v1",
            "artifacts": {relative: {"bytes": labels.stat().st_size, "sha256": _sha256(labels)}},
        },
    )
    marker_path = root / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact_checksums_sha256"] = _sha256(checksums)
    _json(marker_path, marker)
    marker_sha = _sha256(marker_path)
    (root / "COMPLETE.json.sha256").write_text(f"{marker_sha}  COMPLETE.json\n", encoding="utf-8")
    payload["labels"]["complete_sha256"] = marker_sha
    payload["labels"]["artifact_sha256"] = _sha256(labels)
    if update_eligible_lock:
        rows = [json.loads(line) for line in labels.read_text().splitlines()]
        payload["labels"]["expected_eligible_rows_sha256"] = _canonical_rows_sha256(
            [row for row in rows if _is_eligible_conflict(row)]
        )


def _fixtures(tmp_path: Path) -> tuple[Path, list[Path], list[str], list[str]]:
    samples: list[tuple[str, str, str, str]] = []
    for split, count in (("relation_train", 8), ("relation_val", 4), ("official_test", 4)):
        for index in range(count):
            label = "NON_MISREAD" if index % 2 == 0 else "MISREAD"
            sample_id = f"{split}-{index}"
            samples.append((sample_id, f"group-{sample_id}", split, label))
    manual_samples = [
        (f"manual-{index}", f"group-manual-{index}", "official_test", None) for index in range(3)
    ]
    all_samples = [*samples, *manual_samples]
    label_rows = [
        _formal_row(sample_id, group, split, label, manual=label is None)
        for sample_id, group, split, label in all_samples
    ]
    label_root = tmp_path / "formal-label-root"
    labels = _jsonl(label_root / "labels/fixture-model.jsonl", label_rows)
    _json(
        label_root / "COMPLETE.json",
        {
            "schema": FORMAL_ROOT_SCHEMA,
            "status": "partial_manual_review_required",
            "eligible_subset_complete": True,
            "resolved_count": len(samples),
            "unresolved_count": len(manual_samples),
            "artifact_checksums_sha256": "0" * 64,
            "counts": {
                "rows": len(all_samples),
                "probe_eligible": len(samples),
                "needs_manual_review": len(manual_samples),
                "blocked": 0,
            },
            "models": ["fixture-model"],
        },
    )
    specs = (
        ("single_point", "single_point_binary_v1", "penultimate_feature", 2),
        ("trajectory_mlp", "trajectory_mlp_binary_v1", "penultimate_feature", 3),
        ("tme", "tme_proxy_anchor_v1", "sample_relation_feature", 4),
    )
    manifests = []
    for name, repr_key, feature_field, dim in specs:
        rows = []
        for sample_id, group, split, label in all_samples:
            sign = -1.0 if label in {"NON_MISREAD", None} else 1.0
            rows.append(
                {
                    "schema": "frozen-fixture-v1",
                    "sample_id": sample_id,
                    "sample_type": "Conflict",
                    "split_group_id": group,
                    "representation_split": split,
                    "split_assignment_key": "probe-split-v1",
                    "split_assignment_sha256": SPLIT_SHA,
                    "repr_key": repr_key,
                    "model_key": "fixture-model",
                    "protocol": "vt",
                    "prompt_set_key": "fixture-p8",
                    feature_field: [sign * (index + 1) for index in range(dim)],
                }
            )
        manifests.append(_jsonl(tmp_path / f"{name}.jsonl", rows))
    return (
        labels,
        manifests,
        [sample[0] for sample in samples],
        [sample[0] for sample in manual_samples],
    )


def _config(tmp_path: Path, *, mutate=None) -> Path:
    labels, manifests, sample_ids, _ = _fixtures(tmp_path)
    payload = {
        "schema": "mprisk_conflict_misread_probe_config_v1",
        "status": "ready",
        "run_id": "probe-test-v1",
        "model_key": "fixture-model",
        "protocol": "vt",
        "prompt_set_key": "fixture-p8",
        "labels": {
            "root": str(labels.parents[1]),
            "complete_sha256": "0" * 64,
            "artifact_sha256": "0" * 64,
            "expected_eligible_rows_sha256": "0" * 64,
        },
        "representations": [
            {
                "name": "single_point",
                "path": str(manifests[0]),
                "sha256": _sha256(manifests[0]),
                "repr_key": "single_point_binary_v1",
                "feature_field": "penultimate_feature",
                "expected_feature_dim": 2,
            },
            {
                "name": "trajectory_mlp",
                "path": str(manifests[1]),
                "sha256": _sha256(manifests[1]),
                "repr_key": "trajectory_mlp_binary_v1",
                "feature_field": "penultimate_feature",
                "expected_feature_dim": 3,
            },
            {
                "name": "tme",
                "path": str(manifests[2]),
                "sha256": _sha256(manifests[2]),
                "repr_key": "tme_proxy_anchor_v1",
                "feature_field": "sample_relation_feature",
                "expected_feature_dim": 4,
            },
        ],
        "split_assignment_key": "probe-split-v1",
        "split_assignment_sha256": SPLIT_SHA,
        "expected_sample_ids_sha256": _sample_ids_sha256(sample_ids),
        "training": {
            "seed": 7,
            "epochs": 12,
            "batch_size": 4,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "hidden_dim": 128,
            "dropout": 0.1,
            "device": "cpu",
        },
        "output_root": str(tmp_path / "output"),
    }
    _refresh_formal_root(payload)
    if mutate is not None:
        mutate(payload)
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_pending_probe_never_generates_labels_or_starts_training(tmp_path: Path) -> None:
    path = write_pending_conflict_misread_probe(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "Pending Misread annotations"
    assert payload["eligible_sample_type"] == "Conflict"
    assert payload["representation_policy"] == "frozen_no_encoder_gradients"
    assert payload["generated_labels"] == 0
    assert payload["pseudo_labels"] == 0
    assert payload["training_started"] is False


def test_unified_probe_consumes_only_locked_formal_eligible_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config_payload = yaml.safe_load(config.read_text())
    _, _, expected_ids, manual_ids = _fixtures(tmp_path / "unused")
    result = run_conflict_misread_probe(config)
    marker = json.loads(Path(result["run_complete_path"]).read_text())

    assert marker["status"] == "complete"
    assert marker["formal_label_root"]["status"] == "partial_manual_review_required"
    assert marker["formal_label_root"]["eligible_subset_complete"] is True
    assert marker["excluded_label_counts"]["manual_review"] == 3
    assert marker["sample_ids_sha256"] == _sample_ids_sha256(expected_ids)
    assert (
        marker["eligible_labels_sha256"]
        == config_payload["labels"]["expected_eligible_rows_sha256"]
    )
    eligible_rows = [
        json.loads(line) for line in Path(marker["eligible_labels"]).read_text().splitlines()
    ]
    assert {row["sample_id"] for row in eligible_rows} == set(expected_ids)
    assert not ({row["sample_id"] for row in eligible_rows} & set(manual_ids))
    assert all(_is_eligible_conflict(row) for row in eligible_rows)
    assert [row["representation"] for row in marker["representations"]] == [
        "single_point",
        "trajectory_mlp",
        "tme",
    ]
    assert _sha256(Path(result["run_complete_path"])) == result["run_complete_sha256"]
    assert (
        Path(result["run_complete_sha256_path"]).read_text().strip()
        == result["run_complete_sha256"]
    )
    for representation in marker["representations"]:
        assert representation["metrics"] == {
            **representation["metrics"],
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "ap": 1.0,
        }
        for artifact in representation["artifacts"].values():
            assert _sha256(Path(artifact["path"])) == artifact["sha256"]
        checkpoint = torch.load(
            representation["artifacts"]["checkpoint"]["path"], map_location="cpu"
        )
        assert checkpoint["architecture"]["shared_across_representations"] is True
        assert checkpoint["training_budget"]["seed"] == 7
        assert checkpoint["sample_ids_sha256"] == marker["sample_ids_sha256"]
        with Path(representation["artifacts"]["pr_curve"]["path"]).open(newline="") as handle:
            assert tuple(csv.DictReader(handle).fieldnames or ()) == (
                "threshold",
                "recall",
                "precision",
            )


def test_probe_rejects_incomplete_formal_eligible_subset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    marker_path = Path(payload["labels"]["root"]) / "COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["eligible_subset_complete"] = False
    _json(marker_path, marker)
    _refresh_formal_root(payload)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="eligible subset is not complete"):
        run_conflict_misread_probe(config)


def test_probe_requires_schema_on_formal_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    marker_path = Path(payload["labels"]["root"]) / "COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["schema_name"] = marker.pop("schema")
    _json(marker_path, marker)
    _refresh_formal_root(payload)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="schema mismatch"):
        run_conflict_misread_probe(config)


def test_probe_rejects_manual_review_row_with_fabricated_label(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    labels = Path(payload["labels"]["root"]) / "labels/fixture-model.jsonl"
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    manual = next(row for row in rows if row["needs_manual_review"])
    manual["imported_label"] = "MISREAD"
    _jsonl(labels, rows)
    _refresh_formal_root(payload)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="fabricated eligible label"):
        run_conflict_misread_probe(config)


def test_probe_rejects_group_crossing_fixed_splits(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    labels = Path(payload["labels"]["root"]) / "labels/fixture-model.jsonl"
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    eligible = [row for row in rows if _is_eligible_conflict(row)]
    eligible[-1]["split_group_id"] = eligible[0]["split_group_id"]
    _jsonl(labels, rows)
    _refresh_formal_root(payload)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="crosses fixed probe splits"):
        run_conflict_misread_probe(config)


def test_probe_rejects_duplicate_or_conflicting_labels(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    labels = Path(payload["labels"]["root"]) / "labels/fixture-model.jsonl"
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    duplicate = {**rows[0], "imported_label": "MISREAD"}
    _jsonl(labels, [*rows, duplicate])
    _refresh_formal_root(payload)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="duplicate or conflicting"):
        run_conflict_misread_probe(config)


def test_probe_rejects_representation_intersection_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    manifest = Path(payload["representations"][2]["path"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    eligible_id = next(
        row["sample_id"] for row in rows if not row["sample_id"].startswith("manual")
    )
    _jsonl(manifest, [row for row in rows if row["sample_id"] != eligible_id])
    payload["representations"][2]["sha256"] = _sha256(manifest)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="sample intersection drift"):
        run_conflict_misread_probe(config)


def test_probe_rejects_label_or_split_identity_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    manifest = Path(payload["representations"][0]["path"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["split_group_id"] = "wrong-group"
    _jsonl(manifest, rows)
    payload["representations"][0]["sha256"] = _sha256(manifest)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="split_group_id mismatch"):
        run_conflict_misread_probe(config)


def test_probe_rejects_misread_leakage_in_frozen_features(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    manifest = Path(payload["representations"][1]["path"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["misread_label"] = "Misread"
    _jsonl(manifest, rows)
    payload["representations"][1]["sha256"] = _sha256(manifest)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="Misread leakage"):
        run_conflict_misread_probe(config)


def test_probe_rejects_cross_run_representation_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    manifest = Path(payload["representations"][0]["path"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["model_key"] = "another-model"
    _jsonl(manifest, rows)
    payload["representations"][0]["sha256"] = _sha256(manifest)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="model_key drift"):
        run_conflict_misread_probe(config)
