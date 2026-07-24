from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from mprisk.viz.formal_misread import (
    FORMAL_METHODS,
    FORMAL_MODELS,
    PROBE_FIELDS,
    canonical_label_rows,
    canonical_metric_rows,
    load_formal_root,
)
from mprisk.viz.misread_figures import export_misread_figures
from scripts.build_misread_figure_adapters import materialize_label_adapter, write_csv


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_formal_labels(root: Path) -> None:
    protocols = {
        "qwen3_vl_8b": "VT",
        "internvl3_5_8b": "VT",
        "qwen2_5_omni_7b": "VA",
        "gemma4_12b_it": "VA",
    }
    models = (*FORMAL_MODELS, "gemma4_12b_it")
    for model in models:
        row = {
            "subject_model_key": model,
            "sample_id": f"sample-{model}",
            "protocol": protocols[model],
            "sample_type": "Conflict",
            "imported_label": "MISREAD",
            "judge_confidence": 0.91,
            "blocked": False,
            "needs_manual_review": False,
            "label_eligible": True,
            "probe_eligible": model in FORMAL_MODELS,
        }
        path = root / "labels" / f"{model}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _write_json(
        root / "provenance.json",
        {
            "schema": "mprisk_misread_label_import_provenance_v1",
            "judge_protocol": {
                "judge_model": "deepseek-v4-flash",
                "temperature": 0.0,
                "confidence_threshold": 0.5,
                "n_flash": 1,
            },
            "input_artifacts": {
                "delivery_manifest": {
                    "path": "/frozen/unified_sample_manifest.jsonl",
                    "sha256": "a" * 64,
                },
                "split_assignment": {"path": "/frozen/splits.jsonl", "sha256": "b" * 64},
            },
        },
    )
    _write_json(
        root / "summary.json",
        {
            "schema": "mprisk_misread_label_import_summary_v1",
            "models": {model: {"overall": {"rows": 1}} for model in models},
        },
    )
    artifact_paths = [
        *(root / "labels" / f"{model}.jsonl" for model in models),
        root / "provenance.json",
        root / "summary.json",
    ]
    checksums = {
        "schema": "mprisk_artifact_checksums_v1",
        "artifacts": {
            str(path.relative_to(root)): {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    _write_json(root / "artifact_checksums.json", checksums)
    marker = {
        "schema": "mprisk_formal_misread_labels_root_v1",
        "status": "partial_manual_review_required",
        "eligible_subset_complete": True,
        "models": list(models),
        "counts": {"rows": len(models)},
        "artifact_checksums_sha256": _sha(root / "artifact_checksums.json"),
    }
    _write_json(root / "COMPLETE.json", marker)
    (root / "COMPLETE.json.sha256").write_text(
        f"{_sha(root / 'COMPLETE.json')}  COMPLETE.json\n", encoding="utf-8"
    )


def test_formal_label_adapter_accepts_only_verified_eligible_rows(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    _write_formal_labels(root)

    formal = load_formal_root(root, kind="labels")

    assert formal is not None
    rows = canonical_label_rows(formal)
    assert {row["model"] for row in rows} == set(FORMAL_MODELS)
    assert all(row["label_eligible"] and row["label"] == "MISREAD" for row in rows)

    (root / "labels" / "qwen3_vl_8b.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        load_formal_root(root, kind="labels")


def test_completed_probe_root_fails_closed_on_artifact_mutation(tmp_path: Path) -> None:
    root = tmp_path / "probes"
    root.mkdir()
    metrics = root / "probe_metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "model": "qwen3_vl_8b",
                "protocol": "VT",
                "method": "TME",
                "seed": 1,
                "accuracy": 0.7,
                "macro_f1": 0.69,
                "auprc": 0.72,
                "latency_ms": "",
                "n_train": 10,
                "n_val": 4,
                "n_test": 6,
                "test_sample_ids_sha256": "c" * 64,
                "label_artifact_sha256": "d" * 64,
                "status": "Ready",
            }
        )
    _write_json(
        root / "COMPLETE.json",
        {
            "schema": "mprisk_formal_misread_probe_root_v1",
            "status": "complete",
            "dataset_id": "delivery_20260716",
            "split_assignment_sha256": "b" * 64,
            "generated_command": ["python", "scripts/run_conflict_misread_probe.py"],
            "methods": list(FORMAL_METHODS),
            "artifacts": [
                {
                    "role": "probe_metrics",
                    "path": metrics.name,
                    "sha256": _sha(metrics),
                }
            ],
        },
    )
    formal = load_formal_root(root, kind="probes")
    assert formal is not None
    rows = canonical_metric_rows(formal, role="probe_metrics", fields=PROBE_FIELDS)
    assert rows[0]["latency_ms"] is None

    metrics.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_formal_root(root, kind="probes")


def test_canonical_adapter_copy_is_verified_and_csv_hashes_are_git_stable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-labels"
    output = tmp_path / "canonical-labels"
    _write_formal_labels(source)

    result = materialize_label_adapter(source_root=source, output_root=output)

    assert result == output.resolve()
    assert _sha(result / "COMPLETE.json") == _sha(source / "COMPLETE.json")
    metrics = tmp_path / "probe_metrics.csv"
    write_csv(metrics, ("model", "accuracy"), [{"model": "qwen3_vl_8b", "accuracy": 0.75}])
    assert b"\r\n" not in metrics.read_bytes()
    assert metrics.read_bytes().endswith(b"\n")


def test_misread_export_defaults_to_additive_canonical_roots() -> None:
    defaults = export_misread_figures.__kwdefaults__

    assert defaults is not None
    assert defaults["input_root"] == "outputs/paper_exports/figures/misread"
    assert defaults["output_root"] == "paper/figures/generated/misread"
    assert defaults["labels_root"].endswith("/misread/adapters/labels")
    assert defaults["probes_root"].endswith("/misread/adapters/probes")
    assert defaults["budgets_root"].endswith("/misread/adapters/budgets")
