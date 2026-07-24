#!/usr/bin/env python3
"""Bind completed queue artifacts to the formal Misread figure input contract.

The queue and conflict-supervision trees are immutable experiment evidence.  This
adapter only reads them, verifies every referenced checksum, and writes small
CSV/marker roots under the additive figure-input tree.  It does not copy,
rewrite, or reinterpret source experiment files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

MODELS = ("qwen2_5_omni_7b", "qwen3_vl_8b", "internvl3_5_8b")
METHODS = ("Single-Point", "Trajectory MLP", "TME")
METHOD_TO_QUEUE = {
    "Single-Point": "single_point",
    "Trajectory MLP": "trajectory_mlp",
    "TME": "tme",
}
FORMAL_PROBE_FIELDS = (
    "model",
    "protocol",
    "method",
    "seed",
    "accuracy",
    "macro_f1",
    "auprc",
    "latency_ms",
    "n_train",
    "n_val",
    "n_test",
    "test_sample_ids_sha256",
    "label_artifact_sha256",
    "status",
)
FORMAL_BUDGET_FIELDS = (
    "model",
    "protocol",
    "method",
    "budget_pct",
    "seed",
    "accuracy",
    "macro_f1",
    "auprc",
    "n_conflict_supervision",
    "n_aligned_supervision",
    "n_train",
    "n_val",
    "n_test",
    "test_sample_ids_sha256",
    "label_artifact_sha256",
    "encoder_checkpoint_sha256",
    "status",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_sha(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"source checksum mismatch: {path}: {actual} != {expected}")


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256(path)


def label_artifact_sha(labels_root: Path, model: str) -> str:
    checksums = load_json(labels_root / "artifact_checksums.json")
    relative = f"labels/{model}.jsonl"
    evidence = (checksums.get("artifacts") or {}).get(relative)
    if not evidence:
        raise RuntimeError(f"missing label artifact checksum: {relative}")
    path = labels_root / relative
    require_sha(path, str(evidence["sha256"]))
    return str(evidence["sha256"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return sha256(path)


def add_source(sources: list[dict[str, str]], path: Path, expected: str | None = None) -> None:
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"source checksum mismatch: {path}: {actual} != {expected}")
    sources.append({"path": str(path), "sha256": actual})


def probe_adapter(
    *,
    labels_root: Path,
    queue_root: Path,
    output_root: Path,
    split_assignment_sha256: str,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    metrics_path = output_root / "probe_metrics.csv"
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    queue_complete = queue_root / "MISREAD_BUDGET_COMPLETE.json"
    queue_marker = load_json(queue_complete)
    add_source(
        sources, queue_complete, str(queue_marker.get("metrics_csv_sha256")) if False else None
    )
    queue_metrics = queue_root / "misread_budget_probe_metrics.csv"
    add_source(sources, queue_metrics, str(queue_marker["metrics_csv_sha256"]))
    for model in MODELS:
        label_sha = label_artifact_sha(labels_root, model)
        for fraction in (0.10, 0.25, 0.50, 1.00):
            marker_path = (
                queue_root / model / f"fraction_{fraction:.2f}" / "FRACTION_PROBE_COMPLETE.json"
            )
            marker = load_json(marker_path)
            if marker.get("status") != "complete":
                raise RuntimeError(f"probe fraction is not complete: {marker_path}")
            add_source(sources, marker_path)
            run_path = Path(marker["probe_run_complete"]["path"])
            require_sha(run_path, marker["probe_run_complete"]["sha256"])
            add_source(sources, run_path, marker["probe_run_complete"]["sha256"])
            if fraction != 1.00:
                continue
            run = load_json(run_path)
            for entry in run["representations"]:
                method = {
                    "single_point": "Single-Point",
                    "trajectory_mlp": "Trajectory MLP",
                    "tme": "TME",
                }[entry["representation"]]
                values = entry["metrics"]
                provenance = load_json(Path(entry["artifacts"]["provenance"]["path"]))
                rows.append(
                    {
                        "model": model,
                        "protocol": str(marker["protocol"]).upper(),
                        "method": method,
                        "seed": seed,
                        "accuracy": values["accuracy"],
                        "macro_f1": values["macro_f1"],
                        "auprc": values["ap"],
                        "latency_ms": "Pending",
                        "n_train": provenance["sample_counts"]["relation_train"],
                        "n_val": provenance["sample_counts"]["relation_val"],
                        "n_test": provenance["sample_counts"]["official_test"],
                        "test_sample_ids_sha256": marker["official_test_sample_ids_sha256"],
                        "label_artifact_sha256": label_sha,
                        "status": "Ready",
                    }
                )
    if len(rows) != len(MODELS) * len(METHODS):
        raise RuntimeError(f"expected 9 probe rows, got {len(rows)}")
    metrics_sha = write_csv(metrics_path, FORMAL_PROBE_FIELDS, rows)
    marker = {
        "schema": "mprisk_formal_misread_probe_root_v1",
        "status": "complete",
        "dataset_id": "delivery_20260716",
        "split_assignment_sha256": split_assignment_sha256,
        "generated_command": ["python", "scripts/build_misread_figure_adapters.py"],
        "methods": list(METHODS),
        "models": list(MODELS),
        "seed": seed,
        "latency_status": "Pending: queue artifacts did not record probe latency",
        "artifacts": [{"role": "probe_metrics", "path": metrics_path.name, "sha256": metrics_sha}],
        "source_records": sources,
    }
    marker_path = output_root / "COMPLETE.json"
    marker_sha = write_json(marker_path, marker)
    write_json(
        output_root / "figure_input_manifest.json",
        {
            "kind": "probes",
            "marker": str(marker_path),
            "marker_sha256": marker_sha,
            "sources": sources,
            "rows": rows,
        },
    )
    return marker_path, marker


def _aligned_count(dataset_path: Path) -> int:
    groups: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("sample_type") == "Aligned"
                and row.get("representation_split") == "relation_train"
            ):
                groups.add(str(row["sample_id"]))
    return len(groups)


def budget_adapter(
    *,
    labels_root: Path,
    queue_root: Path,
    conflict_root: Path,
    output_root: Path,
    split_assignment_sha256: str,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    queue_complete = load_json(queue_root / "MISREAD_BUDGET_COMPLETE.json")
    queue_metrics_path = queue_root / "misread_budget_probe_metrics.csv"
    queue_metrics = read_csv(queue_metrics_path)
    sources: list[dict[str, str]] = []
    add_source(sources, queue_root / "MISREAD_BUDGET_COMPLETE.json")
    add_source(sources, queue_metrics_path, queue_complete["metrics_csv_sha256"])
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        label_sha = label_artifact_sha(labels_root, model)
        budget_complete_path = conflict_root / f"BUDGET_COMPLETE.{model}.json"
        budget_complete = load_json(budget_complete_path)
        add_source(sources, budget_complete_path)
        ac_rows = read_csv(Path(budget_complete["ac_metrics_csv"]))
        add_source(
            sources,
            Path(budget_complete["ac_metrics_csv"]),
            budget_complete["ac_metrics_csv_sha256"],
        )
        for ac in ac_rows:
            fraction = float(ac["conflict_supervision_fraction"])
            method_queue = ac["method"]
            method = {
                "single_point": "Single-Point",
                "trajectory_mlp": "Trajectory MLP",
                "tme": "TME",
            }[method_queue]
            conflict_fraction_marker = (
                conflict_root / model / f"fraction_{fraction:.2f}" / "FRACTION_COMPLETE.json"
            )
            load_json(conflict_fraction_marker)
            add_source(sources, conflict_fraction_marker)
            method_marker_path = (
                conflict_root
                / model
                / f"fraction_{fraction:.2f}"
                / method_queue
                / "RUN_COMPLETE.json"
            )
            method_marker = load_json(method_marker_path)
            add_source(sources, method_marker_path)
            queue_fraction_marker = (
                queue_root / model / f"fraction_{fraction:.2f}" / "FRACTION_PROBE_COMPLETE.json"
            )
            queue_fraction = load_json(queue_fraction_marker)
            add_source(sources, queue_fraction_marker)
            run_path = Path(queue_fraction["probe_run_complete"]["path"])
            require_sha(run_path, queue_fraction["probe_run_complete"]["sha256"])
            run = load_json(run_path)
            add_source(sources, run_path, queue_fraction["probe_run_complete"]["sha256"])
            probe_entry = next(
                item for item in run["representations"] if item["representation"] == method_queue
            )
            provenance = load_json(Path(probe_entry["artifacts"]["provenance"]["path"]))
            next(
                item
                for item in queue_metrics
                if item["model_key"] == model
                and float(item["fraction"]) == fraction
                and item["representation"] == method_queue
            )
            training_dataset = Path(method_marker["training_relation_dataset"])
            n_aligned = _aligned_count(training_dataset)
            rows.append(
                {
                    "model": model,
                    "protocol": str(ac["protocol"]).upper(),
                    "method": method,
                    "budget_pct": int(round(100 * fraction)),
                    "seed": seed,
                    "accuracy": probe_entry["metrics"]["accuracy"],
                    "macro_f1": probe_entry["metrics"]["macro_f1"],
                    "auprc": probe_entry["metrics"]["ap"],
                    "n_conflict_supervision": int(ac["retained_conflict_train_groups"]),
                    "n_aligned_supervision": n_aligned,
                    "n_train": provenance["sample_counts"]["relation_train"],
                    "n_val": provenance["sample_counts"]["relation_val"],
                    "n_test": provenance["sample_counts"]["official_test"],
                    "test_sample_ids_sha256": queue_fraction["official_test_sample_ids_sha256"],
                    "label_artifact_sha256": label_sha,
                    "encoder_checkpoint_sha256": method_marker["best_checkpoint_sha256"],
                    "status": "Ready",
                }
            )
    expected = len(MODELS) * 4 * len(METHODS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} budget rows, got {len(rows)}")
    metrics_path = output_root / "budget_metrics.csv"
    metrics_sha = write_csv(metrics_path, FORMAL_BUDGET_FIELDS, rows)
    marker = {
        "schema": "mprisk_formal_conflict_budget_root_v1",
        "status": "complete",
        "dataset_id": "delivery_20260716",
        "split_assignment_sha256": split_assignment_sha256,
        "generated_command": ["python", "scripts/build_misread_figure_adapters.py"],
        "methods": list(METHODS),
        "models": list(MODELS),
        "representative_model": "qwen3_vl_8b",
        "budget_pct": [10, 25, 50, 100],
        "seed": seed,
        "seeds": [seed],
        "artifacts": [{"role": "budget_metrics", "path": metrics_path.name, "sha256": metrics_sha}],
        "source_records": sources,
        "representation_policy": "frozen_no_encoder_gradients",
    }
    marker_path = output_root / "COMPLETE.json"
    marker_sha = write_json(marker_path, marker)
    write_json(
        output_root / "figure_input_manifest.json",
        {
            "kind": "budgets",
            "marker": str(marker_path),
            "marker_sha256": marker_sha,
            "sources": sources,
            "rows": rows,
        },
    )
    return marker_path, marker


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    labels_root = repo / "outputs/labels/delivery_20260716_single_flash_v1"
    queue_root = repo / "outputs/downstream/delivery_20260716/seed20260717/misread_budget_probe_v1"
    conflict_root = (
        repo / "outputs/downstream/delivery_20260716/seed20260717/conflict_supervision_budget_v1"
    )
    output = repo / "outputs/paper_exports/figures/misread/adapters"
    output.mkdir(parents=True, exist_ok=True)
    provenance = load_json(labels_root / "provenance.json")
    split_sha = provenance["input_artifacts"]["split_assignment"]["sha256"]
    probe_path, _ = probe_adapter(
        labels_root=labels_root,
        queue_root=queue_root,
        output_root=output / "probes",
        split_assignment_sha256=split_sha,
        seed=20260717,
    )
    budget_path, _ = budget_adapter(
        labels_root=labels_root,
        queue_root=queue_root,
        conflict_root=conflict_root,
        output_root=output / "budgets",
        split_assignment_sha256=split_sha,
        seed=20260717,
    )
    manifest = {
        "schema": "mprisk_misread_figure_adapter_manifest_v1",
        "labels_root": str(labels_root),
        "probe_root": str(probe_path.parent),
        "budget_root": str(budget_path.parent),
        "probe_marker_sha256": sha256(probe_path),
        "budget_marker_sha256": sha256(budget_path),
        "source_read_only": True,
    }
    write_json(output / "figure_input_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
