"""Delivery-locked Conflict-supervision budget experiments.

Each budget retrains the complete C/A encoder, freezes it, and then applies it
to the unchanged full relation dataset.  The downstream Conflict-only probe
therefore sees the same registered samples at 10/25/50/100 percent and differs
only in the C/A supervision used to learn the frozen representation.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from mprisk.evaluation.downstream_metrics import evaluate_official_representation
from mprisk.experiments.downstream import _train_until_converged
from mprisk.representation.relation_models import (
    SINGLE_POINT_BINARY_V1,
    TME_PROXY_ANCHOR_V1,
    TRAJECTORY_MLP_BINARY_V1,
)
from mprisk.representation.training import (
    export_frozen_baseline_probe_representations,
    export_frozen_baseline_representations,
    export_frozen_representations,
    load_training_config,
)
from mprisk.utils.io import write_json, write_jsonl

CONFIG_SCHEMA = "mprisk_conflict_supervision_budget_config_v1"
METHOD_COMPLETE_SCHEMA = "mprisk_conflict_supervision_budget_method_complete_v1"
FRACTION_COMPLETE_SCHEMA = "mprisk_conflict_supervision_budget_fraction_complete_v1"
RUN_COMPLETE_SCHEMA = "mprisk_conflict_supervision_budget_run_complete_v1"
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
PROBE_SPLITS = ("relation_train", "relation_val", "official_test")
METHOD_REPR_KEYS = {
    "single_point": SINGLE_POINT_BINARY_V1,
    "trajectory_mlp": TRAJECTORY_MLP_BINARY_V1,
    "tme": TME_PROXY_ANCHOR_V1,
}


class BudgetPlanError(ValueError):
    """Raised when a budget plan or a resumable artifact has drifted."""


@dataclass(frozen=True)
class BudgetMethod:
    name: str
    training_config: Path
    training_config_sha256: str


@dataclass(frozen=True)
class BudgetJob:
    model_key: str
    protocol: str
    relation_dataset: Path
    relation_dataset_sha256: str
    methods: tuple[BudgetMethod, ...]


@dataclass(frozen=True)
class BudgetPlan:
    path: Path
    output_root: Path
    lock_path: Path
    seed: int
    device: str
    max_gpu_memory_fraction: float
    fractions: tuple[float, ...]
    jobs: tuple[BudgetJob, ...]


def load_budget_plan(path: str | Path) -> BudgetPlan:
    """Load a fully SHA-bound delivery budget configuration."""
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "delivery",
            "seed",
            "fractions",
            "output_root",
            "lock_path",
            "resource_gate",
            "jobs",
        },
        "budget config",
    )
    if payload["schema"] != CONFIG_SCHEMA or payload["status"] != "ready":
        raise BudgetPlanError("budget config must use the registered ready schema")
    if payload["delivery"] != "delivery_20260716":
        raise BudgetPlanError("budget config must be bound to delivery_20260716")
    fractions = tuple(float(value) for value in payload["fractions"])
    if fractions != FRACTIONS:
        raise BudgetPlanError("fractions must be exactly 0.10/0.25/0.50/1.00")
    root = config_path.parents[2]
    gate = payload["resource_gate"]
    _require_exact_keys(gate, {"device", "max_gpu_memory_fraction"}, "resource gate")
    device = str(gate["device"])
    memory_fraction = float(gate["max_gpu_memory_fraction"])
    if not device.startswith("cuda:"):
        raise BudgetPlanError("budget training requires an explicit CUDA device")
    if not 0.0 < memory_fraction < 0.9:
        raise BudgetPlanError("max_gpu_memory_fraction must remain below 0.9")
    raw_jobs = payload["jobs"]
    if not isinstance(raw_jobs, list) or len(raw_jobs) != 3:
        raise BudgetPlanError("budget config requires exactly three representative models")
    jobs: list[BudgetJob] = []
    for raw_job in raw_jobs:
        _require_exact_keys(
            raw_job,
            {"model_key", "protocol", "relation_dataset", "methods"},
            "budget job",
        )
        relation = _pinned_file(root, raw_job["relation_dataset"], "relation dataset")
        methods_payload = raw_job["methods"]
        if set(methods_payload) != set(METHOD_REPR_KEYS):
            raise BudgetPlanError("every job requires Single-Point, Trajectory MLP, and TME")
        methods: list[BudgetMethod] = []
        for name in METHOD_REPR_KEYS:
            training_path = _pinned_file(root, methods_payload[name], f"{name} training config")
            training_config = load_training_config(training_path)
            if training_config.repr_key != METHOD_REPR_KEYS[name]:
                raise BudgetPlanError(f"training config representation drift for {name}")
            if training_config.model_key != raw_job["model_key"]:
                raise BudgetPlanError(f"training config model drift for {name}")
            if training_config.protocol.lower() != str(raw_job["protocol"]).lower():
                raise BudgetPlanError(f"training config protocol drift for {name}")
            methods.append(
                BudgetMethod(
                    name=name,
                    training_config=training_path,
                    training_config_sha256=_sha256(training_path),
                )
            )
        jobs.append(
            BudgetJob(
                model_key=str(raw_job["model_key"]),
                protocol=str(raw_job["protocol"]).lower(),
                relation_dataset=relation,
                relation_dataset_sha256=_sha256(relation),
                methods=tuple(methods),
            )
        )
    if {job.model_key for job in jobs} != {
        "qwen3_vl_8b",
        "internvl3_5_8b",
        "qwen2_5_omni_7b",
    }:
        raise BudgetPlanError("representative model set drift")
    return BudgetPlan(
        path=config_path,
        output_root=_resolve(root, payload["output_root"]),
        lock_path=_resolve(root, payload["lock_path"]),
        seed=int(payload["seed"]),
        device=device,
        max_gpu_memory_fraction=memory_fraction,
        fractions=fractions,
        jobs=tuple(jobs),
    )


def run_conflict_supervision_budget(
    config_path: str | Path,
    *,
    model_keys: set[str] | None = None,
    method_names: set[str] | None = None,
) -> Path:
    """Run all selected budgets and return the checksummed completion marker."""
    plan = load_budget_plan(config_path)
    selected_models = set(model_keys or (job.model_key for job in plan.jobs))
    unknown_models = selected_models - {job.model_key for job in plan.jobs}
    if unknown_models:
        raise BudgetPlanError(f"unknown model selection: {sorted(unknown_models)}")
    selected_methods = set(method_names or METHOD_REPR_KEYS)
    unknown_methods = selected_methods - set(METHOD_REPR_KEYS)
    if unknown_methods:
        raise BudgetPlanError(f"unknown method selection: {sorted(unknown_methods)}")
    device_index = int(plan.device.split(":", 1)[1])
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise BudgetPlanError(f"configured CUDA device is unavailable: {plan.device}")
    torch.cuda.set_per_process_memory_fraction(plan.max_gpu_memory_fraction, device_index)
    plan.output_root.mkdir(parents=True, exist_ok=True)
    plan.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = plan.lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise BudgetPlanError("another budget runner owns the registered lock") from exc
    completed_markers: list[Path] = []
    consolidated_rows: list[dict[str, Any]] = []
    try:
        for job in plan.jobs:
            if job.model_key not in selected_models:
                continue
            source_rows = _read_jsonl(job.relation_dataset)
            _validate_full_relation_rows(source_rows, job)
            nested_groups: set[str] = set()
            for fraction in plan.fractions:
                fraction_root = plan.output_root / job.model_key / f"fraction_{fraction:.2f}"
                filtered_path, subset = _materialize_fraction_dataset(
                    source_rows=source_rows,
                    source_path=job.relation_dataset,
                    source_sha256=job.relation_dataset_sha256,
                    output_root=fraction_root / "training_relation",
                    fraction=fraction,
                    seed=plan.seed,
                )
                kept_groups = set(subset["retained_conflict_group_ids"])
                if not nested_groups <= kept_groups:
                    raise BudgetPlanError("Conflict budget groups are not nested")
                nested_groups = kept_groups
                method_markers: dict[str, Path] = {}
                probe_sets: dict[str, set[str]] = {}
                for method in job.methods:
                    if method.name not in selected_methods:
                        continue
                    marker, metrics, probe_ids = _run_budget_method(
                        plan=plan,
                        job=job,
                        method=method,
                        fraction=fraction,
                        filtered_dataset=filtered_path,
                        subset=subset,
                        output_root=fraction_root / method.name,
                    )
                    method_markers[method.name] = marker
                    probe_sets[method.name] = probe_ids
                    consolidated_rows.append(
                        {
                            "model_key": job.model_key,
                            "protocol": job.protocol,
                            "method": method.name,
                            "repr_key": METHOD_REPR_KEYS[method.name],
                            "conflict_supervision_fraction": fraction,
                            "retained_conflict_train_groups": subset[
                                "retained_conflict_group_count"
                            ],
                            "available_conflict_train_groups": subset[
                                "available_conflict_group_count"
                            ],
                            "accuracy": metrics["accuracy"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "auprc": metrics["auprc"],
                            "misread_probe_status": "awaiting_bound_probe_run",
                        }
                    )
                if set(method_markers) == selected_methods:
                    if len({frozenset(value) for value in probe_sets.values()}) != 1:
                        raise BudgetPlanError(
                            "frozen Conflict probe sample intersection differs across methods"
                        )
                    shared_ids = next(iter(probe_sets.values()))
                    fraction_marker = write_json(
                        fraction_root / "FRACTION_COMPLETE.json",
                        {
                            "schema": FRACTION_COMPLETE_SCHEMA,
                            "model_key": job.model_key,
                            "fraction": fraction,
                            "full_relation_dataset_sha256": job.relation_dataset_sha256,
                            "training_relation_dataset_sha256": _sha256(filtered_path),
                            "retained_conflict_group_ids_sha256": subset[
                                "retained_conflict_group_ids_sha256"
                            ],
                            "full_conflict_probe_sample_ids_sha256": _sample_ids_sha256(shared_ids),
                            "full_conflict_probe_sample_count": len(shared_ids),
                            "method_markers": {
                                name: {
                                    "path": str(path),
                                    "sha256": _sha256(path),
                                }
                                for name, path in sorted(method_markers.items())
                            },
                            "misread_labels_used_for_encoder_training": False,
                        },
                    )
                    completed_markers.append(fraction_marker)
        csv_path = _write_csv(
            plan.output_root / "conflict_supervision_budget_ac_metrics.csv",
            consolidated_rows,
        )
        marker = write_json(
            plan.output_root / "BUDGET_COMPLETE.json",
            {
                "schema": RUN_COMPLETE_SCHEMA,
                "status": (
                    "complete"
                    if selected_models == {job.model_key for job in plan.jobs}
                    and selected_methods == set(METHOD_REPR_KEYS)
                    else "partial_selection_complete"
                ),
                "delivery": "delivery_20260716",
                "seed": plan.seed,
                "fractions": list(plan.fractions),
                "models": sorted(selected_models),
                "methods": sorted(selected_methods),
                "config": str(plan.path),
                "config_sha256": _sha256(plan.path),
                "fraction_markers": [
                    {"path": str(path), "sha256": _sha256(path)} for path in completed_markers
                ],
                "ac_metrics_csv": str(csv_path),
                "ac_metrics_csv_sha256": _sha256(csv_path),
                "misread_probe_policy": (
                    "same fixed probe-eligible Conflict labels and split intersection at "
                    "every supervision fraction; probe runs are bound after all three "
                    "method artifacts for that model/fraction exist"
                ),
                "misread_labels_used_for_encoder_training": False,
            },
        )
        return marker
    finally:
        lock_handle.close()


def retained_conflict_rows(
    rows: list[dict[str, Any]], *, fraction: float, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retain a deterministic nested fraction of relation-train Conflict groups."""
    if fraction not in FRACTIONS:
        raise BudgetPlanError("unregistered Conflict supervision fraction")
    conflict_groups = sorted(
        {
            str(row["split_group_id"])
            for row in rows
            if row["representation_split"] == "relation_train" and row["sample_type"] == "Conflict"
        },
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    if not conflict_groups:
        raise BudgetPlanError("relation_train has no Conflict groups")
    keep_count = max(1, math.ceil(len(conflict_groups) * fraction))
    kept_groups = set(conflict_groups[:keep_count])
    retained = [
        row
        for row in rows
        if not (
            row["representation_split"] == "relation_train"
            and row["sample_type"] == "Conflict"
            and str(row["split_group_id"]) not in kept_groups
        )
    ]
    protected_splits = {"relation_val", "aligned_calibration", "official_test"}
    protected_before = {
        str(row["row_id"]) for row in rows if row["representation_split"] in protected_splits
    }
    retained_ids = {str(row["row_id"]) for row in retained}
    if not protected_before <= retained_ids:
        raise BudgetPlanError("budget filtering altered a protected split")
    train_types = {
        str(row["sample_type"])
        for row in retained
        if row["representation_split"] == "relation_train"
    }
    if train_types != {"Aligned", "Conflict"}:
        raise BudgetPlanError("budget filtering removed a relation-train class")
    kept_list = sorted(kept_groups)
    return retained, {
        "available_conflict_group_count": len(conflict_groups),
        "retained_conflict_group_count": keep_count,
        "retained_conflict_group_ids": kept_list,
        "retained_conflict_group_ids_sha256": _sample_ids_sha256(set(kept_list)),
    }


def _run_budget_method(
    *,
    plan: BudgetPlan,
    job: BudgetJob,
    method: BudgetMethod,
    fraction: float,
    filtered_dataset: Path,
    subset: dict[str, Any],
    output_root: Path,
) -> tuple[Path, dict[str, Any], set[str]]:
    done = output_root / "RUN_COMPLETE.json"
    if done.is_file():
        return _validate_method_completion(
            marker_path=done,
            plan=plan,
            job=job,
            method=method,
            fraction=fraction,
            filtered_dataset=filtered_dataset,
        )
    config = load_training_config(method.training_config)
    training = _train_until_converged(
        dataset_path=filtered_dataset,
        config=config,
        output_dir=output_root / "training",
        device=plan.device,
    )
    if config.repr_key == TME_PROXY_ANCHOR_V1:
        frozen = export_frozen_representations(
            dataset_path=job.relation_dataset,
            checkpoint_path=training.best_checkpoint_path,
            output_dir=output_root / "frozen_full_relation",
        )
        bundle_rows = _read_jsonl(frozen.bundle_manifest_path)
        official_rows = [
            row for row in bundle_rows if row["representation_split"] == "official_test"
        ]
        probe_rows = [
            row
            for row in bundle_rows
            if row["sample_type"] == "Conflict" and row["representation_split"] in PROBE_SPLITS
        ]
        official_manifest = write_jsonl(
            output_root / "official_test" / "frozen_tme_representations.jsonl",
            official_rows,
        )
        probe_manifest = write_jsonl(
            output_root / "conflict_probe" / "frozen_tme_conflict_probe.jsonl",
            probe_rows,
        )
        frozen_summary = frozen.summary_path
    else:
        official = export_frozen_baseline_representations(
            dataset_path=job.relation_dataset,
            checkpoint_path=training.best_checkpoint_path,
            output_dir=output_root / "official_test",
            representation_split="official_test",
        )
        probe = export_frozen_baseline_probe_representations(
            dataset_path=job.relation_dataset,
            checkpoint_path=training.best_checkpoint_path,
            output_dir=output_root / "conflict_probe",
        )
        official_manifest = official.manifest_path
        probe_manifest = probe.manifest_path
        frozen_summary = probe.summary_path
    evaluation = evaluate_official_representation(
        manifest_path=official_manifest,
        checkpoint_path=training.best_checkpoint_path,
        output_dir=output_root / "official_test" / "ac_evaluation",
    )
    metrics_path = Path(str(evaluation["metrics_path"]))
    probe_rows = _read_jsonl(probe_manifest)
    probe_ids = _validate_probe_rows(probe_rows, job)
    marker = write_json(
        done,
        {
            "schema": METHOD_COMPLETE_SCHEMA,
            "delivery": "delivery_20260716",
            "seed": plan.seed,
            "model_key": job.model_key,
            "protocol": job.protocol,
            "method": method.name,
            "repr_key": config.repr_key,
            "conflict_supervision_fraction": fraction,
            "training_config": str(method.training_config),
            "training_config_sha256": method.training_config_sha256,
            "full_relation_dataset": str(job.relation_dataset),
            "full_relation_dataset_sha256": job.relation_dataset_sha256,
            "training_relation_dataset": str(filtered_dataset),
            "training_relation_dataset_sha256": _sha256(filtered_dataset),
            "retained_conflict_group_ids_sha256": subset["retained_conflict_group_ids_sha256"],
            "best_checkpoint": str(training.best_checkpoint_path),
            "best_checkpoint_sha256": _sha256(training.best_checkpoint_path),
            "training_metrics": str(training.metrics_path),
            "training_metrics_sha256": _sha256(training.metrics_path),
            "frozen_summary": str(frozen_summary),
            "frozen_summary_sha256": _sha256(frozen_summary),
            "official_manifest": str(official_manifest),
            "official_manifest_sha256": _sha256(official_manifest),
            "official_ac_metrics": str(metrics_path),
            "official_ac_metrics_sha256": _sha256(metrics_path),
            "conflict_probe_manifest": str(probe_manifest),
            "conflict_probe_manifest_sha256": _sha256(probe_manifest),
            "conflict_probe_sample_ids_sha256": _sample_ids_sha256(probe_ids),
            "conflict_probe_sample_count": len(probe_ids),
            "probe_splits": list(PROBE_SPLITS),
            "misread_labels_used_for_encoder_training": False,
        },
    )
    return marker, evaluation, probe_ids


def _validate_method_completion(
    *,
    marker_path: Path,
    plan: BudgetPlan,
    job: BudgetJob,
    method: BudgetMethod,
    fraction: float,
    filtered_dataset: Path,
) -> tuple[Path, dict[str, Any], set[str]]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "schema": METHOD_COMPLETE_SCHEMA,
        "seed": plan.seed,
        "model_key": job.model_key,
        "method": method.name,
        "repr_key": METHOD_REPR_KEYS[method.name],
        "conflict_supervision_fraction": fraction,
        "training_config_sha256": method.training_config_sha256,
        "full_relation_dataset_sha256": job.relation_dataset_sha256,
        "training_relation_dataset_sha256": _sha256(filtered_dataset),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise BudgetPlanError(f"stale budget completion identity: {marker_path}")
    for path_field, sha_field in (
        ("best_checkpoint", "best_checkpoint_sha256"),
        ("training_metrics", "training_metrics_sha256"),
        ("frozen_summary", "frozen_summary_sha256"),
        ("official_manifest", "official_manifest_sha256"),
        ("official_ac_metrics", "official_ac_metrics_sha256"),
        ("conflict_probe_manifest", "conflict_probe_manifest_sha256"),
    ):
        artifact = Path(str(marker.get(path_field, "")))
        if not artifact.is_file() or _sha256(artifact) != marker.get(sha_field):
            raise BudgetPlanError(f"stale budget completion artifact: {artifact}")
    metrics = json.loads(Path(marker["official_ac_metrics"]).read_text(encoding="utf-8"))
    probe_rows = _read_jsonl(Path(marker["conflict_probe_manifest"]))
    probe_ids = _validate_probe_rows(probe_rows, job)
    if _sample_ids_sha256(probe_ids) != marker["conflict_probe_sample_ids_sha256"]:
        raise BudgetPlanError("probe sample identity drift in completion marker")
    return marker_path, metrics, probe_ids


def _materialize_fraction_dataset(
    *,
    source_rows: list[dict[str, Any]],
    source_path: Path,
    source_sha256: str,
    output_root: Path,
    fraction: float,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    dataset_path = output_root / "relation_dataset.jsonl"
    summary_path = output_root / "subset_provenance.json"
    filtered, subset = retained_conflict_rows(source_rows, fraction=fraction, seed=seed)
    if dataset_path.is_file() or summary_path.is_file():
        if not dataset_path.is_file() or not summary_path.is_file():
            raise BudgetPlanError("partial budget subset must be removed explicitly")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "source_relation_dataset_sha256": source_sha256,
            "fraction": fraction,
            "seed": seed,
            "retained_conflict_group_ids_sha256": subset["retained_conflict_group_ids_sha256"],
            "training_relation_dataset_sha256": _sha256(dataset_path),
        }
        if any(existing.get(key) != value for key, value in expected.items()):
            raise BudgetPlanError("existing budget subset provenance is stale")
        return dataset_path, subset
    write_jsonl(dataset_path, filtered)
    write_json(
        summary_path,
        {
            "schema": "mprisk_conflict_supervision_subset_v1",
            "source_relation_dataset": str(source_path),
            "source_relation_dataset_sha256": source_sha256,
            "training_relation_dataset": str(dataset_path),
            "training_relation_dataset_sha256": _sha256(dataset_path),
            "fraction": fraction,
            "seed": seed,
            "retained_row_count": len(filtered),
            **subset,
        },
    )
    return dataset_path, subset


def _validate_full_relation_rows(rows: list[dict[str, Any]], job: BudgetJob) -> None:
    if not rows:
        raise BudgetPlanError("full relation dataset is empty")
    if {str(row.get("model_key")) for row in rows} != {job.model_key}:
        raise BudgetPlanError("relation dataset model identity drift")
    if {str(row.get("protocol", "")).lower() for row in rows} != {job.protocol}:
        raise BudgetPlanError("relation dataset protocol identity drift")
    registered = {
        "relation_train",
        "relation_val",
        "aligned_calibration",
        "official_test",
    }
    if {str(row.get("representation_split")) for row in rows} != registered:
        raise BudgetPlanError("relation dataset split registry drift")
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(str(row["split_group_id"]), set()).add(
            str(row["representation_split"])
        )
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise BudgetPlanError("split_group_id leaks across representation splits")


def _validate_probe_rows(rows: list[dict[str, Any]], job: BudgetJob) -> set[str]:
    if not rows:
        raise BudgetPlanError("frozen Conflict probe manifest is empty")
    if {str(row.get("model_key")) for row in rows} != {job.model_key}:
        raise BudgetPlanError("probe model identity drift")
    if {str(row.get("sample_type")) for row in rows} != {"Conflict"}:
        raise BudgetPlanError("probe manifest contains non-Conflict samples")
    if {str(row.get("representation_split")) for row in rows} != set(PROBE_SPLITS):
        raise BudgetPlanError("probe manifest does not contain the fixed three splits")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise BudgetPlanError("probe manifest has duplicate sample IDs")
    return set(sample_ids)


def _pinned_file(root: Path, spec: Any, label: str) -> Path:
    _require_exact_keys(spec, {"path", "sha256"}, label)
    path = _resolve(root, spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(spec["sha256"])
    if len(expected) != 64 or _sha256(path) != expected:
        raise BudgetPlanError(f"{label} SHA drift: {path}")
    return path


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise BudgetPlanError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _sample_ids_sha256(sample_ids: set[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(sample_ids), separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        tuple(rows[0])
        if rows
        else (
            "model_key",
            "protocol",
            "method",
            "repr_key",
            "conflict_supervision_fraction",
            "retained_conflict_train_groups",
            "available_conflict_train_groups",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "auprc",
            "misread_probe_status",
        )
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise BudgetPlanError(f"{label} keys differ: expected={sorted(expected)}, actual={actual}")
