"""Strict queue for Misread probes over completed Conflict-supervision budgets."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mprisk.evaluation.misread_probe import run_conflict_misread_probe

CONFIG_SCHEMA = "mprisk_misread_budget_queue_config_v1"
FORMAL_ROOT_SCHEMA = "mprisk_formal_misread_labels_root_v1"
FORMAL_LABEL_SCHEMA = "mprisk_imported_misread_label_v1"
FRACTION_COMPLETE_SCHEMA = "mprisk_conflict_supervision_budget_fraction_complete_v1"
METHOD_COMPLETE_SCHEMA = "mprisk_conflict_supervision_budget_method_complete_v1"
PROBE_RUN_SCHEMA = "mprisk_conflict_misread_probe_run_v1"
FRACTION_PROBE_COMPLETE_SCHEMA = "mprisk_misread_budget_fraction_probe_complete_v1"
QUEUE_COMPLETE_SCHEMA = "mprisk_misread_budget_queue_complete_v1"

FRACTIONS = (0.10, 0.25, 0.50, 1.00)
METHODS = ("single_point", "trajectory_mlp", "tme")
PROBE_SPLITS = ("relation_train", "relation_val", "official_test")
LABELS = {"NON_MISREAD", "MISREAD"}
METHOD_CONTRACTS = {
    "single_point": ("single_point_binary_v1", "penultimate_feature"),
    "trajectory_mlp": ("trajectory_mlp_binary_v1", "penultimate_feature"),
    "tme": ("tme_proxy_anchor_v1", "sample_relation_feature"),
}
MODEL_KEYS = {"qwen3_vl_8b", "internvl3_5_8b", "qwen2_5_omni_7b"}

FRACTION_MARKER_FIELDS = {
    "schema",
    "model_key",
    "fraction",
    "full_relation_dataset_sha256",
    "training_relation_dataset_sha256",
    "retained_conflict_group_ids_sha256",
    "full_conflict_probe_sample_ids_sha256",
    "full_conflict_probe_sample_count",
    "method_markers",
    "misread_labels_used_for_encoder_training",
}
METHOD_MARKER_FIELDS = {
    "schema",
    "delivery",
    "seed",
    "model_key",
    "protocol",
    "method",
    "repr_key",
    "conflict_supervision_fraction",
    "training_config",
    "training_config_sha256",
    "full_relation_dataset",
    "full_relation_dataset_sha256",
    "training_relation_dataset",
    "training_relation_dataset_sha256",
    "retained_conflict_group_ids_sha256",
    "best_checkpoint",
    "best_checkpoint_sha256",
    "training_metrics",
    "training_metrics_sha256",
    "frozen_summary",
    "frozen_summary_sha256",
    "official_manifest",
    "official_manifest_sha256",
    "official_ac_metrics",
    "official_ac_metrics_sha256",
    "conflict_probe_manifest",
    "conflict_probe_manifest_sha256",
    "conflict_probe_sample_ids_sha256",
    "conflict_probe_sample_count",
    "probe_splits",
    "misread_labels_used_for_encoder_training",
}
METHOD_REFERENCES = (
    ("training_config", "training_config_sha256"),
    ("full_relation_dataset", "full_relation_dataset_sha256"),
    ("training_relation_dataset", "training_relation_dataset_sha256"),
    ("best_checkpoint", "best_checkpoint_sha256"),
    ("training_metrics", "training_metrics_sha256"),
    ("frozen_summary", "frozen_summary_sha256"),
    ("official_manifest", "official_manifest_sha256"),
    ("official_ac_metrics", "official_ac_metrics_sha256"),
    ("conflict_probe_manifest", "conflict_probe_manifest_sha256"),
)
PROBE_MARKER_FIELDS = {
    "schema",
    "status",
    "run_id",
    "model_key",
    "protocol",
    "prompt_set_key",
    "task",
    "positive_class",
    "config",
    "config_sha256",
    "formal_label_root",
    "labels",
    "labels_sha256",
    "eligible_labels",
    "eligible_labels_sha256",
    "excluded_label_counts",
    "sample_ids_sha256",
    "split_assignment_key",
    "split_assignment_sha256",
    "representation_policy",
    "split_policy",
    "architecture",
    "training_budget",
    "representations",
}
PROBE_ARTIFACTS = {
    "checkpoint",
    "metrics",
    "pr_curve",
    "predictions",
    "provenance",
    "train_log",
}
CSV_FIELDS = (
    "model_key",
    "protocol",
    "fraction",
    "representation",
    "eligible_sample_ids_sha256",
    "official_test_sample_ids_sha256",
    "official_test_sample_count",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "ap",
)


class MisreadBudgetQueueError(ValueError):
    """Raised when queue inputs or resumable outputs fail their registered contract."""


class PendingFractionsError(MisreadBudgetQueueError):
    """Raised by a one-pass queue when source fractions are still absent."""

    def __init__(self, pending: list[str]) -> None:
        self.pending = tuple(pending)
        super().__init__(f"pending FRACTION_COMPLETE markers: {', '.join(self.pending)}")


@dataclass(frozen=True)
class QueueModel:
    model_key: str
    protocol: str
    prompt_set_key: str


@dataclass(frozen=True)
class QueuePlan:
    path: Path
    delivery: str
    seed: int
    fractions: tuple[float, ...]
    budget_root: Path
    formal_label_root: Path
    output_root: Path
    lock_path: Path
    poll_seconds: float
    models: tuple[QueueModel, ...]
    training: dict[str, Any]


@dataclass(frozen=True)
class LabelSnapshot:
    model_key: str
    labels_path: Path
    labels_sha256: str
    complete_path: Path
    complete_sha256: str
    eligible_rows_sha256: str
    eligible_sample_ids_sha256: str
    official_test_sample_ids_sha256: str
    official_test_sample_count: int


@dataclass(frozen=True)
class RepresentationSource:
    name: str
    path: Path
    sha256: str
    repr_key: str
    feature_field: str
    expected_feature_dim: int


@dataclass(frozen=True)
class FractionSource:
    marker_path: Path
    marker_sha256: str
    model_key: str
    protocol: str
    fraction: float
    representations: tuple[RepresentationSource, ...]
    split_assignment_key: str
    split_assignment_sha256: str
    full_conflict_sample_ids_sha256: str


@dataclass(frozen=True)
class FractionProbeResult:
    model_key: str
    protocol: str
    fraction: float
    source_marker_path: Path
    source_marker_sha256: str
    probe_config_path: Path
    probe_config_sha256: str
    probe_marker_path: Path
    probe_marker_sha256: str
    eligible_sample_ids_sha256: str
    official_test_sample_ids_sha256: str
    official_test_sample_count: int
    metric_rows: tuple[dict[str, Any], ...]


def load_misread_budget_queue_config(path: str | Path) -> QueuePlan:
    """Load the delivery-bound queue configuration and reject contract drift."""

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
            "budget_root",
            "formal_label_root",
            "output_root",
            "lock_path",
            "poll_seconds",
            "models",
            "training",
            "representation_policy",
            "misread_labels_used_for_encoder_training",
        },
        "queue config",
    )
    if payload["schema"] != CONFIG_SCHEMA or payload["status"] != "ready":
        raise MisreadBudgetQueueError("queue config must use the registered ready schema")
    if payload["delivery"] != "delivery_20260716":
        raise MisreadBudgetQueueError("queue config must be bound to delivery_20260716")
    if payload["representation_policy"] != "frozen_no_encoder_gradients":
        raise MisreadBudgetQueueError("queue must use frozen representations")
    if payload["misread_labels_used_for_encoder_training"] is not False:
        raise MisreadBudgetQueueError("Misread labels must not enter encoder training")
    fractions = tuple(float(value) for value in payload["fractions"])
    if fractions != FRACTIONS:
        raise MisreadBudgetQueueError("fractions must be exactly 0.10/0.25/0.50/1.00")
    training = payload["training"]
    _require_exact_keys(
        training,
        {
            "seed",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "hidden_dim",
            "dropout",
            "device",
        },
        "probe training budget",
    )
    if int(training["seed"]) != int(payload["seed"]):
        raise MisreadBudgetQueueError("probe and queue seeds must be identical")
    if int(training["hidden_dim"]) != 128 or float(training["dropout"]) != 0.1:
        raise MisreadBudgetQueueError("registered unified probe architecture drift")
    if str(training["device"]) != "cpu":
        raise MisreadBudgetQueueError("Misread budget probes are registered on CPU")
    for field in ("epochs", "batch_size"):
        if int(training[field]) <= 0:
            raise MisreadBudgetQueueError(f"training.{field} must be positive")
    for field in ("learning_rate", "weight_decay"):
        value = float(training[field])
        if not math.isfinite(value) or value < 0.0 or (field == "learning_rate" and value == 0.0):
            raise MisreadBudgetQueueError(f"training.{field} is invalid")

    raw_models = payload["models"]
    if not isinstance(raw_models, list) or len(raw_models) != 3:
        raise MisreadBudgetQueueError("queue requires exactly three representative models")
    models: list[QueueModel] = []
    for raw_model in raw_models:
        _require_exact_keys(
            raw_model,
            {"model_key", "protocol", "prompt_set_key"},
            "queue model",
        )
        models.append(
            QueueModel(
                model_key=_non_empty_text(raw_model["model_key"], "model_key"),
                protocol=_non_empty_text(raw_model["protocol"], "protocol").lower(),
                prompt_set_key=_non_empty_text(raw_model["prompt_set_key"], "prompt_set_key"),
            )
        )
    if {model.model_key for model in models} != MODEL_KEYS:
        raise MisreadBudgetQueueError("representative model set drift")
    if len({model.model_key for model in models}) != len(models):
        raise MisreadBudgetQueueError("duplicate queue model")
    repo_root = config_path.parents[2]
    poll_seconds = float(payload["poll_seconds"])
    if not math.isfinite(poll_seconds) or poll_seconds <= 0.0:
        raise MisreadBudgetQueueError("poll_seconds must be finite and positive")
    return QueuePlan(
        path=config_path,
        delivery=str(payload["delivery"]),
        seed=int(payload["seed"]),
        fractions=fractions,
        budget_root=_resolve(repo_root, payload["budget_root"]),
        formal_label_root=_resolve(repo_root, payload["formal_label_root"]),
        output_root=_resolve(repo_root, payload["output_root"]),
        lock_path=_resolve(repo_root, payload["lock_path"]),
        poll_seconds=poll_seconds,
        models=tuple(models),
        training=dict(training),
    )


def run_misread_budget_queue(
    config_path: str | Path,
    *,
    once: bool = False,
    poll_seconds: float | None = None,
) -> Path:
    """Monitor source fractions, run strict probes, and return the queue marker."""

    plan = load_misread_budget_queue_config(config_path)
    interval = plan.poll_seconds if poll_seconds is None else float(poll_seconds)
    if not math.isfinite(interval) or interval <= 0.0:
        raise MisreadBudgetQueueError("poll_seconds must be finite and positive")
    plan.output_root.mkdir(parents=True, exist_ok=True)
    plan.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = plan.lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise MisreadBudgetQueueError("another Misread budget queue owns the lock") from exc
    try:
        while True:
            results, pending = process_available_fractions(plan)
            if pending:
                if once:
                    raise PendingFractionsError(pending)
                time.sleep(interval)
                continue
            return finalize_misread_budget_queue(plan, results)
    finally:
        lock_handle.close()


def process_available_fractions(
    plan: QueuePlan,
    *,
    probe_runner: Callable[[str | Path], dict[str, Any]] = run_conflict_misread_probe,
) -> tuple[list[FractionProbeResult], list[str]]:
    """Process every currently ready fraction and report absent source markers."""

    snapshots = {model.model_key: audit_formal_label_root(plan, model) for model in plan.models}
    results: list[FractionProbeResult] = []
    pending: list[str] = []
    for model in plan.models:
        for fraction in plan.fractions:
            source_path = _source_fraction_path(plan, model, fraction)
            if not source_path.exists():
                pending.append(f"{model.model_key}/fraction_{fraction:.2f}")
                continue
            if not source_path.is_file():
                raise MisreadBudgetQueueError(
                    f"source fraction marker is not a file: {source_path}"
                )
            source = audit_fraction_complete(plan, model, fraction)
            result = _run_or_resume_fraction(
                plan,
                model,
                source,
                snapshots[model.model_key],
                probe_runner=probe_runner,
            )
            results.append(result)
    return results, pending


def audit_formal_label_root(plan: QueuePlan, model: QueueModel) -> LabelSnapshot:
    """Verify the current formal root and derive the exact eligible label snapshot."""

    root = plan.formal_label_root
    complete_path = root / "COMPLETE.json"
    sidecar_path = root / "COMPLETE.json.sha256"
    checksums_path = root / "artifact_checksums.json"
    labels_path = root / "labels" / f"{model.model_key}.jsonl"
    for path in (complete_path, sidecar_path, checksums_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    complete_sha = _sha256(complete_path)
    if sidecar_path.read_text(encoding="utf-8").strip().split() != [
        complete_sha,
        "COMPLETE.json",
    ]:
        raise MisreadBudgetQueueError("formal label COMPLETE sidecar mismatch")
    marker = _read_json(complete_path, "formal label COMPLETE marker")
    _require_exact_keys(
        marker,
        {
            "schema",
            "status",
            "eligible_subset_complete",
            "models",
            "counts",
            "resolved_count",
            "unresolved_count",
            "artifact_checksums_sha256",
        },
        "formal label COMPLETE marker",
    )
    if marker["schema"] != FORMAL_ROOT_SCHEMA:
        raise MisreadBudgetQueueError("formal label root schema mismatch")
    if marker["status"] not in {"complete", "partial_manual_review_required"}:
        raise MisreadBudgetQueueError("formal label root status is invalid")
    if marker["eligible_subset_complete"] is not True:
        raise MisreadBudgetQueueError("formal label eligible subset is incomplete")
    if not isinstance(marker["models"], list) or model.model_key not in marker["models"]:
        raise MisreadBudgetQueueError("formal label root is missing a queue model")
    if marker["artifact_checksums_sha256"] != _sha256(checksums_path):
        raise MisreadBudgetQueueError("formal artifact-checksum document SHA mismatch")
    checksum_doc = _read_json(checksums_path, "formal artifact checksums")
    _require_exact_keys(checksum_doc, {"schema", "artifacts"}, "formal artifact checksums")
    if checksum_doc["schema"] != "mprisk_artifact_checksums_v1":
        raise MisreadBudgetQueueError("formal artifact-checksum schema mismatch")
    relative = f"labels/{model.model_key}.jsonl"
    evidence = checksum_doc["artifacts"].get(relative)
    _require_exact_keys(evidence, {"bytes", "sha256"}, f"checksum evidence for {relative}")
    labels_sha = _sha256(labels_path)
    if evidence["sha256"] != labels_sha or evidence["bytes"] != labels_path.stat().st_size:
        raise MisreadBudgetQueueError("formal label artifact evidence mismatch")

    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    for row in _read_jsonl(labels_path):
        required = {
            "schema",
            "sample_id",
            "subject_model_key",
            "protocol",
            "split_group_id",
            "sample_type",
            "representation_split",
            "imported_label",
            "needs_manual_review",
            "blocked",
            "label_eligible",
            "probe_eligible",
        }
        if not required <= set(row):
            raise MisreadBudgetQueueError("formal label row is missing registered fields")
        if row["schema"] != FORMAL_LABEL_SCHEMA:
            raise MisreadBudgetQueueError("formal label row schema mismatch")
        sample_id = _non_empty_text(row["sample_id"], "formal labels.sample_id")
        if sample_id in seen:
            raise MisreadBudgetQueueError(f"duplicate formal label sample_id: {sample_id}")
        seen.add(sample_id)
        if row["subject_model_key"] != model.model_key:
            raise MisreadBudgetQueueError(f"formal label model drift: {sample_id}")
        if str(row["protocol"]).lower() != model.protocol:
            raise MisreadBudgetQueueError(f"formal label protocol drift: {sample_id}")
        for field in ("needs_manual_review", "blocked", "label_eligible", "probe_eligible"):
            if not isinstance(row[field], bool):
                raise MisreadBudgetQueueError(f"formal label {field} must be boolean")
        imported = row["imported_label"]
        if row["blocked"]:
            if (
                row["probe_eligible"]
                or row["label_eligible"]
                or row["needs_manual_review"]
                or imported is not None
            ):
                raise MisreadBudgetQueueError("blocked formal label has contradictory eligibility")
            continue
        if row["needs_manual_review"]:
            if row["probe_eligible"] or row["label_eligible"] or imported is not None:
                raise MisreadBudgetQueueError("manual-review row has a fabricated eligible label")
            continue
        if row["label_eligible"] != (imported in LABELS):
            raise MisreadBudgetQueueError("formal label eligibility conflicts with imported label")
        if row["probe_eligible"] and not row["label_eligible"]:
            raise MisreadBudgetQueueError("probe-eligible label is not label-eligible")
        if not row["probe_eligible"]:
            continue
        if row["sample_type"] != "Conflict" or imported not in LABELS:
            if row["sample_type"] == "Aligned" and imported in LABELS:
                continue
            raise MisreadBudgetQueueError("probe-eligible formal label is invalid")
        split = _non_empty_text(row["representation_split"], "representation_split")
        if split not in PROBE_SPLITS:
            raise MisreadBudgetQueueError("eligible label is outside the fixed probe splits")
        group = _non_empty_text(row["split_group_id"], "split_group_id")
        group_splits.setdefault(group, set()).add(split)
        eligible.append(row)
    if not eligible:
        raise MisreadBudgetQueueError("formal root has no eligible Conflict labels")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise MisreadBudgetQueueError("formal label split_group_id crosses probe splits")
    eligible.sort(key=lambda row: str(row["sample_id"]))
    sample_ids = {str(row["sample_id"]) for row in eligible}
    official_ids = {
        str(row["sample_id"]) for row in eligible if row["representation_split"] == "official_test"
    }
    if not official_ids:
        raise MisreadBudgetQueueError("eligible formal labels have no official-test samples")
    return LabelSnapshot(
        model_key=model.model_key,
        labels_path=labels_path.resolve(),
        labels_sha256=labels_sha,
        complete_path=complete_path.resolve(),
        complete_sha256=complete_sha,
        eligible_rows_sha256=_canonical_jsonl_sha256(eligible),
        eligible_sample_ids_sha256=_sample_ids_sha256(sample_ids),
        official_test_sample_ids_sha256=_sample_ids_sha256(official_ids),
        official_test_sample_count=len(official_ids),
    )


def audit_fraction_complete(
    plan: QueuePlan,
    model: QueueModel,
    fraction: float,
) -> FractionSource:
    """Verify a source fraction marker, three method markers, and every referenced SHA."""

    marker_path = _source_fraction_path(plan, model, fraction)
    marker = _read_json(marker_path, "FRACTION_COMPLETE marker")
    _require_exact_keys(marker, FRACTION_MARKER_FIELDS, "FRACTION_COMPLETE marker")
    if marker["schema"] != FRACTION_COMPLETE_SCHEMA:
        raise MisreadBudgetQueueError("FRACTION_COMPLETE schema mismatch")
    if marker["model_key"] != model.model_key or float(marker["fraction"]) != fraction:
        raise MisreadBudgetQueueError("FRACTION_COMPLETE identity mismatch")
    if marker["misread_labels_used_for_encoder_training"] is not False:
        raise MisreadBudgetQueueError("source fraction used Misread labels for encoder training")
    for field in (
        "full_relation_dataset_sha256",
        "training_relation_dataset_sha256",
        "retained_conflict_group_ids_sha256",
        "full_conflict_probe_sample_ids_sha256",
    ):
        _sha_field(marker[field], f"FRACTION_COMPLETE.{field}")
    if (
        not isinstance(marker["full_conflict_probe_sample_count"], int)
        or marker["full_conflict_probe_sample_count"] <= 0
    ):
        raise MisreadBudgetQueueError("FRACTION_COMPLETE sample count is invalid")
    method_entries = marker["method_markers"]
    if not isinstance(method_entries, dict) or set(method_entries) != set(METHODS):
        raise MisreadBudgetQueueError("FRACTION_COMPLETE requires exactly three method markers")

    representations: list[RepresentationSource] = []
    split_keys: set[str] = set()
    split_shas: set[str] = set()
    manifest_id_sets: list[set[str]] = []
    for method in METHODS:
        entry = method_entries[method]
        _require_exact_keys(entry, {"path", "sha256"}, f"{method} marker reference")
        method_path = _absolute_path(entry["path"], f"{method} marker path")
        expected_path = (marker_path.parent / method / "RUN_COMPLETE.json").resolve()
        if method_path != expected_path:
            raise MisreadBudgetQueueError(f"{method} marker path escaped its fraction root")
        expected_marker_sha = _sha_field(entry["sha256"], f"{method} marker SHA")
        if _sha256(method_path) != expected_marker_sha:
            raise MisreadBudgetQueueError(f"{method} marker SHA mismatch")
        method_marker = _read_json(method_path, f"{method} method marker")
        _require_exact_keys(method_marker, METHOD_MARKER_FIELDS, f"{method} method marker")
        expected_repr_key, feature_field = METHOD_CONTRACTS[method]
        if (
            method_marker["schema"] != METHOD_COMPLETE_SCHEMA
            or method_marker["delivery"] != plan.delivery
            or int(method_marker["seed"]) != plan.seed
            or method_marker["model_key"] != model.model_key
            or str(method_marker["protocol"]).lower() != model.protocol
            or method_marker["method"] != method
            or method_marker["repr_key"] != expected_repr_key
            or float(method_marker["conflict_supervision_fraction"]) != fraction
        ):
            raise MisreadBudgetQueueError(f"{method} method marker identity drift")
        if method_marker["misread_labels_used_for_encoder_training"] is not False:
            raise MisreadBudgetQueueError(f"{method} used Misread labels for encoder training")
        if method_marker["probe_splits"] != list(PROBE_SPLITS):
            raise MisreadBudgetQueueError(f"{method} probe split contract drift")
        if (
            method_marker["full_relation_dataset_sha256"] != marker["full_relation_dataset_sha256"]
            or method_marker["training_relation_dataset_sha256"]
            != marker["training_relation_dataset_sha256"]
            or method_marker["retained_conflict_group_ids_sha256"]
            != marker["retained_conflict_group_ids_sha256"]
            or method_marker["conflict_probe_sample_ids_sha256"]
            != marker["full_conflict_probe_sample_ids_sha256"]
            or method_marker["conflict_probe_sample_count"]
            != marker["full_conflict_probe_sample_count"]
        ):
            raise MisreadBudgetQueueError(f"{method} method marker conflicts with fraction marker")
        for path_field, sha_field in METHOD_REFERENCES:
            referenced = _absolute_path(method_marker[path_field], f"{method}.{path_field}")
            expected_sha = _sha_field(method_marker[sha_field], f"{method}.{sha_field}")
            if _sha256(referenced) != expected_sha:
                raise MisreadBudgetQueueError(
                    f"{method} referenced artifact SHA mismatch: {path_field}"
                )

        manifest_path = _absolute_path(
            method_marker["conflict_probe_manifest"], f"{method} conflict probe manifest"
        )
        ids, split_key, split_sha, feature_dim = _audit_representation_manifest(
            manifest_path,
            model=model,
            method=method,
            expected_count=int(method_marker["conflict_probe_sample_count"]),
            expected_ids_sha=str(method_marker["conflict_probe_sample_ids_sha256"]),
        )
        manifest_id_sets.append(ids)
        split_keys.add(split_key)
        split_shas.add(split_sha)
        representations.append(
            RepresentationSource(
                name=method,
                path=manifest_path,
                sha256=str(method_marker["conflict_probe_manifest_sha256"]),
                repr_key=expected_repr_key,
                feature_field=feature_field,
                expected_feature_dim=feature_dim,
            )
        )
    if len({frozenset(ids) for ids in manifest_id_sets}) != 1:
        raise MisreadBudgetQueueError("three source representations use different Conflict IDs")
    if len(split_keys) != 1 or len(split_shas) != 1:
        raise MisreadBudgetQueueError("three source representations use different split identities")
    return FractionSource(
        marker_path=marker_path.resolve(),
        marker_sha256=_sha256(marker_path),
        model_key=model.model_key,
        protocol=model.protocol,
        fraction=fraction,
        representations=tuple(representations),
        split_assignment_key=next(iter(split_keys)),
        split_assignment_sha256=next(iter(split_shas)),
        full_conflict_sample_ids_sha256=str(marker["full_conflict_probe_sample_ids_sha256"]),
    )


def derive_ready_probe_config(
    plan: QueuePlan,
    model: QueueModel,
    source: FractionSource,
    labels: LabelSnapshot,
) -> dict[str, Any]:
    """Derive the full absolute ready probe configuration from verified artifacts."""

    fraction_root = _queue_fraction_root(plan, model, source.fraction)
    return {
        "schema": "mprisk_conflict_misread_probe_config_v1",
        "status": "ready",
        "run_id": (
            f"{plan.delivery}_{model.model_key}_misread_budget_fraction_{source.fraction:.2f}_v1"
        ),
        "model_key": model.model_key,
        "protocol": model.protocol,
        "prompt_set_key": model.prompt_set_key,
        "labels": {
            "root": str(plan.formal_label_root.resolve()),
            "complete_sha256": labels.complete_sha256,
            "artifact_sha256": labels.labels_sha256,
            "expected_eligible_rows_sha256": labels.eligible_rows_sha256,
        },
        "representations": [
            {
                "name": rep.name,
                "path": str(rep.path),
                "sha256": rep.sha256,
                "repr_key": rep.repr_key,
                "feature_field": rep.feature_field,
                "expected_feature_dim": rep.expected_feature_dim,
            }
            for rep in source.representations
        ],
        "split_assignment_key": source.split_assignment_key,
        "split_assignment_sha256": source.split_assignment_sha256,
        "expected_sample_ids_sha256": labels.eligible_sample_ids_sha256,
        "training": dict(plan.training),
        "output_root": str((fraction_root / "probe").resolve()),
    }


def finalize_misread_budget_queue(
    plan: QueuePlan,
    results: list[FractionProbeResult],
) -> Path:
    """Enforce cross-fraction identity and write deterministic consolidated outputs."""

    expected_pairs = {
        (model.model_key, fraction) for model in plan.models for fraction in plan.fractions
    }
    actual_pairs = {(result.model_key, result.fraction) for result in results}
    if actual_pairs != expected_pairs or len(results) != len(expected_pairs):
        raise MisreadBudgetQueueError("cannot finalize an incomplete or duplicate queue result set")
    per_model_identity: dict[str, dict[str, str]] = {}
    for model in plan.models:
        model_results = [result for result in results if result.model_key == model.model_key]
        eligible_shas = {result.eligible_sample_ids_sha256 for result in model_results}
        official_shas = {result.official_test_sample_ids_sha256 for result in model_results}
        official_counts = {result.official_test_sample_count for result in model_results}
        if len(eligible_shas) != 1:
            raise MisreadBudgetQueueError(
                f"eligible sample IDs differ across fractions for {model.model_key}"
            )
        if len(official_shas) != 1 or len(official_counts) != 1:
            raise MisreadBudgetQueueError(
                f"official-test sample IDs differ across fractions for {model.model_key}"
            )
        per_model_identity[model.model_key] = {
            "eligible_sample_ids_sha256": next(iter(eligible_shas)),
            "official_test_sample_ids_sha256": next(iter(official_shas)),
        }
    rows = [row for result in results for row in result.metric_rows]
    expected_row_count = len(plan.models) * len(plan.fractions) * len(METHODS)
    if len(rows) != expected_row_count:
        raise MisreadBudgetQueueError("consolidated Misread metric row count is incomplete")
    rows.sort(key=lambda row: (row["model_key"], float(row["fraction"]), row["representation"]))
    plan.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = plan.output_root / "misread_budget_probe_metrics.csv"
    _materialize_exact_bytes(csv_path, _csv_bytes(rows))
    ordered_results = sorted(results, key=lambda result: (result.model_key, result.fraction))
    marker_payload = {
        "schema": QUEUE_COMPLETE_SCHEMA,
        "status": "complete",
        "delivery": plan.delivery,
        "seed": plan.seed,
        "fractions": list(plan.fractions),
        "models": [model.model_key for model in plan.models],
        "methods": list(METHODS),
        "config": str(plan.path),
        "config_sha256": _sha256(plan.path),
        "formal_label_root": str(plan.formal_label_root),
        "per_model_identity": per_model_identity,
        "fraction_probe_markers": [
            {
                "model_key": result.model_key,
                "fraction": result.fraction,
                "path": str(
                    _queue_fraction_root(
                        plan,
                        next(model for model in plan.models if model.model_key == result.model_key),
                        result.fraction,
                    )
                    / "FRACTION_PROBE_COMPLETE.json"
                ),
                "sha256": _sha256(
                    _queue_fraction_root(
                        plan,
                        next(model for model in plan.models if model.model_key == result.model_key),
                        result.fraction,
                    )
                    / "FRACTION_PROBE_COMPLETE.json"
                ),
            }
            for result in ordered_results
        ],
        "metrics_csv": str(csv_path),
        "metrics_csv_sha256": _sha256(csv_path),
        "representation_policy": "frozen_no_encoder_gradients",
        "misread_labels_used_for_encoder_training": False,
    }
    marker_path = plan.output_root / "MISREAD_BUDGET_COMPLETE.json"
    _materialize_exact_bytes(marker_path, _json_bytes(marker_payload))
    marker_sha = _sha256(marker_path)
    _materialize_exact_bytes(
        plan.output_root / "MISREAD_BUDGET_COMPLETE.json.sha256",
        f"{marker_sha}  MISREAD_BUDGET_COMPLETE.json\n".encode(),
    )
    return marker_path


def _run_or_resume_fraction(
    plan: QueuePlan,
    model: QueueModel,
    source: FractionSource,
    labels: LabelSnapshot,
    *,
    probe_runner: Callable[[str | Path], dict[str, Any]],
) -> FractionProbeResult:
    fraction_root = _queue_fraction_root(plan, model, source.fraction)
    fraction_root.mkdir(parents=True, exist_ok=True)
    config_path = (fraction_root / "probe_config.yaml").resolve()
    config_payload = derive_ready_probe_config(plan, model, source, labels)
    config_bytes = yaml.safe_dump(
        config_payload,
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")
    _materialize_exact_bytes(config_path, config_bytes)
    probe_root = Path(config_payload["output_root"])
    probe_marker = probe_root / "RUN_COMPLETE.json"
    if probe_marker.exists():
        result = _audit_probe_run(plan, model, source, labels, config_path)
    else:
        if probe_root.exists() and any(probe_root.iterdir()):
            raise MisreadBudgetQueueError(f"partial probe output cannot be resumed: {probe_root}")
        probe_runner(config_path)
        result = _audit_probe_run(plan, model, source, labels, config_path)
    fraction_payload = _fraction_probe_marker_payload(result)
    completion_path = fraction_root / "FRACTION_PROBE_COMPLETE.json"
    _materialize_exact_bytes(completion_path, _json_bytes(fraction_payload))
    completion_sha = _sha256(completion_path)
    _materialize_exact_bytes(
        fraction_root / "FRACTION_PROBE_COMPLETE.json.sha256",
        f"{completion_sha}  FRACTION_PROBE_COMPLETE.json\n".encode(),
    )
    return result


def _audit_probe_run(
    plan: QueuePlan,
    model: QueueModel,
    source: FractionSource,
    labels: LabelSnapshot,
    config_path: Path,
) -> FractionProbeResult:
    probe_root = _queue_fraction_root(plan, model, source.fraction) / "probe"
    marker_path = probe_root / "RUN_COMPLETE.json"
    sidecar_path = probe_root / "RUN_COMPLETE.sha256"
    if not marker_path.is_file() or not sidecar_path.is_file():
        raise MisreadBudgetQueueError("probe runner did not produce checksummed completion")
    marker_sha = _sha256(marker_path)
    if sidecar_path.read_text(encoding="utf-8").strip().split() != [marker_sha]:
        raise MisreadBudgetQueueError("probe RUN_COMPLETE sidecar mismatch")
    marker = _read_json(marker_path, "probe RUN_COMPLETE marker")
    _require_exact_keys(marker, PROBE_MARKER_FIELDS, "probe RUN_COMPLETE marker")
    if (
        marker["schema"] != PROBE_RUN_SCHEMA
        or marker["status"] != "complete"
        or marker["model_key"] != model.model_key
        or str(marker["protocol"]).lower() != model.protocol
        or marker["prompt_set_key"] != model.prompt_set_key
        or marker["task"] != "Conflict_only_Misread_vs_Non-misread"
        or marker["positive_class"] != "Misread"
        or marker["representation_policy"] != "frozen_no_encoder_gradients"
    ):
        raise MisreadBudgetQueueError("probe RUN_COMPLETE identity or policy drift")
    if _absolute_path(marker["config"], "probe config path") != config_path:
        raise MisreadBudgetQueueError("probe RUN_COMPLETE config path mismatch")
    config_sha = _sha256(config_path)
    if marker["config_sha256"] != config_sha:
        raise MisreadBudgetQueueError("probe RUN_COMPLETE config SHA mismatch")
    if marker["sample_ids_sha256"] != labels.eligible_sample_ids_sha256:
        raise MisreadBudgetQueueError("probe eligible sample-ID SHA mismatch")
    if marker["eligible_labels_sha256"] != labels.eligible_rows_sha256:
        raise MisreadBudgetQueueError("probe eligible-label artifact SHA mismatch")
    if marker["labels_sha256"] != labels.labels_sha256:
        raise MisreadBudgetQueueError("probe full-label SHA mismatch")
    if _absolute_path(marker["labels"], "probe labels path") != labels.labels_path:
        raise MisreadBudgetQueueError("probe full-label path mismatch")
    eligible_path = _absolute_path(marker["eligible_labels"], "eligible labels path")
    if eligible_path != (probe_root / "eligible_labels.jsonl").resolve():
        raise MisreadBudgetQueueError("probe eligible-label path mismatch")
    if _sha256(eligible_path) != labels.eligible_rows_sha256:
        raise MisreadBudgetQueueError("probe eligible-label artifact drift")
    formal_root = marker["formal_label_root"]
    _require_exact_keys(
        formal_root,
        {"path", "complete_path", "complete_sha256", "status", "eligible_subset_complete"},
        "probe formal label root",
    )
    if (
        _absolute_path(formal_root["path"], "probe formal root") != plan.formal_label_root.resolve()
        or _absolute_path(formal_root["complete_path"], "probe formal COMPLETE")
        != labels.complete_path
        or formal_root["complete_sha256"] != labels.complete_sha256
        or formal_root["eligible_subset_complete"] is not True
    ):
        raise MisreadBudgetQueueError("probe formal label snapshot mismatch")
    if (
        marker["split_assignment_key"] != source.split_assignment_key
        or marker["split_assignment_sha256"] != source.split_assignment_sha256
    ):
        raise MisreadBudgetQueueError("probe split identity mismatch")
    if marker["training_budget"] != plan.training:
        raise MisreadBudgetQueueError("probe training budget drift")
    if marker["split_policy"] != {
        "train": "relation_train",
        "validation": "relation_val",
        "test": "official_test",
        "unit": "split_group_id",
    }:
        raise MisreadBudgetQueueError("probe split policy drift")

    reps = marker["representations"]
    if not isinstance(reps, list) or [row.get("representation") for row in reps] != list(METHODS):
        raise MisreadBudgetQueueError("probe requires exactly three registered representations")
    metric_rows: list[dict[str, Any]] = []
    official_sets: list[set[str]] = []
    for source_rep, result_rep in zip(source.representations, reps, strict=True):
        _require_exact_keys(
            result_rep,
            {"representation", "repr_key", "feature_dim", "metrics", "artifacts"},
            f"probe result {source_rep.name}",
        )
        if (
            result_rep["representation"] != source_rep.name
            or result_rep["repr_key"] != source_rep.repr_key
            or int(result_rep["feature_dim"]) != source_rep.expected_feature_dim
        ):
            raise MisreadBudgetQueueError(f"probe representation drift for {source_rep.name}")
        artifacts = result_rep["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != PROBE_ARTIFACTS:
            raise MisreadBudgetQueueError(f"probe artifact set drift for {source_rep.name}")
        artifact_paths: dict[str, Path] = {}
        for artifact_name in PROBE_ARTIFACTS:
            evidence = artifacts[artifact_name]
            _require_exact_keys(evidence, {"path", "sha256"}, "probe artifact evidence")
            path = _absolute_path(evidence["path"], f"probe {artifact_name}")
            if path.parent != (probe_root / source_rep.name).resolve():
                raise MisreadBudgetQueueError("probe artifact escaped its representation root")
            if _sha256(path) != _sha_field(evidence["sha256"], "probe artifact SHA"):
                raise MisreadBudgetQueueError(f"probe artifact SHA drift: {artifact_name}")
            artifact_paths[artifact_name] = path
        metrics = _read_json(artifact_paths["metrics"], "probe metrics")
        if metrics != result_rep["metrics"]:
            raise MisreadBudgetQueueError("inline and artifact probe metrics differ")
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1", "ap"):
            metric = float(metrics[metric_name])
            if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
                raise MisreadBudgetQueueError(f"invalid probe metric: {metric_name}")
        prediction_rows = _read_jsonl(artifact_paths["predictions"])
        official_ids: set[str] = set()
        for prediction in prediction_rows:
            if prediction.get("representation") != source_rep.name:
                raise MisreadBudgetQueueError("probe prediction representation drift")
            if prediction.get("representation_split") != "official_test":
                raise MisreadBudgetQueueError("probe prediction is outside official_test")
            sample_id = _non_empty_text(prediction.get("sample_id"), "prediction.sample_id")
            if sample_id in official_ids:
                raise MisreadBudgetQueueError("duplicate official-test prediction sample_id")
            official_ids.add(sample_id)
        if _sample_ids_sha256(official_ids) != labels.official_test_sample_ids_sha256:
            raise MisreadBudgetQueueError("probe official-test sample-ID SHA mismatch")
        if len(official_ids) != labels.official_test_sample_count:
            raise MisreadBudgetQueueError("probe official-test sample count mismatch")
        official_sets.append(official_ids)
        metric_rows.append(
            {
                "model_key": model.model_key,
                "protocol": model.protocol,
                "fraction": f"{source.fraction:.2f}",
                "representation": source_rep.name,
                "eligible_sample_ids_sha256": labels.eligible_sample_ids_sha256,
                "official_test_sample_ids_sha256": labels.official_test_sample_ids_sha256,
                "official_test_sample_count": len(official_ids),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "ap": float(metrics["ap"]),
            }
        )
    if len({frozenset(ids) for ids in official_sets}) != 1:
        raise MisreadBudgetQueueError("probe representations use different official-test IDs")
    return FractionProbeResult(
        model_key=model.model_key,
        protocol=model.protocol,
        fraction=source.fraction,
        source_marker_path=source.marker_path,
        source_marker_sha256=source.marker_sha256,
        probe_config_path=config_path,
        probe_config_sha256=config_sha,
        probe_marker_path=marker_path.resolve(),
        probe_marker_sha256=marker_sha,
        eligible_sample_ids_sha256=labels.eligible_sample_ids_sha256,
        official_test_sample_ids_sha256=labels.official_test_sample_ids_sha256,
        official_test_sample_count=labels.official_test_sample_count,
        metric_rows=tuple(metric_rows),
    )


def _audit_representation_manifest(
    path: Path,
    *,
    model: QueueModel,
    method: str,
    expected_count: int,
    expected_ids_sha: str,
) -> tuple[set[str], str, str, int]:
    rows = _read_jsonl(path)
    if not rows:
        raise MisreadBudgetQueueError(f"empty Conflict probe manifest: {path}")
    repr_key, feature_field = METHOD_CONTRACTS[method]
    ids: set[str] = set()
    split_keys: set[str] = set()
    split_shas: set[str] = set()
    feature_dims: set[int] = set()
    for row in rows:
        _reject_misread_fields(row)
        sample_id = _non_empty_text(row.get("sample_id"), f"{method}.sample_id")
        if sample_id in ids:
            raise MisreadBudgetQueueError(f"duplicate {method} sample_id: {sample_id}")
        ids.add(sample_id)
        if (
            row.get("model_key") != model.model_key
            or str(row.get("protocol", "")).lower() != model.protocol
            or row.get("prompt_set_key") != model.prompt_set_key
            or row.get("repr_key") != repr_key
            or row.get("sample_type") != "Conflict"
            or row.get("representation_split") not in PROBE_SPLITS
        ):
            raise MisreadBudgetQueueError(f"{method} representation identity drift")
        vector = row.get(feature_field)
        if not isinstance(vector, list) or not vector:
            raise MisreadBudgetQueueError(f"{method} feature vector is empty or invalid")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for value in vector
        ):
            raise MisreadBudgetQueueError(f"{method} feature vector is not finite")
        feature_dims.add(len(vector))
        split_keys.add(_non_empty_text(row.get("split_assignment_key"), "split_assignment_key"))
        split_shas.add(_sha_field(row.get("split_assignment_sha256"), "split_assignment_sha256"))
    if len(ids) != expected_count or _sample_ids_sha256(ids) != expected_ids_sha:
        raise MisreadBudgetQueueError(f"{method} Conflict sample identity mismatch")
    if len(split_keys) != 1 or len(split_shas) != 1 or len(feature_dims) != 1:
        raise MisreadBudgetQueueError(f"{method} contains mixed split identities")
    return ids, next(iter(split_keys)), next(iter(split_shas)), next(iter(feature_dims))


def _fraction_probe_marker_payload(result: FractionProbeResult) -> dict[str, Any]:
    return {
        "schema": FRACTION_PROBE_COMPLETE_SCHEMA,
        "status": "complete",
        "model_key": result.model_key,
        "protocol": result.protocol,
        "fraction": result.fraction,
        "source_fraction_marker": {
            "path": str(result.source_marker_path),
            "sha256": result.source_marker_sha256,
        },
        "probe_config": {
            "path": str(result.probe_config_path),
            "sha256": result.probe_config_sha256,
        },
        "probe_run_complete": {
            "path": str(result.probe_marker_path),
            "sha256": result.probe_marker_sha256,
        },
        "eligible_sample_ids_sha256": result.eligible_sample_ids_sha256,
        "official_test_sample_ids_sha256": result.official_test_sample_ids_sha256,
        "official_test_sample_count": result.official_test_sample_count,
        "representation_policy": "frozen_no_encoder_gradients",
        "misread_labels_used_for_encoder_training": False,
    }


def _source_fraction_path(plan: QueuePlan, model: QueueModel, fraction: float) -> Path:
    return (
        plan.budget_root / model.model_key / f"fraction_{fraction:.2f}" / "FRACTION_COMPLETE.json"
    )


def _queue_fraction_root(plan: QueuePlan, model: QueueModel, fraction: float) -> Path:
    return plan.output_root / model.model_key / f"fraction_{fraction:.2f}"


def _reject_misread_fields(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if "misread" in key_text.lower():
                raise MisreadBudgetQueueError(
                    f"Misread leakage in frozen representation: {path}.{key_text}"
                )
            _reject_misread_fields(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_misread_fields(child, path=f"{path}[{index}]")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MisreadBudgetQueueError(f"malformed {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise MisreadBudgetQueueError(f"{label} must be a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise MisreadBudgetQueueError(f"blank JSONL row: {path}:{line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise MisreadBudgetQueueError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MisreadBudgetQueueError(f"malformed JSONL artifact: {path}") from exc
    return rows


def _materialize_exact_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise MisreadBudgetQueueError(f"resumable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _canonical_jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    content = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _sample_ids_sha256(sample_ids: set[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(sample_ids), separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_path(value: Any, field: str) -> Path:
    path = Path(_non_empty_text(value, field)).expanduser()
    if not path.is_absolute():
        raise MisreadBudgetQueueError(f"{field} must be absolute")
    resolved = path.resolve()
    if not resolved.is_file() and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _resolve(root: Path, value: Any) -> Path:
    path = Path(_non_empty_text(value, "path")).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MisreadBudgetQueueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha_field(value: Any, field: str) -> str:
    text = _non_empty_text(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise MisreadBudgetQueueError(f"{field} must be a lowercase SHA-256")
    return text


def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise MisreadBudgetQueueError(
            f"{label} keys mismatch: expected {sorted(expected)}, got {actual}"
        )
