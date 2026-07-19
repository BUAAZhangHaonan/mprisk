"""Conflict-only Misread probe with frozen, identity-locked representations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from mprisk.utils.io import write_json

PENDING_SCHEMA = "mprisk_conflict_only_misread_probe_v1"
CONFIG_SCHEMA = "mprisk_conflict_misread_probe_config_v1"
FORMAL_LABEL_ROOT_SCHEMA = "mprisk_formal_misread_labels_root_v1"
FORMAL_LABEL_SCHEMA = "mprisk_imported_misread_label_v1"
RUN_SCHEMA = "mprisk_conflict_misread_probe_run_v1"
REPRESENTATION_NAMES = ("single_point", "trajectory_mlp", "tme")
REPRESENTATION_CONTRACTS = {
    "single_point": ("single_point_binary_v1", "penultimate_feature"),
    "trajectory_mlp": ("trajectory_mlp_binary_v1", "penultimate_feature"),
    "tme": ("tme_proxy_anchor_v1", "sample_relation_feature"),
}
PROBE_SPLITS = ("relation_train", "relation_val", "official_test")
LABEL_TO_ID = {"NON_MISREAD": 0, "MISREAD": 1}
FORMAL_LABEL_REQUIRED_FIELDS = frozenset(
    {
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
)


@dataclass(frozen=True)
class FormalLabelSpec:
    root: Path
    complete_path: Path
    complete_sha256: str
    labels_path: Path
    sha256: str
    expected_eligible_rows_sha256: str


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    path: Path
    sha256: str
    repr_key: str
    feature_field: str
    expected_feature_dim: int


@dataclass(frozen=True)
class TrainingBudget:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    device: str


@dataclass(frozen=True)
class ProbeConfig:
    config_path: Path
    repo_root: Path
    status: str
    run_id: str
    model_key: str
    protocol: str
    prompt_set_key: str
    labels: FormalLabelSpec
    representations: tuple[RepresentationSpec, ...]
    split_assignment_key: str
    split_assignment_sha256: str
    expected_sample_ids_sha256: str
    training: TrainingBudget
    output_root: Path


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    split_group_id: str
    representation_split: str
    label: str
    label_id: int


@dataclass(frozen=True)
class ProbeDataset:
    samples: tuple[ProbeSample, ...]
    features: dict[str, dict[str, list[float]]]
    feature_dims: dict[str, int]
    sample_ids_sha256: str
    eligible_rows: tuple[dict[str, Any], ...]
    eligible_rows_sha256: str
    excluded_label_counts: dict[str, int]
    label_root_status: str


class UnifiedMisreadMLP(nn.Module):
    """The single registered architecture used for every representation."""

    def __init__(self, input_dim: int, *, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def write_pending_conflict_misread_probe(output_dir: str | Path) -> Path:
    """Keep the explicit Pending artifact used before verified labels exist."""

    payload: dict[str, Any] = {
        "schema": PENDING_SCHEMA,
        "status": "Pending Misread annotations",
        "labels_available": False,
        "eligible_sample_type": "Conflict",
        "excluded_sample_type": "Aligned",
        "representation_policy": "frozen_no_encoder_gradients",
        "split_policy": "group_disjoint_within_conflict_only",
        "probe_architecture": {
            "shared_across_representations": True,
            "layers": ["Linear(input_dim,128)", "GELU", "Dropout(0.1)", "Linear(128,2)"],
            "target": "Misread_vs_Non-misread",
        },
        "required_future_fields": [
            "sample_id",
            "split_group_id",
            "sample_type=Conflict",
            "misread_label",
        ],
        "metrics_when_available": ["Accuracy", "Balanced Accuracy", "Macro-F1", "AP"],
        "generated_labels": 0,
        "pseudo_labels": 0,
        "training_started": False,
    }
    return write_json(Path(output_dir) / "PENDING.json", payload)


def load_probe_config(path: str | Path) -> ProbeConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "run_id",
            "model_key",
            "protocol",
            "prompt_set_key",
            "labels",
            "representations",
            "split_assignment_key",
            "split_assignment_sha256",
            "expected_sample_ids_sha256",
            "training",
            "output_root",
        },
        "probe config",
    )
    if payload["schema"] != CONFIG_SCHEMA:
        raise ValueError(f"probe config schema must be {CONFIG_SCHEMA}")
    if payload["status"] not in {"pending", "ready"}:
        raise ValueError("probe config status must be pending or ready")
    run_id = _non_empty_text(payload["run_id"], "run_id")
    model_key = _non_empty_text(payload["model_key"], "model_key")
    protocol = _non_empty_text(payload["protocol"], "protocol").lower()
    prompt_set_key = _non_empty_text(payload["prompt_set_key"], "prompt_set_key")
    repo_root = config_path.parents[2]

    label_row = payload["labels"]
    _require_exact_keys(
        label_row,
        {
            "root",
            "complete_sha256",
            "artifact_sha256",
            "expected_eligible_rows_sha256",
        },
        "formal labels artifact",
    )
    label_root = _resolve(repo_root, label_row["root"])
    labels = FormalLabelSpec(
        root=label_root,
        complete_path=label_root / "COMPLETE.json",
        complete_sha256=_sha_field(label_row["complete_sha256"], "labels.complete_sha256"),
        labels_path=label_root / "labels" / f"{model_key}.jsonl",
        sha256=_sha_field(label_row["artifact_sha256"], "labels.artifact_sha256"),
        expected_eligible_rows_sha256=_sha_field(
            label_row["expected_eligible_rows_sha256"],
            "labels.expected_eligible_rows_sha256",
        ),
    )

    raw_representations = payload["representations"]
    if not isinstance(raw_representations, list) or not 1 <= len(raw_representations) <= len(
        REPRESENTATION_NAMES
    ):
        raise ValueError("probe config requires one to three frozen representations")
    representations: list[RepresentationSpec] = []
    for row in raw_representations:
        _require_exact_keys(
            row,
            {
                "name",
                "path",
                "sha256",
                "repr_key",
                "feature_field",
                "expected_feature_dim",
            },
            "representation artifact",
        )
        name = _non_empty_text(row["name"], "representations.name")
        if name not in REPRESENTATION_CONTRACTS:
            raise ValueError(f"unknown registered representation: {name}")
        expected_repr_key, expected_feature = REPRESENTATION_CONTRACTS[name]
        if row["repr_key"] != expected_repr_key or row["feature_field"] != expected_feature:
            raise ValueError(f"representation contract drift for {name}")
        expected_dim = int(row["expected_feature_dim"])
        if expected_dim <= 0:
            raise ValueError("expected_feature_dim must be positive")
        representations.append(
            RepresentationSpec(
                name=name,
                path=_resolve(repo_root, row["path"]),
                sha256=_sha_field(row["sha256"], f"representations.{name}.sha256"),
                repr_key=expected_repr_key,
                feature_field=expected_feature,
                expected_feature_dim=expected_dim,
            )
        )
    representation_names = tuple(spec.name for spec in representations)
    if len(set(representation_names)) != len(representation_names):
        raise ValueError("probe config representation names must be unique")
    registered_order = tuple(name for name in REPRESENTATION_NAMES if name in representation_names)
    if representation_names != registered_order:
        raise ValueError("probe config representations must use registered order")

    raw_training = payload["training"]
    _require_exact_keys(
        raw_training,
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
        "training budget",
    )
    training = TrainingBudget(
        seed=int(raw_training["seed"]),
        epochs=int(raw_training["epochs"]),
        batch_size=int(raw_training["batch_size"]),
        learning_rate=float(raw_training["learning_rate"]),
        weight_decay=float(raw_training["weight_decay"]),
        hidden_dim=int(raw_training["hidden_dim"]),
        dropout=float(raw_training["dropout"]),
        device=_non_empty_text(raw_training["device"], "training.device"),
    )
    if training.epochs <= 0 or training.batch_size <= 0:
        raise ValueError("training epochs and batch_size must be positive")
    if training.learning_rate <= 0 or training.weight_decay < 0:
        raise ValueError("training learning_rate must be positive and weight_decay non-negative")
    if training.hidden_dim != 128 or training.dropout != 0.1:
        raise ValueError("unified probe architecture requires hidden_dim=128 and dropout=0.1")

    return ProbeConfig(
        config_path=config_path,
        repo_root=repo_root,
        status=str(payload["status"]),
        run_id=run_id,
        model_key=model_key,
        protocol=protocol,
        prompt_set_key=prompt_set_key,
        labels=labels,
        representations=tuple(representations),
        split_assignment_key=_non_empty_text(
            payload["split_assignment_key"], "split_assignment_key"
        ),
        split_assignment_sha256=_sha_field(
            payload["split_assignment_sha256"], "split_assignment_sha256"
        ),
        expected_sample_ids_sha256=_sha_field(
            payload["expected_sample_ids_sha256"], "expected_sample_ids_sha256"
        ),
        training=training,
        output_root=_resolve(repo_root, payload["output_root"]),
    )


def run_conflict_misread_probe(config_path: str | Path) -> dict[str, Any]:
    config = load_probe_config(config_path)
    if config.status != "ready":
        raise ValueError("Conflict-only Misread probe config is pending verified inputs")
    label_marker = _verify_formal_label_root(config)
    _verify_artifact(config.labels.labels_path, config.labels.sha256, "formal labels")
    for spec in config.representations:
        _verify_artifact(spec.path, spec.sha256, f"{spec.name} representation")
    dataset = _load_probe_dataset(config, label_root_status=str(label_marker["status"]))
    _prepare_empty_output_root(config.output_root)
    eligible_labels_path = _atomic_canonical_jsonl(
        config.output_root / "eligible_labels.jsonl", list(dataset.eligible_rows)
    )
    if _sha256(eligible_labels_path) != dataset.eligible_rows_sha256:
        raise RuntimeError("materialized eligible label artifact SHA-256 drift")

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        results = [
            _run_representation(
                config=config,
                dataset=dataset,
                spec=spec,
                eligible_labels_path=eligible_labels_path,
            )
            for spec in config.representations
        ]
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    marker = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "run_id": config.run_id,
        "model_key": config.model_key,
        "protocol": config.protocol,
        "prompt_set_key": config.prompt_set_key,
        "task": "Conflict_only_Misread_vs_Non-misread",
        "positive_class": "Misread",
        "config": str(config.config_path),
        "config_sha256": _sha256(config.config_path),
        "formal_label_root": {
            "path": str(config.labels.root),
            "complete_path": str(config.labels.complete_path),
            "complete_sha256": config.labels.complete_sha256,
            "status": dataset.label_root_status,
            "eligible_subset_complete": True,
        },
        "labels": str(config.labels.labels_path),
        "labels_sha256": config.labels.sha256,
        "eligible_labels": str(eligible_labels_path),
        "eligible_labels_sha256": dataset.eligible_rows_sha256,
        "excluded_label_counts": dataset.excluded_label_counts,
        "sample_ids_sha256": dataset.sample_ids_sha256,
        "split_assignment_key": config.split_assignment_key,
        "split_assignment_sha256": config.split_assignment_sha256,
        "representation_policy": "frozen_no_encoder_gradients",
        "split_policy": {
            "train": "relation_train",
            "validation": "relation_val",
            "test": "official_test",
            "unit": "split_group_id",
        },
        "architecture": _architecture_payload(config.training),
        "training_budget": _training_payload(config.training),
        "representations": results,
    }
    marker_path = _atomic_json(config.output_root / "RUN_COMPLETE.json", marker)
    marker_sha = _sha256(marker_path)
    sha_path = _atomic_text(config.output_root / "RUN_COMPLETE.sha256", f"{marker_sha}\n")
    return {
        **marker,
        "run_complete_path": str(marker_path),
        "run_complete_sha256": marker_sha,
        "run_complete_sha256_path": str(sha_path),
    }


def _load_probe_dataset(config: ProbeConfig, *, label_root_status: str) -> ProbeDataset:
    label_rows = _read_jsonl(config.labels.labels_path)
    if not label_rows:
        raise ValueError("formal labels manifest is empty")
    labels: dict[str, ProbeSample] = {}
    eligible_rows: list[dict[str, Any]] = []
    seen_label_ids: set[str] = set()
    excluded_label_counts = {
        "manual_review": 0,
        "blocked": 0,
        "not_probe_eligible": 0,
        "aligned_out_of_scope": 0,
    }
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in label_rows:
        if not FORMAL_LABEL_REQUIRED_FIELDS <= set(row):
            missing = sorted(FORMAL_LABEL_REQUIRED_FIELDS - set(row))
            raise ValueError(f"formal label row is missing fields: {missing}")
        if row["schema"] != FORMAL_LABEL_SCHEMA:
            raise ValueError("formal label row schema mismatch")
        sample_id = _non_empty_text(row["sample_id"], "labels.sample_id")
        if sample_id in seen_label_ids:
            raise ValueError(f"duplicate or conflicting Misread label for sample_id: {sample_id}")
        seen_label_ids.add(sample_id)
        if row["subject_model_key"] != config.model_key:
            raise ValueError(f"formal label model_key drift: {sample_id}")
        if str(row["protocol"]).lower() != config.protocol:
            raise ValueError(f"formal label protocol drift: {sample_id}")
        if row["sample_type"] not in {"Conflict", "Aligned"}:
            raise ValueError(f"formal label sample_type is invalid: {sample_id}")
        for field in ("probe_eligible", "label_eligible", "blocked", "needs_manual_review"):
            if not isinstance(row[field], bool):
                raise ValueError(f"formal label {field} must be boolean: {sample_id}")
        imported_label = row["imported_label"]
        if row["blocked"]:
            if (
                row["probe_eligible"]
                or row["label_eligible"]
                or row["needs_manual_review"]
                or imported_label is not None
            ):
                raise ValueError(f"blocked formal label has contradictory eligibility: {sample_id}")
            excluded_label_counts["blocked"] += 1
            continue
        if row["needs_manual_review"]:
            if row["probe_eligible"] or row["label_eligible"] or imported_label is not None:
                raise ValueError(f"manual-review row has a fabricated eligible label: {sample_id}")
            excluded_label_counts["manual_review"] += 1
            continue
        if row["label_eligible"] != (imported_label in LABEL_TO_ID):
            raise ValueError(f"formal label eligibility conflicts with imported_label: {sample_id}")
        if row["probe_eligible"] and not row["label_eligible"]:
            raise ValueError(f"probe_eligible row is not label_eligible: {sample_id}")
        if not row["probe_eligible"]:
            excluded_label_counts["not_probe_eligible"] += 1
            continue
        if imported_label not in LABEL_TO_ID:
            raise ValueError(f"probe-eligible imported_label is invalid: {sample_id}")
        if row["sample_type"] != "Conflict":
            excluded_label_counts["aligned_out_of_scope"] += 1
            continue
        split = _non_empty_text(row["representation_split"], "labels.representation_split")
        if split not in PROBE_SPLITS:
            raise ValueError(f"Conflict-only probe uses only the fixed registered splits: {split}")
        group = _non_empty_text(row["split_group_id"], "labels.split_group_id")
        group_splits[group].add(split)
        labels[sample_id] = ProbeSample(
            sample_id,
            group,
            split,
            str(imported_label),
            LABEL_TO_ID[str(imported_label)],
        )
        eligible_rows.append(row)
    if not eligible_rows:
        raise ValueError("formal labels contain no eligible Conflict probe rows")
    leaked = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if leaked:
        raise ValueError(f"split_group_id crosses fixed probe splits: {leaked[:3]}")
    eligible_rows.sort(key=lambda row: str(row["sample_id"]))
    eligible_rows_sha256 = _canonical_jsonl_sha256(eligible_rows)
    if eligible_rows_sha256 != config.labels.expected_eligible_rows_sha256:
        raise ValueError("eligible formal label artifact SHA does not match probe config")

    features: dict[str, dict[str, list[float]]] = {}
    feature_dims: dict[str, int] = {}
    representation_ids: dict[str, set[str]] = {}
    for spec in config.representations:
        rows = _read_jsonl(spec.path)
        if not rows:
            raise ValueError(f"{spec.name} frozen representation manifest is empty")
        by_id: dict[str, list[float]] = {}
        seen_ids: set[str] = set()
        for row in rows:
            _reject_misread_fields(row)
            sample_id = _non_empty_text(row.get("sample_id"), f"{spec.name}.sample_id")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id in {spec.name}: {sample_id}")
            seen_ids.add(sample_id)
            if row.get("repr_key") != spec.repr_key:
                raise ValueError(f"repr_key drift in {spec.name}: {sample_id}")
            if row.get("model_key") != config.model_key:
                raise ValueError(f"model_key drift in {spec.name}: {sample_id}")
            if str(row.get("protocol", "")).lower() != config.protocol:
                raise ValueError(f"protocol drift in {spec.name}: {sample_id}")
            if row.get("prompt_set_key") != config.prompt_set_key:
                raise ValueError(f"prompt_set_key drift in {spec.name}: {sample_id}")
            if row.get("split_assignment_key") != config.split_assignment_key:
                raise ValueError(f"split_assignment_key drift in {spec.name}: {sample_id}")
            if row.get("split_assignment_sha256") != config.split_assignment_sha256:
                raise ValueError(f"split_assignment_sha256 drift in {spec.name}: {sample_id}")
            sample_type = row.get("sample_type")
            if sample_type not in {"Conflict", "Aligned"}:
                raise ValueError(f"invalid sample_type in {spec.name}: {sample_id}")
            vector = _finite_vector(row.get(spec.feature_field), spec, sample_id)
            if sample_type == "Conflict" and sample_id in labels:
                split = row.get("representation_split")
                if split not in PROBE_SPLITS:
                    raise ValueError(
                        f"Conflict row is outside fixed probe splits in {spec.name}: {sample_id}"
                    )
                by_id[sample_id] = vector
        features[spec.name] = by_id
        feature_dims[spec.name] = spec.expected_feature_dim
        representation_ids[spec.name] = set(by_id)

    selected_names = tuple(spec.name for spec in config.representations)
    intersection = set.intersection(*(representation_ids[name] for name in selected_names))
    if any(representation_ids[name] != intersection for name in selected_names):
        counts = {name: len(representation_ids[name]) for name in selected_names}
        raise ValueError(f"frozen representation sample intersection drift: {counts}")
    if set(labels) != intersection:
        raise ValueError(
            "eligible formal label sample IDs must be present in all three representations"
        )
    digest = _sample_ids_sha256(intersection)
    if digest != config.expected_sample_ids_sha256:
        raise ValueError("Conflict sample intersection SHA does not match probe config")

    for spec in config.representations:
        rows_by_id = _index_jsonl(spec.path)
        for sample_id, sample in labels.items():
            row = rows_by_id[sample_id]
            if row["sample_type"] != "Conflict":
                raise ValueError(f"sample_type mismatch for {spec.name}: {sample_id}")
            if row.get("split_group_id") != sample.split_group_id:
                raise ValueError(f"split_group_id mismatch for {spec.name}: {sample_id}")
            if row.get("representation_split") != sample.representation_split:
                raise ValueError(f"representation_split mismatch for {spec.name}: {sample_id}")

    samples = tuple(labels[sample_id] for sample_id in sorted(labels))
    _validate_split_content(samples)
    return ProbeDataset(
        samples,
        features,
        feature_dims,
        digest,
        tuple(eligible_rows),
        eligible_rows_sha256,
        excluded_label_counts,
        label_root_status,
    )


def _run_representation(
    *,
    config: ProbeConfig,
    dataset: ProbeDataset,
    spec: RepresentationSpec,
    eligible_labels_path: Path,
) -> dict[str, Any]:
    output_root = config.output_root / spec.name
    output_root.mkdir(parents=True, exist_ok=False)
    device = _registered_device(config.training.device)
    _seed(config.training.seed)
    model = UnifiedMisreadMLP(
        dataset.feature_dims[spec.name],
        hidden_dim=config.training.hidden_dim,
        dropout=config.training.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    objective = nn.CrossEntropyLoss()
    train_samples = _split_samples(dataset.samples, "relation_train")
    val_samples = _split_samples(dataset.samples, "relation_val")
    test_samples = _split_samples(dataset.samples, "official_test")
    train_features, train_labels = _tensors(dataset, spec.name, train_samples, device)
    val_features, val_labels = _tensors(dataset, spec.name, val_samples, device)
    log_rows: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    optimizer_steps = 0

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.training.seed + epoch)
        order = torch.randperm(len(train_samples), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(train_samples), config.training.batch_size):
            indices = order[start : start + config.training.batch_size].to(device)
            logits = model(train_features.index_select(0, indices))
            loss = objective(logits, train_labels.index_select(0, indices))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.detach().cpu()) * len(indices)
        val_predictions, val_scores = _predict(model, val_features)
        val_metrics = _binary_metrics(
            val_labels.detach().cpu().tolist(), val_predictions, val_scores
        )
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(train_samples),
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_ap": val_metrics["ap"],
                "optimizer_steps": optimizer_steps,
            }
        )
        if val_metrics["balanced_accuracy"] > best_score:
            best_score = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("fixed training budget produced no checkpoint")
    model.load_state_dict(best_state)

    test_features, test_labels = _tensors(dataset, spec.name, test_samples, device)
    test_predictions, test_scores = _predict(model, test_features)
    labels = test_labels.detach().cpu().tolist()
    metrics = _binary_metrics(labels, test_predictions, test_scores)
    metrics_payload = {
        "schema": "mprisk_conflict_misread_probe_metrics_v1",
        "task": "Conflict_only_Misread_vs_Non-misread",
        "positive_class": "Misread",
        "representation": spec.name,
        "repr_key": spec.repr_key,
        "selection_rule": "representation_split=official_test",
        "sample_count": len(test_samples),
        "label_counts": _label_counts(test_samples),
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_score,
        **metrics,
    }
    metrics_path = _atomic_json(output_root / "metrics.json", metrics_payload)
    predictions_path = _atomic_jsonl(
        output_root / "predictions.jsonl",
        [
            {
                "schema": "mprisk_conflict_misread_probe_prediction_v1",
                "representation": spec.name,
                "sample_id": sample.sample_id,
                "split_group_id": sample.split_group_id,
                "representation_split": sample.representation_split,
                "misread_label": sample.label,
                "misread_binary_label": sample.label_id,
                "prediction_label": "Misread" if prediction == 1 else "Non-misread",
                "prediction_id": prediction,
                "misread_score": score,
            }
            for sample, prediction, score in zip(
                test_samples, test_predictions, test_scores, strict=True
            )
        ],
    )
    pr_path = _atomic_csv(
        output_root / "pr_curve.csv",
        ("threshold", "recall", "precision"),
        _precision_recall_rows(labels, test_scores),
    )
    log_path = _atomic_jsonl(output_root / "train_log.jsonl", log_rows)
    checkpoint_path = _atomic_torch_save(
        output_root / "checkpoint.pt",
        {
            "schema": "mprisk_conflict_misread_probe_checkpoint_v1",
            "run_id": config.run_id,
            "model_key": config.model_key,
            "protocol": config.protocol,
            "prompt_set_key": config.prompt_set_key,
            "representation": spec.name,
            "repr_key": spec.repr_key,
            "feature_field": spec.feature_field,
            "feature_dim": dataset.feature_dims[spec.name],
            "sample_ids_sha256": dataset.sample_ids_sha256,
            "split_assignment_sha256": config.split_assignment_sha256,
            "architecture": _architecture_payload(config.training),
            "training_budget": _training_payload(config.training),
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_score,
            "model_state_dict": best_state,
        },
    )
    provenance_path = _atomic_json(
        output_root / "provenance.json",
        {
            "schema": "mprisk_conflict_misread_probe_provenance_v1",
            "run_id": config.run_id,
            "representation": spec.name,
            "config": {"path": str(config.config_path), "sha256": _sha256(config.config_path)},
            "inputs": {
                "formal_label_root": {
                    "path": str(config.labels.root),
                    "complete_path": str(config.labels.complete_path),
                    "complete_sha256": config.labels.complete_sha256,
                    "status": dataset.label_root_status,
                    "eligible_subset_complete": True,
                },
                "labels": {
                    "path": str(config.labels.labels_path),
                    "sha256": config.labels.sha256,
                },
                "eligible_labels": {
                    "path": str(eligible_labels_path),
                    "sha256": dataset.eligible_rows_sha256,
                },
                "representation": {"path": str(spec.path), "sha256": spec.sha256},
            },
            "sample_ids_sha256": dataset.sample_ids_sha256,
            "excluded_label_counts": dataset.excluded_label_counts,
            "split_assignment_key": config.split_assignment_key,
            "split_assignment_sha256": config.split_assignment_sha256,
            "split_policy": {
                "train": "relation_train",
                "validation": "relation_val",
                "test": "official_test",
                "unit": "split_group_id",
            },
            "sample_counts": {
                split: len(_split_samples(dataset.samples, split)) for split in PROBE_SPLITS
            },
            "group_counts": {
                split: len(
                    {sample.split_group_id for sample in _split_samples(dataset.samples, split)}
                )
                for split in PROBE_SPLITS
            },
            "label_counts": {
                split: _label_counts(_split_samples(dataset.samples, split))
                for split in PROBE_SPLITS
            },
            "representation_policy": "frozen_no_encoder_gradients",
            "feature_field": spec.feature_field,
            "feature_dim": dataset.feature_dims[spec.name],
            "architecture": _architecture_payload(config.training),
            "training_budget": _training_payload(config.training),
            "selection_metric": "relation_val balanced_accuracy",
            "best_epoch": best_epoch,
            "optimizer_steps": optimizer_steps,
        },
    )
    artifacts = {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "pr_curve": pr_path,
        "train_log": log_path,
        "checkpoint": checkpoint_path,
        "provenance": provenance_path,
    }
    return {
        "representation": spec.name,
        "repr_key": spec.repr_key,
        "feature_dim": dataset.feature_dims[spec.name],
        "metrics": metrics_payload,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in artifacts.items()
        },
    }


def _validate_split_content(samples: tuple[ProbeSample, ...]) -> None:
    groups_by_split: dict[str, set[str]] = {}
    for split in PROBE_SPLITS:
        selected = _split_samples(samples, split)
        if not selected:
            raise ValueError(f"fixed probe split is empty: {split}")
        if {sample.label_id for sample in selected} != {0, 1}:
            raise ValueError(f"fixed probe split must contain both Misread classes: {split}")
        groups_by_split[split] = {sample.split_group_id for sample in selected}
    for index, left in enumerate(PROBE_SPLITS):
        for right in PROBE_SPLITS[index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise ValueError(f"split_group_id leakage between {left} and {right}")


def _tensors(
    dataset: ProbeDataset,
    representation: str,
    samples: list[ProbeSample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [dataset.features[representation][sample.sample_id] for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor([sample.label_id for sample in samples], dtype=torch.long, device=device)
    return features, labels


def _predict(model: nn.Module, features: torch.Tensor) -> tuple[list[int], list[float]]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("probe produced non-finite logits")
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()
        scores = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
    return [int(value) for value in predictions], [float(value) for value in scores]


def _binary_metrics(
    labels: list[int], predictions: list[int], scores: list[float]
) -> dict[str, float]:
    if not labels or len(labels) != len(predictions) or len(labels) != len(scores):
        raise ValueError("binary metric inputs must be non-empty and aligned")
    if set(labels) != {0, 1}:
        raise ValueError("Accuracy/BA/Macro-F1/AP require both classes")
    if any(value not in {0, 1} for value in predictions):
        raise ValueError("binary predictions must be 0 or 1")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Misread scores must be finite probabilities")
    accuracy = sum(left == right for left, right in zip(labels, predictions, strict=True)) / len(
        labels
    )
    recalls: list[float] = []
    f1_values: list[float] = []
    for class_id in (0, 1):
        tp = sum(
            truth == class_id and predicted == class_id
            for truth, predicted in zip(labels, predictions, strict=True)
        )
        fn = sum(
            truth == class_id and predicted != class_id
            for truth, predicted in zip(labels, predictions, strict=True)
        )
        fp = sum(
            truth != class_id and predicted == class_id
            for truth, predicted in zip(labels, predictions, strict=True)
        )
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recalls.append(recall)
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(sum(recalls) / 2),
        "macro_f1": float(sum(f1_values) / 2),
        "ap": float(_average_precision(labels, scores)),
    }


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives <= 0:
        raise ValueError("AP requires at least one Misread sample")
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    ap = 0.0
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == threshold:
            if ranked[index][1] == 1:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def _precision_recall_rows(labels: list[int], scores: list[float]) -> list[dict[str, Any]]:
    positives = sum(labels)
    if positives <= 0:
        raise ValueError("PR curve requires at least one Misread sample")
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    rows: list[dict[str, Any]] = [{"threshold": "", "recall": 0.0, "precision": 1.0}]
    true_positive = 0
    false_positive = 0
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == threshold:
            if ranked[index][1] == 1:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        rows.append(
            {
                "threshold": threshold,
                "recall": true_positive / positives,
                "precision": true_positive / (true_positive + false_positive),
            }
        )
    return rows


def _finite_vector(value: Any, spec: RepresentationSpec, sample_id: str) -> list[float]:
    if not isinstance(value, list) or len(value) != spec.expected_feature_dim:
        raise ValueError(
            f"{spec.name} feature dimension drift for {sample_id}: "
            f"expected {spec.expected_feature_dim}"
        )
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise ValueError(f"{spec.name} feature must be numeric: {sample_id}")
    vector = [float(item) for item in value]
    if any(not math.isfinite(item) for item in vector):
        raise ValueError(f"{spec.name} feature must be finite: {sample_id}")
    return vector


def _reject_misread_fields(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "misread" in str(key).casefold():
                raise ValueError(f"Misread leakage in frozen representation: {path}.{key}")
            _reject_misread_fields(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_misread_fields(child, path=f"{path}[{index}]")


def _architecture_payload(training: TrainingBudget) -> dict[str, Any]:
    return {
        "shared_across_representations": True,
        "layers": [
            f"Linear(input_dim,{training.hidden_dim})",
            "GELU",
            f"Dropout({training.dropout})",
            f"Linear({training.hidden_dim},2)",
        ],
    }


def _training_payload(training: TrainingBudget) -> dict[str, Any]:
    return asdict(training)


def _registered_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"configured probe device is unavailable: {value}")
    return device


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_samples(samples: tuple[ProbeSample, ...], split: str) -> list[ProbeSample]:
    return [sample for sample in samples if sample.representation_split == split]


def _label_counts(samples: list[ProbeSample]) -> dict[str, int]:
    return {label: sum(sample.label == label for sample in samples) for label in LABEL_TO_ID}


def _sample_ids_sha256(sample_ids: set[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(sample_ids), separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_jsonl_bytes(rows)).hexdigest()


def _canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _verify_formal_label_root(config: ProbeConfig) -> dict[str, Any]:
    root = config.labels.root
    if not root.is_dir():
        raise FileNotFoundError(root)
    _verify_artifact(
        config.labels.complete_path,
        config.labels.complete_sha256,
        "formal label COMPLETE marker",
    )
    sidecar_path = root / "COMPLETE.json.sha256"
    checksums_path = root / "artifact_checksums.json"
    for path in (sidecar_path, checksums_path, config.labels.labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sidecar = sidecar_path.read_text(encoding="utf-8").strip().split()
    if sidecar != [config.labels.complete_sha256, "COMPLETE.json"]:
        raise ValueError("formal label COMPLETE.json.sha256 mismatch")
    marker = json.loads(config.labels.complete_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict) or marker.get("schema") != FORMAL_LABEL_ROOT_SCHEMA:
        raise ValueError("formal label root schema mismatch")
    if marker.get("status") not in {"complete", "partial_manual_review_required"}:
        raise ValueError("formal label root status is invalid")
    if marker.get("eligible_subset_complete") is not True:
        raise ValueError("formal label eligible subset is not complete")
    models = marker.get("models")
    if (
        not isinstance(models, list)
        or any(not isinstance(model, str) or not model for model in models)
        or config.model_key not in models
    ):
        raise ValueError("formal label root does not contain the configured model")
    checksums_sha256 = _sha256(checksums_path)
    if marker.get("artifact_checksums_sha256") != checksums_sha256:
        raise ValueError("formal label artifact_checksums SHA-256 mismatch")
    checksum_doc = json.loads(checksums_path.read_text(encoding="utf-8"))
    artifacts = checksum_doc.get("artifacts") if isinstance(checksum_doc, dict) else None
    relative = f"labels/{config.model_key}.jsonl"
    evidence = artifacts.get(relative) if isinstance(artifacts, dict) else None
    if not isinstance(evidence, dict):
        raise ValueError(f"formal label checksum manifest is missing {relative}")
    if (
        evidence.get("sha256") != config.labels.sha256
        or evidence.get("bytes") != config.labels.labels_path.stat().st_size
    ):
        raise ValueError("formal label artifact evidence conflicts with probe config")
    return marker


def _verify_artifact(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 does not match probe config")


def _index_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        sample_id = _non_empty_text(row.get("sample_id"), f"{path}.sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        result[sample_id] = row
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def _prepare_empty_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"probe output_root must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _atomic_json(path: Path, payload: Any) -> Path:
    return _atomic_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _atomic_canonical_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_jsonl_bytes(rows))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _atomic_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _atomic_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def _resolve(root: Path, value: Any) -> Path:
    path = Path(_non_empty_text(value, "artifact path"))
    return path if path.is_absolute() else (root / path).resolve()


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _sha_field(value: Any, field: str) -> str:
    text = _non_empty_text(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _require_exact_keys(value: Any, expected: set[str] | frozenset[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(expected))}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
