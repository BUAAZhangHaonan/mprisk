"""Training and frozen export for sample-level relation representations."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.cache.prompt_conditioned_cache import prompt_conditioned_entry_from_row
from mprisk.representation.losses import ProxyAnchorLoss
from mprisk.representation.relation_dataset import CONDITIONS, _reject_forbidden_fields
from mprisk.representation.relation_models import (
    REPRESENTATION_KEYS,
    SINGLE_POINT_BINARY_V1,
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
    TRAJECTORY_MLP_BINARY_V1,
    build_representation_model,
)
from mprisk.utils.io import write_json

TRAINING_CONFIG_SCHEMA = "mprisk_representation_training_v3"
REGISTERED_SPLITS = frozenset(
    {"relation_train", "relation_val", "aligned_calibration", "official_test"}
)


@dataclass(frozen=True)
class TrainingConfig:
    repr_key: str
    model_key: str
    hidden_dim: int = 128
    condition_dim: int = 64
    relation_dim: int = 32
    dropout: float = 0.1
    max_epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    proxy_alpha: float = 32.0
    proxy_margin: float = 0.1
    patience: int = 10
    min_delta: float = 1e-4
    seed: int = 0


@dataclass(frozen=True)
class TrainingResult:
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    config_path: Path
    metrics_path: Path
    log_path: Path
    metrics: dict[str, Any]
    resumed_from: Path | None

    @property
    def checkpoint_path(self) -> Path:
        return self.best_checkpoint_path


@dataclass(frozen=True)
class FrozenRepresentationExportResult:
    manifest_path: Path
    bundle_manifest_path: Path
    summary_path: Path
    count: int


@dataclass(frozen=True)
class FrozenBaselineExportResult:
    manifest_path: Path
    summary_path: Path
    count: int


@dataclass(frozen=True)
class _Sample:
    row_id: str
    sample_id: str
    sample_type: str
    label_id: int
    split_group_id: str
    master_split: str
    representation_split: str
    calibration_split: str
    split_assignment_key: str
    split_assignment_sha256: str
    protocol: str
    prompt_set_key: str
    prompt_id: str
    condition_entries: tuple[Any, Any, Any]


def load_training_config(path: str | Path) -> TrainingConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML mapping")
    payload = dict(payload)
    if payload.pop("schema", None) != TRAINING_CONFIG_SCHEMA:
        raise ValueError(f"training config schema must be {TRAINING_CONFIG_SCHEMA}")
    key = payload.pop("key", None)
    if not isinstance(key, str) or not key.strip():
        raise ValueError("training config key must be non-empty text")
    architecture_version = payload.pop("architecture_version", None)
    if payload.get("repr_key") == TME_PROXY_ANCHOR_V1:
        if architecture_version != TME_ARCHITECTURE_V1:
            raise ValueError(
                f"TME architecture_version must be {TME_ARCHITECTURE_V1}"
            )
    elif architecture_version is not None and architecture_version != payload.get("repr_key"):
        raise ValueError("baseline architecture_version must match repr_key when provided")
    unknown = set(payload) - set(TrainingConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown training config fields: {', '.join(sorted(unknown))}")
    config = TrainingConfig(**payload)
    _validate_config(config)
    return config


def train_trajectory_encoder(
    *,
    dataset_path: str | Path,
    config: TrainingConfig,
    output_dir: str | Path,
    resume_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train one backbone-specific representation with group-disjoint A/C validation."""
    _validate_config(config)
    _set_deterministic_seed(config.seed)
    signature = _training_signature(dataset_path, config)
    resume_payload: dict[str, Any] | None = None
    resumed_from_path: Path | None = None
    if resume_checkpoint is not None:
        resumed_from_path = Path(resume_checkpoint)
        resume_payload = torch.load(resumed_from_path, map_location="cpu")
        if resume_payload.get("training_signature") != signature:
            raise ValueError("resume signature mismatch")
    rows = _read_relation_rows(dataset_path, expected_model_key=config.model_key)
    split_contract = _validate_registered_splits(rows)
    training_rows = [
        row
        for row in rows
        if row["representation_split"] in {"relation_train", "relation_val"}
    ]
    samples = _rows_to_sample_refs(training_rows)
    train_samples, val_samples = _registered_group_split(samples)
    layer_count, input_dim = _trajectory_shape(samples)
    torch_device = _resolve_device(device)
    model = build_representation_model(
        config.repr_key,
        input_dim=input_dim,
        layer_count=layer_count,
        hidden_dim=config.hidden_dim,
        condition_dim=config.condition_dim,
        relation_dim=config.relation_dim,
        dropout=config.dropout,
    ).to(torch_device)
    objective: ProxyAnchorLoss | None = None
    parameters: list[nn.Parameter] = list(model.parameters())
    if config.repr_key == TME_PROXY_ANCHOR_V1:
        objective = ProxyAnchorLoss(
            embed_dim=config.relation_dim,
            num_classes=2,
            alpha=config.proxy_alpha,
            margin=config.proxy_margin,
        ).to(torch_device)
        parameters.extend(objective.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    best_path = output_root / "best_checkpoint.pt"
    last_path = output_root / "last_checkpoint.pt"
    config_path = output_root / "train_config.yaml"
    metrics_path = output_root / "train_metrics.json"
    log_path = output_root / "train_log.jsonl"
    start_epoch = 1
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    if resume_payload is not None:
        checkpoint = resume_payload
        model.load_state_dict(checkpoint["model_state_dict"])
        if objective is not None:
            objective.load_state_dict(checkpoint["proxy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, torch_device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
    else:
        log_path.write_text("", encoding="utf-8")

    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=True), encoding="utf-8")
    stop_reason = "max_epochs"
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.max_epochs + 1):
        final_epoch = epoch
        train_loss = _train_epoch(
            model,
            objective,
            optimizer,
            train_samples,
            config=config,
            epoch=epoch,
        )
        val_loss, val_score = _evaluate(model, objective, val_samples, config=config)
        improved = val_score > best_score + config.min_delta
        if improved:
            best_score = val_score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_balanced_accuracy_ac": val_score,
            "val_sample_count": len({sample.sample_id for sample in val_samples}),
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy_ac": best_score,
            "stale_epochs": stale_epochs,
            "converged": stale_epochs >= config.patience,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_row, sort_keys=True) + "\n")
        checkpoint = _checkpoint_payload(
            model=model,
            objective=objective,
            optimizer=optimizer,
            config=config,
            input_dim=input_dim,
            layer_count=layer_count,
            signature=signature,
            epoch=epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
        )
        _atomic_torch_save(last_path, checkpoint)
        if improved:
            _atomic_torch_save(best_path, checkpoint)
        if stale_epochs >= config.patience:
            stop_reason = "early_stopping"
            break
    if not best_path.is_file() and last_path.is_file():
        _atomic_torch_save(best_path, torch.load(last_path, map_location="cpu"))
    metrics = {
        "schema": "mprisk_representation_training_metrics_v2",
        "repr_key": config.repr_key,
        "model_key": config.model_key,
        "selection_metric": "val_balanced_accuracy_ac",
        "selection_unit": "sample_id",
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy_ac": best_score,
        "final_epoch": final_epoch,
        "stop_reason": stop_reason,
        "train_rows": len(train_samples),
        "val_rows": len(val_samples),
        "train_sample_count": len({sample.sample_id for sample in train_samples}),
        "val_sample_count": len({sample.sample_id for sample in val_samples}),
        "train_examples_per_epoch": len({sample.sample_id for sample in train_samples}),
        "prompt_augmentation": "one_deterministic_prompt_per_sample_per_epoch",
        "train_group_count": len({sample.split_group_id for sample in train_samples}),
        "val_group_count": len({sample.split_group_id for sample in val_samples}),
        "train_groups_sha256": _group_checksum(train_samples),
        "val_groups_sha256": _group_checksum(val_samples),
        "split_assignment_key": split_contract["split_assignment_key"],
        "split_assignment_sha256": split_contract["split_assignment_sha256"],
        "excluded_rows": {
            split: sum(row["representation_split"] == split for row in rows)
            for split in ("aligned_calibration", "official_test")
        },
        "training_signature": signature,
        "resumed_from": str(resumed_from_path) if resumed_from_path else None,
        "device": str(torch_device),
    }
    write_json(metrics_path, metrics)
    return TrainingResult(
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
        config_path=config_path,
        metrics_path=metrics_path,
        log_path=log_path,
        metrics=metrics,
        resumed_from=resumed_from_path,
    )


def export_frozen_representations(
    *,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> FrozenRepresentationExportResult:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    if checkpoint.get("repr_key") != TME_PROXY_ANCHOR_V1:
        raise ValueError(
            "condition z and relation r export requires a tme_proxy_anchor_v1 checkpoint"
        )
    config = TrainingConfig(**checkpoint["training_config"])
    rows = _read_relation_rows(dataset_path, expected_model_key=config.model_key)
    samples = _rows_to_sample_refs(rows)
    model = build_representation_model(
        config.repr_key,
        input_dim=int(checkpoint["model_config"]["input_dim"]),
        layer_count=int(checkpoint["model_config"]["layer_count"]),
        hidden_dim=config.hidden_dim,
        condition_dim=config.condition_dim,
        relation_dim=config.relation_dim,
        dropout=config.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "frozen_representations.jsonl"
    bundle_manifest_path = output_root / "spherical_embedding_manifest.jsonl"
    sample_count = _stream_frozen_exports(
        samples=sorted(samples, key=lambda sample: (sample.sample_id, sample.prompt_id)),
        model=model,
        config=config,
        manifest_path=manifest_path,
        bundle_manifest_path=bundle_manifest_path,
    )
    summary_path = write_json(
        output_root / "frozen_representation_summary.json",
        {
            "schema": "mprisk_frozen_spherical_representation_summary_v1",
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_path),
            "count": len(samples),
            "sample_count": sample_count,
            "bundle_manifest": str(bundle_manifest_path),
            "repr_key": config.repr_key,
            "model_key": config.model_key,
        },
    )
    return FrozenRepresentationExportResult(
        manifest_path, bundle_manifest_path, summary_path, len(samples)
    )


def export_frozen_baseline_representations(
    *,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    representation_split: str = "official_test",
) -> FrozenBaselineExportResult:
    if representation_split not in {"relation_val", "aligned_calibration", "official_test"}:
        raise ValueError("baseline export requires a held-out representation split")
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    repr_key = str(checkpoint.get("repr_key", ""))
    if repr_key not in {SINGLE_POINT_BINARY_V1, TRAJECTORY_MLP_BINARY_V1}:
        raise ValueError("baseline export requires a Single-Point or Trajectory MLP checkpoint")
    if checkpoint.get("proxy_state_dict") is not None:
        raise ValueError("baseline checkpoints must not contain Proxy Anchor state")
    config = TrainingConfig(**checkpoint["training_config"])
    rows = _read_relation_rows(dataset_path, expected_model_key=config.model_key)
    _validate_registered_splits(rows)
    selected_rows = [
        row for row in rows if row["representation_split"] == representation_split
    ]
    if not selected_rows:
        raise ValueError(f"relation dataset has no rows for {representation_split}")
    samples = sorted(
        _rows_to_sample_refs(selected_rows),
        key=lambda sample: (sample.sample_id, sample.prompt_id),
    )
    model = build_representation_model(
        repr_key,
        input_dim=int(checkpoint["model_config"]["input_dim"]),
        layer_count=int(checkpoint["model_config"]["layer_count"]),
        hidden_dim=config.hidden_dim,
        condition_dim=config.condition_dim,
        relation_dim=config.relation_dim,
        dropout=config.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "frozen_baseline_representations.jsonl"
    checkpoint_sha256 = _sha256(checkpoint_file)
    sample_count, feature_dim = _stream_baseline_exports(
        samples=samples,
        model=model,
        batch_size=config.batch_size,
        model_key=config.model_key,
        repr_key=repr_key,
        checkpoint_sha256=checkpoint_sha256,
        manifest_path=manifest_path,
    )
    summary_path = write_json(
        output_root / "frozen_baseline_summary.json",
        {
            "schema": "mprisk_frozen_baseline_summary_v1",
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(Path(dataset_path)),
            "checkpoint": str(checkpoint_file),
            "encoder_checkpoint_sha256": checkpoint_sha256,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "model_key": config.model_key,
            "repr_key": repr_key,
            "representation_split": representation_split,
            "aggregation": "mean_over_synchronized_prompts",
            "feature_dim": feature_dim,
            "sample_count": sample_count,
        },
    )
    return FrozenBaselineExportResult(manifest_path, summary_path, sample_count)


def _stream_baseline_exports(
    *,
    samples: list[_Sample],
    model: nn.Module,
    batch_size: int,
    model_key: str,
    repr_key: str,
    checkpoint_sha256: str,
    manifest_path: Path,
) -> tuple[int, int]:
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    current_sample: _Sample | None = None
    feature_sum: torch.Tensor | None = None
    logits_sum: torch.Tensor | None = None
    prompt_count = 0
    prompt_counts: set[int] = set()
    sample_count = 0
    feature_dim = 0
    with temporary.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in _batches(samples, batch_size):
            trajectories, _labels = _load_trajectory_batch(
                batch, device=next(model.parameters()).device
            )
            features = model.forward_features(trajectories)
            logits = model.classifier(features)
            for index, sample in enumerate(batch):
                if current_sample is not None and sample.sample_id != current_sample.sample_id:
                    row = _baseline_export_row(
                        current_sample,
                        feature_sum=feature_sum,
                        logits_sum=logits_sum,
                        prompt_count=prompt_count,
                        model_key=model_key,
                        repr_key=repr_key,
                        checkpoint_sha256=checkpoint_sha256,
                    )
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    prompt_counts.add(prompt_count)
                    sample_count += 1
                    feature_dim = len(row["penultimate_feature"])
                    feature_sum = None
                    logits_sum = None
                    prompt_count = 0
                current_sample = sample
                feature_sum = (
                    features[index].clone()
                    if feature_sum is None
                    else feature_sum + features[index]
                )
                logits_sum = (
                    logits[index].clone() if logits_sum is None else logits_sum + logits[index]
                )
                prompt_count += 1
        if current_sample is not None:
            row = _baseline_export_row(
                current_sample,
                feature_sum=feature_sum,
                logits_sum=logits_sum,
                prompt_count=prompt_count,
                model_key=model_key,
                repr_key=repr_key,
                checkpoint_sha256=checkpoint_sha256,
            )
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            prompt_counts.add(prompt_count)
            sample_count += 1
            feature_dim = len(row["penultimate_feature"])
        handle.flush()
        os.fsync(handle.fileno())
    if len(prompt_counts) != 1:
        raise ValueError("held-out samples must have synchronized prompt counts")
    os.replace(temporary, manifest_path)
    return sample_count, feature_dim


def _baseline_export_row(
    sample: _Sample,
    *,
    feature_sum: torch.Tensor | None,
    logits_sum: torch.Tensor | None,
    prompt_count: int,
    model_key: str,
    repr_key: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    if feature_sum is None or logits_sum is None or prompt_count <= 0:
        raise ValueError("baseline sample aggregate is empty")
    mean_feature = feature_sum / prompt_count
    mean_logits = logits_sum / prompt_count
    prediction_id = int(mean_logits.argmax())
    return {
        "schema": "mprisk_frozen_baseline_representation_v1",
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "label_id": sample.label_id,
        "model_key": model_key,
        "protocol": sample.protocol,
        "prompt_set_key": sample.prompt_set_key,
        "master_split": sample.master_split,
        "representation_split": sample.representation_split,
        "split_group_id": sample.split_group_id,
        "split_assignment_key": sample.split_assignment_key,
        "split_assignment_sha256": sample.split_assignment_sha256,
        "repr_key": repr_key,
        "encoder_checkpoint_sha256": checkpoint_sha256,
        "aggregation": "mean_over_synchronized_prompts",
        "prompt_count": prompt_count,
        "penultimate_feature": _vector_values(mean_feature),
        "mean_logits": _vector_values(mean_logits),
        "prediction_id": prediction_id,
        "prediction_label": "Conflict" if prediction_id == 1 else "Aligned",
    }


def _stream_frozen_exports(
    *,
    samples: list[_Sample],
    model: nn.Module,
    config: TrainingConfig,
    manifest_path: Path,
    bundle_manifest_path: Path,
) -> int:
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    bundle_tmp = bundle_manifest_path.with_suffix(bundle_manifest_path.suffix + ".tmp")
    current_bundle: dict[str, Any] | None = None
    sample_count = 0
    with (
        manifest_tmp.open("w", encoding="utf-8") as manifest_handle,
        bundle_tmp.open("w", encoding="utf-8") as bundle_handle,
        torch.no_grad(),
    ):
        for batch in _batches(samples, config.batch_size):
            trajectories, _labels = _load_trajectory_batch(
                batch, device=next(model.parameters()).device
            )
            condition_z, relation_r = model(
                trajectories,
                sample_ids=[sample.sample_id for sample in batch],
            )
            for index, sample in enumerate(batch):
                row = _frozen_row(
                    sample,
                    model_key=config.model_key,
                    repr_key=config.repr_key,
                    condition_z=condition_z[index],
                    relation_r=relation_r[index],
                )
                manifest_handle.write(json.dumps(row, sort_keys=True) + "\n")
                if current_bundle is None or current_bundle["sample_id"] != sample.sample_id:
                    if current_bundle is not None:
                        bundle_handle.write(json.dumps(current_bundle, sort_keys=True) + "\n")
                    current_bundle = _empty_frozen_bundle(row)
                    sample_count += 1
                _append_frozen_row(current_bundle, row)
        if current_bundle is not None:
            bundle_handle.write(json.dumps(current_bundle, sort_keys=True) + "\n")
        for handle in (manifest_handle, bundle_handle):
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(manifest_tmp, manifest_path)
    os.replace(bundle_tmp, bundle_manifest_path)
    return sample_count


def _frozen_row(
    sample: _Sample,
    *,
    model_key: str,
    repr_key: str,
    condition_z: torch.Tensor,
    relation_r: torch.Tensor,
) -> dict[str, Any]:
    return {
        "schema": "mprisk_frozen_spherical_representation_v1",
        "row_id": sample.row_id,
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "model_key": model_key,
        "protocol": sample.protocol,
        "prompt_set_key": sample.prompt_set_key,
        "calibration_split": sample.calibration_split,
        "master_split": sample.master_split,
        "representation_split": sample.representation_split,
        "split_group_id": sample.split_group_id,
        "split_assignment_key": sample.split_assignment_key,
        "split_assignment_sha256": sample.split_assignment_sha256,
        "prompt_id": sample.prompt_id,
        "repr_key": repr_key,
        "condition_z": {
            condition: _vector_values(condition_z[index])
            for index, condition in enumerate(CONDITIONS)
        },
        "relation_r": _vector_values(relation_r),
    }


def _empty_frozen_bundle(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "sample_id",
            "sample_type",
            "model_key",
            "protocol",
            "prompt_set_key",
            "calibration_split",
            "master_split",
            "representation_split",
            "split_group_id",
            "split_assignment_key",
            "split_assignment_sha256",
            "repr_key",
        )
    } | {
        "embeddings": {condition: {} for condition in CONDITIONS},
        "relations": {},
    }


def _append_frozen_row(bundle: dict[str, Any], row: dict[str, Any]) -> None:
    prompt_id = str(row["prompt_id"])
    for condition in CONDITIONS:
        bundle["embeddings"][condition][prompt_id] = row["condition_z"][condition]
    bundle["relations"][prompt_id] = row["relation_r"]


def _vector_values(vector: torch.Tensor) -> list[float]:
    return [float(value) for value in vector.detach().cpu().numpy()]


def _read_relation_rows(path: str | Path, *, expected_model_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: relation row must be an object")
            _reject_forbidden_fields(row)
            if row.get("schema") != "mprisk_relation_sample_v1":
                raise ValueError("relation row schema mismatch")
            if row.get("model_key") != expected_model_key:
                raise ValueError("relation dataset model_key does not match training backbone")
            if row.get("sample_type") not in {"Aligned", "Conflict"}:
                raise ValueError("relation training labels must be Conflict or Aligned")
            expected_label = int(row["sample_type"] == "Conflict")
            if row.get("label_id") != expected_label:
                raise ValueError("label_id must be derived from the sample-level A/C label")
            if set(row.get("conditions", {})) != set(CONDITIONS):
                raise ValueError("relation row requires exactly M1, M2, and M12")
            rows.append(row)
    if not rows:
        raise ValueError("relation dataset is empty")
    row_ids = [str(row.get("row_id")) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("relation dataset row_id values must be unique")
    return rows


def _rows_to_sample_refs(rows: list[dict[str, Any]]) -> list[_Sample]:
    samples: list[_Sample] = []
    expected_shape: tuple[int, int] | None = None
    for row in rows:
        entries = tuple(
            prompt_conditioned_entry_from_row(row["conditions"][condition])
            for condition in CONDITIONS
        )
        expected_key = (
            str(row["sample_id"]),
            str(row["model_key"]),
            str(row["protocol"]).lower(),
            str(row["prompt_set_key"]),
            str(row["prompt_id"]),
        )
        for condition, entry in zip(CONDITIONS, entries, strict=True):
            actual_key = (
                entry.sample_id,
                entry.model_key,
                entry.protocol,
                entry.prompt_set_key,
                entry.prompt_id,
            )
            if actual_key != expected_key or entry.condition != condition:
                raise ValueError(
                    "M1, M2, and M12 cache entries must use the same "
                    "sample/model/protocol/prompt as the relation row"
                )
        shapes = {(entry.layer_count, entry.hidden_dim) for entry in entries}
        if len(shapes) != 1:
            raise ValueError("all three condition trajectories must have the same shape")
        shape = next(iter(shapes))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all condition trajectories must have the same layer/hidden shape")
        samples.append(
            _Sample(
                row_id=str(row["row_id"]),
                sample_id=str(row["sample_id"]),
                sample_type=str(row["sample_type"]),
                label_id=int(row["label_id"]),
                split_group_id=str(row["split_group_id"]),
                master_split=str(row.get("master_split", "")),
                representation_split=str(row.get("representation_split", "")),
                calibration_split=str(row.get("calibration_split", "")),
                split_assignment_key=str(row.get("split_assignment_key", "")),
                split_assignment_sha256=str(row.get("split_assignment_sha256", "")),
                protocol=str(row["protocol"]),
                prompt_set_key=str(row["prompt_set_key"]),
                prompt_id=str(row["prompt_id"]),
                condition_entries=tuple(entry.to_hidden_state_entry() for entry in entries),
            )
        )
    return samples


def _load_trajectory_batch(
    batch: list[_Sample], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = [
        np.stack([extract_t0_trajectory(entry) for entry in sample.condition_entries])
        for sample in batch
    ]
    trajectories = torch.from_numpy(np.stack(arrays).astype(np.float32, copy=False)).to(device)
    labels = torch.tensor(
        [sample.label_id for sample in batch], dtype=torch.long, device=device
    )
    return trajectories, labels


def _registered_group_split(samples: list[_Sample]) -> tuple[list[_Sample], list[_Sample]]:
    groups: dict[str, list[_Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.split_group_id, []).append(sample)
    for group_samples in groups.values():
        splits = {sample.representation_split for sample in group_samples}
        if len(splits) != 1:
            raise ValueError("split_group_id crosses registered representation splits")
    train = [sample for sample in samples if sample.representation_split == "relation_train"]
    val = [sample for sample in samples if sample.representation_split == "relation_val"]
    if not train or not val:
        raise ValueError("registered relation_train and relation_val splits are both required")
    if {sample.label_id for sample in train} != {0, 1} or {sample.label_id for sample in val} != {
        0,
        1,
    }:
        raise ValueError("train and val must both contain Aligned and Conflict samples")
    return train, val


def _validate_registered_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    expected_master = {
        "relation_train": "train",
        "relation_val": "val",
        "aligned_calibration": "val",
        "official_test": "test",
    }
    group_splits: dict[str, set[str]] = {}
    keys: set[str] = set()
    checksums: set[str] = set()
    for row in rows:
        for field in (
            "split_group_id",
            "master_split",
            "representation_split",
            "split_assignment_key",
            "split_assignment_sha256",
        ):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"relation row requires non-empty {field}")
        split = row["representation_split"]
        if split not in REGISTERED_SPLITS:
            raise ValueError(f"unknown representation_split: {split}")
        if row["master_split"] != expected_master[split]:
            raise ValueError(f"{split} mismatches official master_split")
        expected_calibration = "aligned_calibration" if split == "aligned_calibration" else ""
        if str(row.get("calibration_split", "")) != expected_calibration:
            raise ValueError(f"{split} has invalid calibration_split")
        group_splits.setdefault(row["split_group_id"], set()).add(split)
        keys.add(row["split_assignment_key"])
        checksums.add(row["split_assignment_sha256"])
    leaked = [group for group, splits in group_splits.items() if len(splits) != 1]
    if leaked:
        raise ValueError(f"split groups cross registered assignments: {leaked[:3]}")
    if len(keys) != 1 or len(checksums) != 1 or len(next(iter(checksums), "")) != 64:
        raise ValueError("relation rows require one valid split assignment key/checksum")
    return {
        "split_assignment_key": next(iter(keys)),
        "split_assignment_sha256": next(iter(checksums)),
    }


def _group_checksum(samples: list[_Sample]) -> str:
    groups = sorted({sample.split_group_id for sample in samples})
    return hashlib.sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest()


def _train_epoch(
    model: nn.Module,
    objective: ProxyAnchorLoss | None,
    optimizer: torch.optim.Optimizer,
    samples: list[_Sample],
    *,
    config: TrainingConfig,
    epoch: int,
) -> float:
    model.train()
    if objective is not None:
        objective.train()
    shuffled = _sample_prompt_augmentations(
        samples,
        seed=config.seed,
        epoch=epoch,
    )
    random.Random(config.seed + epoch).shuffle(shuffled)
    losses: list[float] = []
    for batch in _batches(shuffled, config.batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss, _outputs = _batch_loss_and_outputs(model, objective, batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _sample_prompt_augmentations(
    samples: list[_Sample],
    *,
    seed: int,
    epoch: int,
) -> list[_Sample]:
    if epoch <= 0:
        raise ValueError("prompt augmentation epoch must be positive")
    grouped: dict[str, list[_Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.sample_id, []).append(sample)
    selected: list[_Sample] = []
    prompt_counts: set[int] = set()
    for sample_id in sorted(grouped):
        prompt_rows = sorted(grouped[sample_id], key=lambda sample: sample.prompt_id)
        if len({sample.prompt_id for sample in prompt_rows}) != len(prompt_rows):
            raise ValueError(f"sample {sample_id} has duplicate prompt rows")
        if len({sample.label_id for sample in prompt_rows}) != 1:
            raise ValueError(f"sample {sample_id} prompt rows disagree on the A/C label")
        prompt_counts.add(len(prompt_rows))
        base = int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest(), 16)
        prompt_index = (base + epoch - 1) % len(prompt_rows)
        selected.append(prompt_rows[prompt_index])
    if len(prompt_counts) != 1:
        raise ValueError("training samples must have synchronized prompt counts")
    return selected


def _evaluate(
    model: nn.Module,
    objective: ProxyAnchorLoss | None,
    samples: list[_Sample],
    *,
    config: TrainingConfig,
) -> tuple[float, float]:
    model.eval()
    if objective is not None:
        objective.eval()
    losses: list[float] = []
    metric_samples: list[_Sample] = []
    metric_outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in _batches(samples, config.batch_size):
            loss, outputs = _batch_loss_and_outputs(model, objective, batch)
            losses.append(float(loss))
            metric_samples.extend(batch)
            metric_outputs.append(outputs)
    _sample_ids, labels, aggregate = _aggregate_sample_outputs(
        metric_samples,
        torch.cat(metric_outputs, dim=0),
        normalize=objective is not None,
    )
    predictions = _sample_level_predictions(aggregate, objective=objective)
    prediction_values = [int(value) for value in predictions.detach().cpu().numpy()]
    return float(np.mean(losses)), _balanced_accuracy(labels, prediction_values)


def _batch_loss_and_outputs(
    model: nn.Module,
    objective: ProxyAnchorLoss | None,
    batch: list[_Sample],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    trajectories, labels = _load_trajectory_batch(batch, device=device)
    if objective is not None:
        sample_ids = [sample.sample_id for sample in batch]
        _condition_z, relation_r = model(trajectories, sample_ids=sample_ids)
        loss = objective(relation_r, labels, sample_ids=sample_ids)
        return loss, relation_r
    logits = model(trajectories)
    return F.cross_entropy(logits, labels), logits


def _aggregate_sample_outputs(
    samples: list[Any],
    outputs: torch.Tensor,
    *,
    normalize: bool,
) -> tuple[list[str], list[int], torch.Tensor]:
    if outputs.ndim != 2 or outputs.shape[0] != len(samples):
        raise ValueError("validation outputs must match prompt rows")
    order: list[str] = []
    labels: dict[str, int] = {}
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for sample, output in zip(samples, outputs, strict=True):
        sample_id = str(sample.sample_id)
        label_id = int(sample.label_id)
        if sample_id not in sums:
            order.append(sample_id)
            labels[sample_id] = label_id
            sums[sample_id] = output.clone()
            counts[sample_id] = 1
            continue
        if labels[sample_id] != label_id:
            raise ValueError("all prompts for a sample_id must share one A/C label")
        sums[sample_id] = sums[sample_id] + output
        counts[sample_id] += 1
    prompt_counts = set(counts.values())
    if len(prompt_counts) != 1:
        raise ValueError("validation samples must have synchronized prompt counts")
    aggregate = torch.stack([sums[sample_id] / counts[sample_id] for sample_id in order])
    if normalize:
        norms = torch.linalg.vector_norm(aggregate, dim=-1)
        if bool((norms <= 1e-12).any()):
            raise ValueError("sample-level relation aggregate cannot have zero norm")
        aggregate = aggregate / norms.unsqueeze(-1)
    return order, [labels[sample_id] for sample_id in order], aggregate


def _sample_level_predictions(
    aggregate: torch.Tensor,
    *,
    objective: ProxyAnchorLoss | None,
) -> torch.Tensor:
    if objective is None:
        return aggregate.argmax(dim=-1)
    similarities = aggregate @ objective.normalized_proxies().T
    return similarities.argmax(dim=-1)


def _balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    recalls = []
    for label in (0, 1):
        indexes = [index for index, value in enumerate(labels) if value == label]
        if not indexes:
            raise ValueError("validation must contain both A/C labels")
        recalls.append(sum(predictions[index] == label for index in indexes) / len(indexes))
    return float(sum(recalls) / len(recalls))


def _checkpoint_payload(
    *,
    model: nn.Module,
    objective: ProxyAnchorLoss | None,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    input_dim: int,
    layer_count: int,
    signature: str,
    epoch: int,
    best_score: float,
    best_epoch: int,
    stale_epochs: int,
) -> dict[str, Any]:
    return {
        "schema": "mprisk_representation_checkpoint_v2",
        "repr_key": config.repr_key,
        "architecture_version": (
            TME_ARCHITECTURE_V1 if config.repr_key == TME_PROXY_ANCHOR_V1 else config.repr_key
        ),
        "model_key": config.model_key,
        "selection_metric": "val_balanced_accuracy_ac",
        "selection_unit": "sample_id",
        "model_config": {"input_dim": input_dim, "layer_count": layer_count},
        "training_config": asdict(config),
        "training_signature": signature,
        "model_state_dict": model.state_dict(),
        "proxy_state_dict": objective.state_dict() if objective is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
    }


def _trajectory_shape(samples: list[_Sample]) -> tuple[int, int]:
    entry = samples[0].condition_entries[0]
    return int(entry.layer_count), int(entry.hidden_dim)


def _batches(samples: list[_Sample], batch_size: int) -> list[list[_Sample]]:
    return [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]


def _training_signature(dataset_path: str | Path, config: TrainingConfig) -> str:
    config_payload = asdict(config)
    config_payload.pop("max_epochs")
    payload = {
        "dataset_sha256": _sha256(Path(dataset_path)),
        "config": config_payload,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_config(config: TrainingConfig) -> None:
    if config.repr_key not in REPRESENTATION_KEYS:
        raise ValueError(f"repr_key must be one of {', '.join(REPRESENTATION_KEYS)}")
    if not config.model_key:
        raise ValueError("model_key is required")
    integer_fields = (
        config.hidden_dim,
        config.condition_dim,
        config.relation_dim,
        config.max_epochs,
        config.batch_size,
        config.patience,
    )
    if any(value <= 0 for value in integer_fields):
        raise ValueError("training dimensions/counts must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout is out of range")
    if config.lr <= 0.0 or config.weight_decay < 0.0 or config.min_delta < 0.0:
        raise ValueError("optimizer and stopping values are out of range")


def _set_deterministic_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    return resolved


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
