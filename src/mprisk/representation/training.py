"""Training and frozen export for sample-level relation representations."""

from __future__ import annotations

import hashlib
import json
import math
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
from mprisk.representation.losses import ModalitySplitRankingLoss, ProxyAnchorLoss
from mprisk.representation.relation_dataset import CONDITIONS, _reject_forbidden_fields
from mprisk.representation.relation_models import (
    REPRESENTATION_KEYS,
    SINGLE_POINT_BINARY_V1,
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
    TRAJECTORY_MLP_BINARY_V1,
    build_representation_model,
    strict_l2_normalize,
)
from mprisk.utils.io import write_json

TRAINING_CONFIG_SCHEMA = "mprisk_representation_training_v4"
REGISTERED_SPLITS = frozenset(
    {"relation_train", "relation_val", "aligned_calibration", "official_test"}
)


@dataclass(frozen=True)
class TrainingConfig:
    repr_key: str
    model_key: str
    protocol: str
    classification_objective: str
    prompt_set_key: str = ""
    prompt_set_artifact_sha256: str = ""
    expected_prompt_count: int = 8
    expected_prompt_ids: tuple[str, ...] = ()
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
    d_supervision_weight: float = 0.0
    d_ranking_margin: float = 0.0
    angular_supervision_weight: float = 0.0
    angular_ranking_margin_rad: float = 0.0
    d_aux_samples_per_class: int = 0
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
            raise ValueError(f"TME architecture_version must be {TME_ARCHITECTURE_V1}")
    elif architecture_version is not None and architecture_version != payload.get("repr_key"):
        raise ValueError("baseline architecture_version must match repr_key when provided")
    unknown = set(payload) - set(TrainingConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown training config fields: {', '.join(sorted(unknown))}")
    if isinstance(payload.get("expected_prompt_ids"), list):
        payload["expected_prompt_ids"] = tuple(payload["expected_prompt_ids"])
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
        _validate_checkpoint_architecture(resume_payload)
        if resume_payload.get("training_signature") != signature:
            raise ValueError("resume signature mismatch")
    rows = _read_relation_rows(
        dataset_path,
        expected_model_key=config.model_key,
        expected_protocol=config.protocol,
        expected_prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
    )
    split_contract = _validate_registered_splits(rows)
    training_rows = [
        row for row in rows if row["representation_split"] in {"relation_train", "relation_val"}
    ]
    samples = _rows_to_sample_refs(training_rows)
    _validate_prompt_contract(samples, config=config)
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
    d_objective: ModalitySplitRankingLoss | None = None
    parameters: list[nn.Parameter] = list(model.parameters())
    if config.repr_key == TME_PROXY_ANCHOR_V1:
        objective = ProxyAnchorLoss(
            embed_dim=config.relation_dim,
            num_classes=2,
            alpha=config.proxy_alpha,
            margin=config.proxy_margin,
        ).to(torch_device)
        d_objective = ModalitySplitRankingLoss(
            d_margin=config.d_ranking_margin,
            angular_margin_rad=config.angular_ranking_margin_rad,
        ).to(torch_device)
        parameters.extend(objective.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    class_weights = _baseline_class_weights(
        train_samples,
        config=config,
        device=torch_device,
    )
    train_label_counts = _sample_label_counts(train_samples)

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
    best_validation_state_separation: dict[str, float] | None = None
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
        best_validation_state_separation = checkpoint.get(
            "best_validation_state_separation"
        )
    else:
        log_path.write_text("", encoding="utf-8")

    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=True), encoding="utf-8")
    stop_reason = "max_epochs"
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.max_epochs + 1):
        final_epoch = epoch
        train_metrics = _train_epoch(
            model,
            objective,
            d_objective,
            optimizer,
            train_samples,
            config=config,
            epoch=epoch,
            class_weights=class_weights,
        )
        val_loss, val_score, val_state_separation = _evaluate(
            model,
            objective,
            d_objective,
            val_samples,
            config=config,
            class_weights=class_weights,
        )
        improved = val_score > best_score + config.min_delta
        if improved:
            best_score = val_score
            best_epoch = epoch
            best_validation_state_separation = val_state_separation
            stale_epochs = 0
        else:
            stale_epochs += 1
        log_row = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_loss,
            "val_balanced_accuracy_ac": val_score,
            "val_state_separation": val_state_separation,
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
            best_validation_state_separation=best_validation_state_separation,
            class_weights=class_weights,
            train_label_counts=train_label_counts,
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
        "schema": "mprisk_representation_training_metrics_v3",
        "repr_key": config.repr_key,
        "model_key": config.model_key,
        "selection_metric": "val_balanced_accuracy_ac",
        "selection_unit": "sample_id",
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy_ac": best_score,
        "best_validation_state_separation": best_validation_state_separation,
        "final_epoch": final_epoch,
        "stop_reason": stop_reason,
        "train_rows": len(train_samples),
        "val_rows": len(val_samples),
        "train_sample_count": len({sample.sample_id for sample in train_samples}),
        "val_sample_count": len({sample.sample_id for sample in val_samples}),
        "train_examples_per_epoch": len({sample.sample_id for sample in train_samples}),
        "prompt_augmentation": "one_deterministic_prompt_per_sample_per_epoch",
        "state_supervision": (
            {
                "definition": "full_prompt_exact_D_detached_denominator_plus_raw_angle_ranking",
                "prompt_count": config.expected_prompt_count,
                "samples_per_class_per_step": config.d_aux_samples_per_class,
                "d_weight": config.d_supervision_weight,
                "d_margin": config.d_ranking_margin,
                "angular_weight": config.angular_supervision_weight,
                "angular_margin_rad": config.angular_ranking_margin_rad,
                "angular_margin_deg": math.degrees(config.angular_ranking_margin_rad),
            }
            if config.repr_key == TME_PROXY_ANCHOR_V1
            else None
        ),
        "classification_objective": config.classification_objective,
        "train_sample_label_counts": train_label_counts,
        "baseline_class_weights": (
            [float(value) for value in class_weights.detach().cpu().tolist()]
            if class_weights is not None
            else None
        ),
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
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    _validate_checkpoint_architecture(checkpoint)
    if checkpoint.get("repr_key") != TME_PROXY_ANCHOR_V1:
        raise ValueError(
            "condition z and relation r export requires a tme_proxy_anchor_v1 checkpoint"
        )
    config = TrainingConfig(**checkpoint["training_config"])
    _validate_config(config)
    rows = _read_relation_rows(
        dataset_path,
        expected_model_key=config.model_key,
        expected_protocol=config.protocol,
        expected_prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
    )
    samples = _rows_to_sample_refs(rows)
    _validate_prompt_contract(samples, config=config)
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
        encoder_checkpoint_sha256=_sha256(checkpoint_file),
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
            "prompt_set_key": config.prompt_set_key,
            "prompt_set_artifact_sha256": config.prompt_set_artifact_sha256,
            "encoder_checkpoint_sha256": _sha256(checkpoint_file),
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
    _validate_checkpoint_architecture(checkpoint)
    repr_key = str(checkpoint.get("repr_key", ""))
    if repr_key not in {SINGLE_POINT_BINARY_V1, TRAJECTORY_MLP_BINARY_V1}:
        raise ValueError("baseline export requires a Single-Point or Trajectory MLP checkpoint")
    if checkpoint.get("proxy_state_dict") is not None:
        raise ValueError("baseline checkpoints must not contain Proxy Anchor state")
    config = TrainingConfig(**checkpoint["training_config"])
    _validate_config(config)
    rows = _read_relation_rows(
        dataset_path,
        expected_model_key=config.model_key,
        expected_protocol=config.protocol,
        expected_prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
    )
    _validate_registered_splits(rows)
    selected_rows = [row for row in rows if row["representation_split"] == representation_split]
    if not selected_rows:
        raise ValueError(f"relation dataset has no rows for {representation_split}")
    samples = sorted(
        _rows_to_sample_refs(selected_rows),
        key=lambda sample: (sample.sample_id, sample.prompt_id),
    )
    _validate_prompt_contract(samples, config=config)
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
        prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
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
            "prompt_set_key": config.prompt_set_key,
            "prompt_set_artifact_sha256": config.prompt_set_artifact_sha256,
            "repr_key": repr_key,
            "representation_split": representation_split,
            "aggregation": "mean_over_synchronized_prompts",
            "feature_definition": _baseline_feature_definition(repr_key),
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
    prompt_set_artifact_sha256: str,
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
                        prompt_set_artifact_sha256=prompt_set_artifact_sha256,
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
                prompt_set_artifact_sha256=prompt_set_artifact_sha256,
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
    prompt_set_artifact_sha256: str,
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
        "prompt_set_artifact_sha256": prompt_set_artifact_sha256,
        "aggregation": "mean_over_synchronized_prompts",
        "feature_definition": _baseline_feature_definition(repr_key),
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
    encoder_checkpoint_sha256: str,
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
                    encoder_checkpoint_sha256=encoder_checkpoint_sha256,
                    prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
                )
                manifest_handle.write(json.dumps(row, sort_keys=True) + "\n")
                if current_bundle is None or current_bundle["sample_id"] != sample.sample_id:
                    if current_bundle is not None:
                        _finalize_frozen_bundle(current_bundle)
                        bundle_handle.write(json.dumps(current_bundle, sort_keys=True) + "\n")
                    current_bundle = _empty_frozen_bundle(row)
                    sample_count += 1
                _append_frozen_row(current_bundle, row)
        if current_bundle is not None:
            _finalize_frozen_bundle(current_bundle)
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
    encoder_checkpoint_sha256: str,
    prompt_set_artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "mprisk_frozen_spherical_representation_v1",
        "row_id": sample.row_id,
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "label_id": sample.label_id,
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
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "prompt_set_artifact_sha256": prompt_set_artifact_sha256,
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
            "label_id",
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
            "encoder_checkpoint_sha256",
            "prompt_set_artifact_sha256",
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


def _finalize_frozen_bundle(bundle: dict[str, Any]) -> None:
    relations = bundle["relations"]
    if not relations:
        raise ValueError("frozen TME sample aggregate is empty")
    relation_rows = torch.tensor(list(relations.values()), dtype=torch.float32)
    mean_relation = relation_rows.mean(dim=0, keepdim=True)
    normalized = strict_l2_normalize(
        mean_relation,
        stage="tme_sample_relation_aggregate",
        sample_ids=[str(bundle["sample_id"])],
    )[0]
    bundle["sample_relation_feature"] = _vector_values(normalized)
    bundle["prompt_count"] = len(relations)
    bundle["aggregation"] = "mean_over_synchronized_prompts_then_l2"
    bundle["feature_definition"] = "unit_normalized_mean_prompt_ordered_relation_r"


def _baseline_feature_definition(repr_key: str) -> str:
    if repr_key == SINGLE_POINT_BINARY_V1:
        return "mean_prompt_final_layer_m1_m2_m12_concat"
    if repr_key == TRAJECTORY_MLP_BINARY_V1:
        return "mean_prompt_first_linear_gelu_hidden"
    raise ValueError(f"unsupported baseline representation: {repr_key}")


def _validate_prompt_contract(samples: list[_Sample], *, config: TrainingConfig) -> None:
    grouped: dict[str, list[str]] = {}
    for sample in samples:
        if sample.prompt_set_key != config.prompt_set_key:
            raise ValueError(
                f"sample {sample.sample_id} prompt_set_key does not match training config"
            )
        grouped.setdefault(sample.sample_id, []).append(sample.prompt_id)
    expected_prompt_ids = set(config.expected_prompt_ids)
    for sample_id in sorted(grouped):
        prompt_ids = grouped[sample_id]
        unique_prompt_ids = set(prompt_ids)
        if len(unique_prompt_ids) != len(prompt_ids):
            raise ValueError(f"sample {sample_id} has duplicate prompt rows")
        if len(prompt_ids) != config.expected_prompt_count:
            raise ValueError(
                f"sample {sample_id} must have exactly {config.expected_prompt_count} prompts; "
                f"found {len(prompt_ids)}"
            )
        if unique_prompt_ids != expected_prompt_ids:
            raise ValueError(
                f"sample {sample_id} prompt IDs do not match the configured prompt set"
            )


def _validate_checkpoint_architecture(checkpoint: dict[str, Any]) -> None:
    repr_key = str(checkpoint.get("repr_key", ""))
    architecture_version = str(checkpoint.get("architecture_version", ""))
    expected_architecture = TME_ARCHITECTURE_V1 if repr_key == TME_PROXY_ANCHOR_V1 else repr_key
    if architecture_version != expected_architecture:
        raise ValueError("checkpoint architecture_version does not match its representation")
    if repr_key != SINGLE_POINT_BINARY_V1:
        return
    model_state = checkpoint.get("model_state_dict")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_state, dict) or not isinstance(model_config, dict):
        raise ValueError("Single-Point checkpoint architecture drift: metadata is incomplete")
    input_dim = model_config.get("input_dim")
    weight = model_state.get("classifier.weight")
    bias = model_state.get("classifier.bias")
    if (
        not isinstance(input_dim, int)
        or input_dim <= 0
        or set(model_state) != {"classifier.weight", "classifier.bias"}
        or not isinstance(weight, torch.Tensor)
        or tuple(weight.shape) != (2, 3 * input_dim)
        or not isinstance(bias, torch.Tensor)
        or tuple(bias.shape) != (2,)
    ):
        raise ValueError(
            "Single-Point checkpoint architecture drift: expected direct Linear(3H, 2)"
        )


def _vector_values(vector: torch.Tensor) -> list[float]:
    return [float(value) for value in vector.detach().cpu().numpy()]


def _read_relation_rows(
    path: str | Path,
    *,
    expected_model_key: str,
    expected_protocol: str,
    expected_prompt_set_artifact_sha256: str,
) -> list[dict[str, Any]]:
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
            if str(row.get("protocol", "")).lower() != expected_protocol.lower():
                raise ValueError("relation dataset protocol does not match training config")
            if row.get("prompt_set_artifact_sha256") != expected_prompt_set_artifact_sha256:
                raise ValueError(
                    "relation dataset prompt artifact SHA does not match training config"
                )
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
    labels = torch.tensor([sample.label_id for sample in batch], dtype=torch.long, device=device)
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
    d_objective: ModalitySplitRankingLoss | None,
    optimizer: torch.optim.Optimizer,
    samples: list[_Sample],
    *,
    config: TrainingConfig,
    epoch: int,
    class_weights: torch.Tensor | None,
) -> dict[str, float]:
    model.train()
    if objective is not None:
        objective.train()
    shuffled = _sample_prompt_augmentations(
        samples,
        seed=config.seed,
        epoch=epoch,
    )
    random.Random(config.seed + epoch).shuffle(shuffled)
    proxy_batches = _batches(shuffled, config.batch_size)
    d_batches = (
        _class_balanced_full_prompt_batches(
            samples,
            batch_count=len(proxy_batches),
            samples_per_class=config.d_aux_samples_per_class,
            seed=config.seed,
            epoch=epoch,
        )
        if d_objective is not None
        else [None] * len(proxy_batches)
    )
    total_losses: list[float] = []
    proxy_losses: list[float] = []
    d_losses: list[float] = []
    angular_losses: list[float] = []
    d_values: list[torch.Tensor] = []
    angle_values: list[torch.Tensor] = []
    d_labels: list[torch.Tensor] = []
    for batch, d_batch in zip(proxy_batches, d_batches, strict=True):
        optimizer.zero_grad(set_to_none=True)
        proxy_loss, _outputs = _batch_loss_and_outputs(
            model, objective, batch, class_weights=class_weights
        )
        proxy_loss.backward()
        total_loss_value = float(proxy_loss.detach())
        proxy_losses.append(float(proxy_loss.detach()))
        if d_objective is not None:
            if d_batch is None:
                raise AssertionError("TME D supervision batch was not constructed")
            grouped_z, grouped_labels, grouped_sample_ids = _encode_prompt_groups(
                model,
                d_batch,
            )
            d_loss, angular_loss, diagnostics = d_objective(
                grouped_z,
                grouped_labels,
                sample_ids=grouped_sample_ids,
            )
            auxiliary_loss = (
                config.d_supervision_weight * d_loss
                + config.angular_supervision_weight * angular_loss
            )
            auxiliary_loss.backward()
            total_loss_value += float(auxiliary_loss.detach())
            d_losses.append(float(d_loss.detach()))
            angular_losses.append(float(angular_loss.detach()))
            d_values.append(diagnostics["D"].detach())
            angle_values.append(diagnostics["split_angle_rad"].detach())
            d_labels.append(grouped_labels.detach())
        optimizer.step()
        total_losses.append(total_loss_value)
    metrics = {
        "train_loss": float(np.mean(total_losses)),
        "train_proxy_anchor_loss": float(np.mean(proxy_losses)),
    }
    if d_objective is not None:
        metrics.update(
            {
                "train_d_ranking_loss": float(np.mean(d_losses)),
                "train_angular_ranking_loss": float(np.mean(angular_losses)),
                **_state_separation_summary(
                    torch.cat(d_values),
                    torch.cat(angle_values),
                    torch.cat(d_labels),
                    d_margin=config.d_ranking_margin,
                    angular_margin_rad=config.angular_ranking_margin_rad,
                    prefix="train",
                ),
            }
        )
    return metrics


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


def _class_balanced_full_prompt_batches(
    samples: list[_Sample],
    *,
    batch_count: int,
    samples_per_class: int,
    seed: int,
    epoch: int,
) -> list[list[_Sample]]:
    if batch_count <= 0 or samples_per_class <= 0:
        raise ValueError("D supervision batch counts must be positive")
    grouped: dict[str, list[_Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.sample_id, []).append(sample)
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for sample_id, prompt_rows in grouped.items():
        labels = {sample.label_id for sample in prompt_rows}
        if len(labels) != 1:
            raise ValueError(f"sample {sample_id} prompt rows disagree on the A/C label")
        by_label[next(iter(labels))].append(sample_id)
    if any(len(sample_ids) < samples_per_class for sample_ids in by_label.values()):
        raise ValueError("D supervision requires enough samples in both A/C classes")
    for label in (0, 1):
        by_label[label].sort()
        random.Random(seed + epoch * 104729 + label).shuffle(by_label[label])

    batches: list[list[_Sample]] = []
    offsets = {0: 0, 1: 0}
    for _batch_index in range(batch_count):
        selected_ids: list[str] = []
        for label in (0, 1):
            class_ids = by_label[label]
            for _ in range(samples_per_class):
                selected_ids.append(class_ids[offsets[label] % len(class_ids)])
                offsets[label] += 1
        batch_rows: list[_Sample] = []
        for sample_id in selected_ids:
            batch_rows.extend(sorted(grouped[sample_id], key=lambda row: row.prompt_id))
        batches.append(batch_rows)
    return batches


def _encode_prompt_groups(
    model: nn.Module,
    samples: list[_Sample],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    device = next(model.parameters()).device
    trajectories, _row_labels = _load_trajectory_batch(samples, device=device)
    row_sample_ids = [sample.sample_id for sample in samples]
    condition_z, _relation_r = model(trajectories, sample_ids=row_sample_ids)
    return _group_prompt_condition_z(samples, condition_z)


def _group_prompt_condition_z(
    samples: list[_Sample],
    condition_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if condition_z.ndim != 3 or condition_z.shape[:2] != (len(samples), 3):
        raise ValueError("condition_z rows must match [prompt_row, 3, condition_dim]")
    grouped: dict[str, list[tuple[str, torch.Tensor]]] = {}
    labels: dict[str, int] = {}
    order: list[str] = []
    for sample, row_z in zip(samples, condition_z, strict=True):
        if sample.sample_id not in grouped:
            grouped[sample.sample_id] = []
            labels[sample.sample_id] = sample.label_id
            order.append(sample.sample_id)
        elif labels[sample.sample_id] != sample.label_id:
            raise ValueError("prompt rows disagree on the A/C label")
        grouped[sample.sample_id].append((sample.prompt_id, row_z))
    prompt_counts = {len(rows) for rows in grouped.values()}
    if len(prompt_counts) != 1 or next(iter(prompt_counts), 0) < 2:
        raise ValueError("D supervision requires synchronized multi-prompt sample groups")
    grouped_z = torch.stack(
        [
            torch.stack([row_z for _prompt_id, row_z in sorted(grouped[sample_id])])
            for sample_id in order
        ]
    )
    grouped_labels = torch.tensor(
        [labels[sample_id] for sample_id in order],
        dtype=torch.long,
        device=condition_z.device,
    )
    return grouped_z, grouped_labels, order


def _state_separation_summary(
    d_values: torch.Tensor,
    split_angles_rad: torch.Tensor,
    labels: torch.Tensor,
    *,
    d_margin: float,
    angular_margin_rad: float,
    prefix: str,
) -> dict[str, float]:
    if d_values.ndim != 1 or split_angles_rad.shape != d_values.shape:
        raise ValueError("state separation diagnostics require aligned one-dimensional values")
    aligned = labels == 0
    conflict = labels == 1
    if not bool(aligned.any()) or not bool(conflict.any()):
        raise ValueError("state separation diagnostics require both A/C classes")
    d_aligned = d_values[aligned]
    d_conflict = d_values[conflict]
    angle_aligned = split_angles_rad[aligned]
    angle_conflict = split_angles_rad[conflict]
    d_gaps = d_conflict[:, None] - d_aligned[None, :]
    angle_gaps = angle_conflict[:, None] - angle_aligned[None, :]
    degrees = 180.0 / math.pi
    return {
        f"{prefix}_aligned_D_mean": float(d_aligned.mean()),
        f"{prefix}_conflict_D_mean": float(d_conflict.mean()),
        f"{prefix}_D_gap": float(d_conflict.mean() - d_aligned.mean()),
        f"{prefix}_D_effect_size": _pooled_effect_size(d_aligned, d_conflict),
        f"{prefix}_D_pair_margin_satisfaction": float((d_gaps >= d_margin).float().mean()),
        f"{prefix}_aligned_split_angle_deg_mean": float(angle_aligned.mean() * degrees),
        f"{prefix}_conflict_split_angle_deg_mean": float(angle_conflict.mean() * degrees),
        f"{prefix}_split_angle_gap_deg": float(
            (angle_conflict.mean() - angle_aligned.mean()) * degrees
        ),
        f"{prefix}_split_angle_effect_size": _pooled_effect_size(
            angle_aligned, angle_conflict
        ),
        f"{prefix}_angular_pair_margin_satisfaction": float(
            (angle_gaps >= angular_margin_rad).float().mean()
        ),
    }


def _pooled_effect_size(aligned: torch.Tensor, conflict: torch.Tensor) -> float:
    pooled_scale = torch.sqrt(
        (aligned.var(unbiased=False) + conflict.var(unbiased=False)) / 2.0
    )
    if float(pooled_scale) <= 1e-12:
        return 0.0
    return float((conflict.mean() - aligned.mean()) / pooled_scale)


def _evaluate(
    model: nn.Module,
    objective: ProxyAnchorLoss | None,
    d_objective: ModalitySplitRankingLoss | None,
    samples: list[_Sample],
    *,
    config: TrainingConfig,
    class_weights: torch.Tensor | None,
) -> tuple[float, float, dict[str, float] | None]:
    model.eval()
    if objective is not None:
        objective.eval()
    losses: list[float] = []
    metric_samples: list[_Sample] = []
    metric_outputs: list[torch.Tensor] = []
    condition_outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in _batches(samples, config.batch_size):
            if objective is not None:
                device = next(model.parameters()).device
                trajectories, labels = _load_trajectory_batch(batch, device=device)
                sample_ids = [sample.sample_id for sample in batch]
                condition_z, outputs = model(trajectories, sample_ids=sample_ids)
                loss = objective(outputs, labels, sample_ids=sample_ids)
                condition_outputs.append(condition_z)
            else:
                loss, outputs = _batch_loss_and_outputs(
                    model, objective, batch, class_weights=class_weights
                )
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
    state_separation = None
    if d_objective is not None:
        grouped_z, grouped_labels, grouped_sample_ids = _group_prompt_condition_z(
            metric_samples,
            torch.cat(condition_outputs),
        )
        _d_loss, _angular_loss, diagnostics = d_objective(
            grouped_z,
            grouped_labels,
            sample_ids=grouped_sample_ids,
        )
        state_separation = _state_separation_summary(
            diagnostics["D"],
            diagnostics["split_angle_rad"],
            grouped_labels,
            d_margin=config.d_ranking_margin,
            angular_margin_rad=config.angular_ranking_margin_rad,
            prefix="val",
        )
    return (
        float(np.mean(losses)),
        _balanced_accuracy(labels, prediction_values),
        state_separation,
    )


def _batch_loss_and_outputs(
    model: nn.Module,
    objective: ProxyAnchorLoss | None,
    batch: list[_Sample],
    *,
    class_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    trajectories, labels = _load_trajectory_batch(batch, device=device)
    if objective is not None:
        if class_weights is not None:
            raise ValueError("TME Proxy Anchor must not receive cross-entropy class weights")
        sample_ids = [sample.sample_id for sample in batch]
        _condition_z, relation_r = model(trajectories, sample_ids=sample_ids)
        loss = objective(relation_r, labels, sample_ids=sample_ids)
        return loss, relation_r
    logits = model(trajectories)
    if class_weights is None:
        raise ValueError("baseline cross-entropy requires pre-registered class weights")
    return F.cross_entropy(logits, labels, weight=class_weights), logits


def _baseline_class_weights(
    samples: list[_Sample],
    *,
    config: TrainingConfig,
    device: torch.device,
) -> torch.Tensor | None:
    if config.repr_key == TME_PROXY_ANCHOR_V1:
        if config.classification_objective != "proxy_anchor_only":
            raise ValueError("TME classification_objective must be proxy_anchor_only")
        return None
    if config.classification_objective != "inverse_frequency_cross_entropy":
        raise ValueError(
            "baseline classification_objective must be inverse_frequency_cross_entropy"
        )
    counts_by_label = _sample_label_counts(samples)
    counts = [counts_by_label["Aligned"], counts_by_label["Conflict"]]
    if any(count <= 0 for count in counts):
        raise ValueError("inverse-frequency baseline weights require both A/C classes")
    total = sum(counts)
    return torch.tensor(
        [total / (2.0 * count) for count in counts],
        dtype=torch.float32,
        device=device,
    )


def _sample_label_counts(samples: list[_Sample]) -> dict[str, int]:
    labels_by_sample: dict[str, int] = {}
    for sample in samples:
        previous = labels_by_sample.setdefault(sample.sample_id, sample.label_id)
        if previous != sample.label_id:
            raise ValueError("training prompts disagree on the sample-level A/C label")
    return {
        "Aligned": sum(label == 0 for label in labels_by_sample.values()),
        "Conflict": sum(label == 1 for label in labels_by_sample.values()),
    }


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
    best_validation_state_separation: dict[str, float] | None,
    class_weights: torch.Tensor | None,
    train_label_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema": "mprisk_representation_checkpoint_v3",
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
        "best_validation_state_separation": best_validation_state_separation,
        "classification_objective": config.classification_objective,
        "train_sample_label_counts": dict(train_label_counts),
        "baseline_class_weights": (
            [float(value) for value in class_weights.detach().cpu().tolist()]
            if class_weights is not None
            else None
        ),
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
    if config.protocol not in {"vt", "va", "vta"}:
        raise ValueError("protocol must be one of vt, va, or vta")
    expected_objective = (
        "proxy_anchor_only"
        if config.repr_key == TME_PROXY_ANCHOR_V1
        else "inverse_frequency_cross_entropy"
    )
    if config.classification_objective != expected_objective:
        raise ValueError(
            f"classification_objective for {config.repr_key} must be {expected_objective}"
        )
    if not config.prompt_set_key:
        raise ValueError("prompt_set_key is required")
    if len(config.prompt_set_artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in config.prompt_set_artifact_sha256
    ):
        raise ValueError("prompt_set_artifact_sha256 must be lowercase sha256")
    if config.expected_prompt_count <= 0:
        raise ValueError("expected_prompt_count must be positive")
    if (
        len(config.expected_prompt_ids) != config.expected_prompt_count
        or len(set(config.expected_prompt_ids)) != config.expected_prompt_count
        or any(not prompt_id for prompt_id in config.expected_prompt_ids)
    ):
        raise ValueError(
            "expected_prompt_ids must contain exactly expected_prompt_count unique IDs"
        )
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
    if config.repr_key == TME_PROXY_ANCHOR_V1:
        if config.d_supervision_weight <= 0.0 or config.angular_supervision_weight <= 0.0:
            raise ValueError("TME requires positive D and angular supervision weights")
        if config.d_ranking_margin < 0.0:
            raise ValueError("TME d_ranking_margin must be non-negative")
        if not 0.0 <= config.angular_ranking_margin_rad <= math.pi:
            raise ValueError("TME angular_ranking_margin_rad must be in [0, pi]")
        if config.d_aux_samples_per_class <= 0:
            raise ValueError("TME d_aux_samples_per_class must be positive")
    elif any(
        value != 0
        for value in (
            config.d_supervision_weight,
            config.d_ranking_margin,
            config.angular_supervision_weight,
            config.angular_ranking_margin_rad,
            config.d_aux_samples_per_class,
        )
    ):
        raise ValueError("D/angular supervision fields are TME-only")


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
