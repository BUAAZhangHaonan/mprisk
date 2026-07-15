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
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
    build_representation_model,
)
from mprisk.utils.io import write_json

TRAINING_CONFIG_SCHEMA = "mprisk_representation_training_v2"


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
    val_fraction: float = 0.2
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
class _Sample:
    row_id: str
    sample_id: str
    sample_type: str
    label_id: int
    split_group_id: str
    master_split: str
    calibration_split: str
    protocol: str
    prompt_set_key: str
    prompt_id: str
    trajectories: torch.Tensor


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
    samples = _rows_to_samples(rows)
    train_samples, val_samples = _master_group_split(samples, config=config)
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
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy_ac": best_score,
        "final_epoch": final_epoch,
        "stop_reason": stop_reason,
        "train_rows": len(train_samples),
        "val_rows": len(val_samples),
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
    samples = _rows_to_samples(rows)
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
    exported: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            condition_z, relation_r = model(sample.trajectories.unsqueeze(0))
            exported.append(
                {
                    "schema": "mprisk_frozen_spherical_representation_v1",
                    "row_id": sample.row_id,
                    "sample_id": sample.sample_id,
                    "sample_type": sample.sample_type,
                    "model_key": config.model_key,
                    "protocol": sample.protocol,
                    "prompt_set_key": sample.prompt_set_key,
                    "calibration_split": sample.calibration_split,
                    "prompt_id": sample.prompt_id,
                    "repr_key": config.repr_key,
                    "condition_z": {
                        condition: condition_z[0, index].cpu().tolist()
                        for index, condition in enumerate(CONDITIONS)
                    },
                    "relation_r": relation_r[0].cpu().tolist(),
                }
            )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "frozen_representations.jsonl"
    _atomic_jsonl(manifest_path, exported)
    bundle_rows = _frozen_bundle_rows(exported)
    bundle_manifest_path = output_root / "spherical_embedding_manifest.jsonl"
    _atomic_jsonl(bundle_manifest_path, bundle_rows)
    summary_path = write_json(
        output_root / "frozen_representation_summary.json",
        {
            "schema": "mprisk_frozen_spherical_representation_summary_v1",
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_path),
            "count": len(exported),
            "sample_count": len(bundle_rows),
            "bundle_manifest": str(bundle_manifest_path),
            "repr_key": config.repr_key,
            "model_key": config.model_key,
        },
    )
    return FrozenRepresentationExportResult(
        manifest_path, bundle_manifest_path, summary_path, len(exported)
    )


def _frozen_bundle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        bundle = grouped.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "sample_type": row["sample_type"],
                "model_key": row["model_key"],
                "protocol": row["protocol"],
                "prompt_set_key": row["prompt_set_key"],
                "calibration_split": row["calibration_split"],
                "repr_key": row["repr_key"],
                "embeddings": {condition: {} for condition in CONDITIONS},
                "relations": {},
            },
        )
        prompt_id = str(row["prompt_id"])
        for condition in CONDITIONS:
            bundle["embeddings"][condition][prompt_id] = row["condition_z"][condition]
        bundle["relations"][prompt_id] = row["relation_r"]
    return [grouped[sample_id] for sample_id in sorted(grouped)]


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


def _rows_to_samples(rows: list[dict[str, Any]]) -> list[_Sample]:
    samples: list[_Sample] = []
    expected_shape: tuple[int, int] | None = None
    for row in rows:
        trajectories = torch.stack(
            [
                torch.tensor(
                    extract_t0_trajectory(
                        prompt_conditioned_entry_from_row(row["conditions"][condition])
                        .to_hidden_state_entry()
                    ),
                    dtype=torch.float32,
                )
                for condition in CONDITIONS
            ]
        )
        shape = tuple(trajectories.shape[1:])
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
                calibration_split=str(row.get("calibration_split", "")),
                protocol=str(row["protocol"]),
                prompt_set_key=str(row["prompt_set_key"]),
                prompt_id=str(row["prompt_id"]),
                trajectories=trajectories,
            )
        )
    return samples


def _master_group_split(
    samples: list[_Sample], *, config: TrainingConfig
) -> tuple[list[_Sample], list[_Sample]]:
    groups: dict[str, list[_Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.split_group_id, []).append(sample)
    for group_samples in groups.values():
        if len({sample.label_id for sample in group_samples}) != 1:
            raise ValueError("split_group_id must not cross A/C labels")
        splits = {sample.master_split for sample in group_samples if sample.master_split}
        if len(splits) > 1:
            raise ValueError("split_group_id must not cross master splits")
    explicit = {sample.master_split for sample in samples if sample.master_split}
    if explicit:
        if not explicit <= {"train", "val"} or explicit != {"train", "val"}:
            raise ValueError("master_split must provide both train and val only")
        train = [sample for sample in samples if sample.master_split == "train"]
        val = [sample for sample in samples if sample.master_split == "val"]
    else:
        val_groups: set[str] = set()
        for label_id in (0, 1):
            label_groups = sorted(
                group
                for group, values in groups.items()
                if values[0].label_id == label_id
            )
            if len(label_groups) < 2:
                raise ValueError("each A/C class needs at least two split groups")
            count = max(1, round(len(label_groups) * config.val_fraction))
            ranked = sorted(
                label_groups,
                key=lambda group: hashlib.sha256(
                    f"{config.seed}:{group}".encode()
                ).hexdigest(),
            )
            val_groups.update(ranked[:count])
        train = [sample for sample in samples if sample.split_group_id not in val_groups]
        val = [sample for sample in samples if sample.split_group_id in val_groups]
    if {sample.label_id for sample in train} != {0, 1} or {sample.label_id for sample in val} != {
        0,
        1,
    }:
        raise ValueError("train and val must both contain Aligned and Conflict samples")
    return train, val


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
    shuffled = list(samples)
    random.Random(config.seed + epoch).shuffle(shuffled)
    losses: list[float] = []
    for batch in _batches(shuffled, config.batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss, _predictions = _batch_loss_and_predictions(model, objective, batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


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
    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in _batches(samples, config.batch_size):
            loss, predicted = _batch_loss_and_predictions(model, objective, batch)
            losses.append(float(loss))
            labels.extend(sample.label_id for sample in batch)
            predictions.extend(predicted.tolist())
    return float(np.mean(losses)), _balanced_accuracy(labels, predictions)


def _batch_loss_and_predictions(
    model: nn.Module,
    objective: ProxyAnchorLoss | None,
    batch: list[_Sample],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    trajectories = torch.stack([sample.trajectories for sample in batch]).to(device)
    labels = torch.tensor(
        [sample.label_id for sample in batch], dtype=torch.long, device=device
    )
    if objective is not None:
        _condition_z, relation_r = model(trajectories)
        loss = objective(relation_r, labels)
        similarities = F.normalize(relation_r, dim=-1) @ F.normalize(objective.proxies, dim=-1).T
        return loss, similarities.argmax(dim=-1)
    logits = model(trajectories)
    return F.cross_entropy(logits, labels), logits.argmax(dim=-1)


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
    shape = samples[0].trajectories.shape
    return int(shape[1]), int(shape[2])


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
    if not 0.0 <= config.dropout < 1.0 or not 0.0 < config.val_fraction < 0.5:
        raise ValueError("dropout or val_fraction is out of range")
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
