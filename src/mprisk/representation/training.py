"""Training utilities for learned trajectory representations."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.cache.prompt_conditioned_cache import prompt_conditioned_entry_from_row
from mprisk.representation.losses import combined_trajectory_loss
from mprisk.representation.trajectory_model import MLPProjection
from mprisk.utils.io import ensure_parent, write_json


@dataclass(frozen=True)
class TrainingConfig:
    embed_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    lambda_prompt: float = 0.5
    temperature: float = 0.07
    negative_budget_ratio: float = 0.5
    seed: int = 0

    @property
    def embedding_dim(self) -> int:
        return self.embed_dim


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    config_path: Path
    metrics_path: Path
    log_path: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _TrainingSample:
    row_id: str
    sample_id: str
    label: str
    label_id: int
    view_key: str
    prompt_id: str
    split_group_id: str
    trajectory: torch.Tensor


REPR_KEY = "tme_supcon_v1"
REPRESENTATION_DATASET_FIELDS = (
    "row_id",
    "sample_id",
    "sample_type",
    "model_key",
    "protocol",
    "view_key",
    "prompt_id",
    "prompt_set_key",
    "label",
    "specific_affect",
    "is_clear",
    "prompt_conditioned_state",
    "split_group_id",
    "source_dataset",
)


def load_training_config(path: str | Path) -> TrainingConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML mapping")
    if "embedding_dim" in payload and "embed_dim" not in payload:
        payload["embed_dim"] = payload.pop("embedding_dim")
    metadata_fields = {"schema", "key", "repr_key"}
    payload = {key: value for key, value in payload.items() if key not in metadata_fields}
    allowed = set(TrainingConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown training config fields: {', '.join(unknown)}")
    config = TrainingConfig(**payload)
    _validate_config(config)
    return config


def train_trajectory_encoder(
    *,
    dataset_path: str | Path,
    config: TrainingConfig,
    output_dir: str | Path,
) -> TrainingResult:
    _validate_config(config)
    _set_deterministic_seed(config.seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(dataset_path)
    label_to_id = _label_to_id(rows)
    samples = _rows_to_samples(rows, label_to_id=label_to_id)
    train_samples, val_samples = _split_samples(samples, seed=config.seed)
    input_dim = _trajectory_input_dim(samples)

    model = MLPProjection(
        input_dim=input_dim,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = _run_train_epoch(
            model,
            optimizer,
            train_samples,
            config=config,
            epoch=epoch,
        )
        model.eval()
        with torch.no_grad():
            val_loss = _evaluate_loss(model, val_samples, config=config)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_size": len(train_samples),
                "val_size": len(val_samples),
            }
        )

    model_config = {
        "input_dim": input_dim,
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "pooling": "mean",
        "normalize_output": True,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "repr_key": REPR_KEY,
        "label_to_id": label_to_id,
    }
    checkpoint_path = output_root / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    config_path = output_root / "train_config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=True), encoding="utf-8")

    metrics = {
        "repr_key": REPR_KEY,
        "epochs": config.epochs,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "final_train_loss": log_rows[-1]["train_loss"],
        "final_val_loss": log_rows[-1]["val_loss"],
        "label_to_id": label_to_id,
    }
    metrics_path = write_json(output_root / "train_metrics.json", metrics)

    log_path = ensure_parent(output_root / "train_log.jsonl")
    with log_path.open("w", encoding="utf-8") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        metrics_path=metrics_path,
        log_path=log_path,
        metrics=metrics,
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: representation row must be an object")
            _validate_representation_row(row, source=source, line_number=line_number)
            rows.append(row)
    if not rows:
        raise ValueError("representation dataset is empty")
    return rows


def _validate_representation_row(row: dict[str, Any], *, source: Path, line_number: int) -> None:
    missing = [field for field in REPRESENTATION_DATASET_FIELDS if field not in row]
    if missing:
        raise ValueError(f"{source}:{line_number}: missing fields: {', '.join(missing)}")


def _label_to_id(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels = sorted({str(row["label"]) for row in rows})
    if not labels:
        raise ValueError("representation dataset must contain at least one label")
    return {label: index for index, label in enumerate(labels)}


def _rows_to_samples(
    rows: list[dict[str, Any]],
    *,
    label_to_id: dict[str, int],
) -> list[_TrainingSample]:
    samples = []
    expected_shape: tuple[int, int] | None = None
    for row in rows:
        state_row = _state_row_from_representation_row(row)
        entry = prompt_conditioned_entry_from_row(state_row)
        trajectory = torch.tensor(
            extract_t0_trajectory(entry.to_hidden_state_entry()),
            dtype=torch.float32,
        )
        shape = tuple(trajectory.shape)
        if len(shape) != 2:
            raise ValueError(
                "prompt-conditioned trajectory must have shape [layer_count, hidden_dim]"
            )
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                "all training trajectories must have the same shape; "
                f"expected {expected_shape}, got {shape}"
            )
        label = str(row["label"])
        samples.append(
            _TrainingSample(
                row_id=str(row["row_id"]),
                sample_id=str(row["sample_id"]),
                label=label,
                label_id=label_to_id[label],
                view_key=str(row["view_key"]),
                prompt_id=str(row["prompt_id"]),
                split_group_id=str(row["split_group_id"]),
                trajectory=trajectory,
            )
        )
    return samples


def _state_row_from_representation_row(row: dict[str, Any]) -> dict[str, Any]:
    state = row["prompt_conditioned_state"]
    if isinstance(state, str):
        state = json.loads(state)
    if isinstance(state, dict):
        return state
    state_row = dict(row)
    state_row.setdefault("condition", row["view_key"])
    return state_row


def _split_samples(
    samples: list[_TrainingSample],
    *,
    seed: int,
) -> tuple[list[_TrainingSample], list[_TrainingSample]]:
    groups = sorted({sample.split_group_id for sample in samples})
    if len(groups) == 1:
        return samples, samples
    val_count = max(1, len(groups) // 5)
    ranked_groups = sorted(groups, key=lambda group: _stable_group_hash(group, seed=seed))
    val_groups = set(ranked_groups[:val_count])
    train = [sample for sample in samples if sample.split_group_id not in val_groups]
    val = [sample for sample in samples if sample.split_group_id in val_groups]
    if not train:
        train, val = samples, samples
    return train, val


def _stable_group_hash(group: str, *, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return int(digest, 16)


def _trajectory_input_dim(samples: list[_TrainingSample]) -> int:
    if not samples:
        raise ValueError("no training samples loaded")
    return int(samples[0].trajectory.shape[-1])


def _run_train_epoch(
    model: MLPProjection,
    optimizer: torch.optim.Optimizer,
    samples: list[_TrainingSample],
    *,
    config: TrainingConfig,
    epoch: int,
) -> float:
    shuffled = list(samples)
    random.Random(config.seed + epoch).shuffle(shuffled)
    losses: list[float] = []
    for batch in _batches(shuffled, config.batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, config=config)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _evaluate_loss(
    model: MLPProjection,
    samples: list[_TrainingSample],
    *,
    config: TrainingConfig,
) -> float | None:
    if not samples:
        return None
    losses = [
        float(_batch_loss(model, batch, config=config).detach().cpu())
        for batch in _batches(samples, config.batch_size)
    ]
    return float(np.mean(losses)) if losses else None


def _batch_loss(
    model: MLPProjection,
    batch: list[_TrainingSample],
    *,
    config: TrainingConfig,
) -> torch.Tensor:
    trajectories = torch.stack([sample.trajectory for sample in batch])
    embeddings = model(trajectories)
    labels = torch.tensor([sample.label_id for sample in batch], dtype=torch.long)
    return combined_trajectory_loss(
        embeddings,
        labels=labels,
        sample_ids=[sample.sample_id for sample in batch],
        view_keys=[sample.view_key for sample in batch],
        prompt_keys=[sample.prompt_id for sample in batch],
        prompt_weight=config.lambda_prompt,
        temperature=config.temperature,
        negative_budget_ratio=config.negative_budget_ratio,
    )


def _batches(samples: list[_TrainingSample], batch_size: int) -> list[list[_TrainingSample]]:
    return [samples[start : start + batch_size] for start in range(0, len(samples), batch_size)]


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_config(config: TrainingConfig) -> None:
    if config.embed_dim <= 0:
        raise ValueError("embed_dim must be positive")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    if config.lambda_prompt < 0.0:
        raise ValueError("lambda_prompt must be non-negative")
    if config.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= config.negative_budget_ratio <= 1.0:
        raise ValueError("negative_budget_ratio must be in [0.0, 1.0]")
