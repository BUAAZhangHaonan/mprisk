"""Checkpoint payload assembly helpers extracted from training.py.

These helpers build the on-disk checkpoint dictionary and derive the
selection metric / group checksum. They are pure functions that operate
on a ``TrainingConfig`` / ``_Sample`` and do not perform any I/O.

Round 1 of the P2-R1-C refactor: moved out of ``training.py`` unchanged.
``training.py`` still owns the callers; only the definitions live here now.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mprisk.representation.losses import ProxyAnchorLoss
from mprisk.representation.relation_models import (
    TME_ARCHITECTURE_LSTM_V1,
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
)
from mprisk.representation.training import TrainingConfig, _Sample

__all__ = [
    "_checkpoint_payload",
    "_selection_metric_name",
    "_group_checksum",
]


def _group_checksum(samples: list[_Sample]) -> str:
    groups = sorted({sample.split_group_id for sample in samples})
    return hashlib.sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest()


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
    unconstrained_best_score: float,
    unconstrained_best_epoch: int,
    unconstrained_best_validation_state_separation: dict[str, float] | None,
    checkpoint_feasibility: dict[str, Any],
    class_weights: torch.Tensor | None,
    train_label_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema": "mprisk_representation_checkpoint_v4",
        "repr_key": config.repr_key,
        "architecture_version": (
            (
                (
                    TME_ARCHITECTURE_LSTM_V1
                    if getattr(config, "encoder_type", "gru") == "lstm"
                    else TME_ARCHITECTURE_V1
                )
                if config.repr_key == TME_PROXY_ANCHOR_V1
                else config.repr_key
            )
        ),
        "model_key": config.model_key,
        "selection_metric": _selection_metric_name(config),
        "selection_unit": "sample_id",
        "checkpoint_role": "training_state",
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
        "unconstrained_best_score": unconstrained_best_score,
        "unconstrained_best_epoch": unconstrained_best_epoch,
        "unconstrained_best_validation_state_separation": (
            unconstrained_best_validation_state_separation
        ),
        "checkpoint_feasibility": checkpoint_feasibility,
        "classification_objective": config.classification_objective,
        "train_sample_label_counts": dict(train_label_counts),
        "baseline_class_weights": (
            [float(value) for value in class_weights.detach().cpu().tolist()]
            if class_weights is not None
            else None
        ),
    }


def _selection_metric_name(config: TrainingConfig) -> str:
    if config.repr_key == TME_PROXY_ANCHOR_V1 and config.enable_state_supervision:
        return "val_balanced_accuracy_ac_subject_to_state_feasibility"
    return "val_balanced_accuracy_ac"
