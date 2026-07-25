"""Training configuration dataclasses and validators for sample-level relation representations.

Extracted from ``training.py`` (P2-R1-A split). Holds the YAML-backed
:class:`TrainingConfig` schema, the result/export dataclasses, and the
``_validate_config``/``load_training_config`` pair that front-loads all
contract checks before training begins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mprisk.representation.relation_models import (
    REPRESENTATION_KEYS,
    TME_ARCHITECTURE_LSTM_V1,
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
)

__all__ = [
    "TRAINING_CONFIG_SCHEMA",
    "REGISTERED_SPLITS",
    "TrainingConfig",
    "TrainingResult",
    "FrozenRepresentationExportResult",
    "FrozenBaselineExportResult",
    "load_training_config",
]


TRAINING_CONFIG_SCHEMA = "mprisk_representation_training_v4"
# canonical_rerun_v2 (20260721): added "cross_domain_test" so
# _validate_registered_splits accepts ch_sims_v2 rows BEFORE the
# exclude_prefix filter runs (Stage B bug: validator failed early on
# ch_sims rows that legitimately carry this representation_split).
REGISTERED_SPLITS = frozenset(
    {
        "relation_train",
        "relation_val",
        "aligned_calibration",
        "official_test",
        "cross_domain_test",
    }
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
    # Encoder type for TME_PROXY_ANCHOR_V1: "gru" (SphericalTMEV1, default,
    # backward compatible) or "lstm" (SphericalTME_LSTM, multi-layer LSTM).
    # Ignored for non-TME repr_keys. Selected architecture_version becomes
    # TME_ARCHITECTURE_V1 (gru) or TME_ARCHITECTURE_LSTM_V1 (lstm).
    encoder_type: str = "gru"
    dropout: float = 0.1
    max_epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    proxy_alpha: float = 32.0
    proxy_margin: float = 0.1
    enable_state_supervision: bool = True
    d_supervision_weight: float = 0.0
    d_ranking_margin: float = 0.0
    angular_supervision_weight: float = 0.0
    angular_ranking_margin_rad: float = 0.0
    d_aux_samples_per_class: int = 0
    state_selection_min_d_gap: float = 1e-6
    state_selection_min_raw_theta_gap_rad: float = 0.08726646259971647
    state_selection_max_d_mannwhitney_p: float = 0.05
    state_selection_min_d_effect_size: float = 0.20
    patience: int = 10
    min_delta: float = 1e-4
    seed: int = 0
    # canonical_rerun_v2: global gradient clipping norm. 0.0 disables it.
    # The previous codepath had no clipping at all; for the v2 rerun we
    # default to 1.0 to keep proxy + D-aux backward passes stable.
    grad_clip_norm: float = 1.0
    # canonical_rerun_v2: hard-disable the early-stopping branch so the
    # full max_epochs budget runs every time (best.pt is still tracked by
    # the val_score improvement check above). Previous runs would bail at
    # the default patience=10 and produce under-fit encoders.
    disable_early_stopping: bool = True


@dataclass(frozen=True)
class TrainingResult:
    best_checkpoint_path: Path
    unconstrained_best_checkpoint_path: Path | None
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
        # Resolve encoder_type first so an explicit YAML value wins over the
        # architecture_version default. If encoder_type is absent we infer
        # it from architecture_version (gru for V1, lstm for LSTM_V1) so
        # older GRU configs continue to load unchanged.
        encoder_type = payload.get("encoder_type")
        if encoder_type is None:
            if architecture_version == TME_ARCHITECTURE_LSTM_V1:
                encoder_type = "lstm"
            else:
                encoder_type = "gru"
        if encoder_type not in ("gru", "lstm"):
            raise ValueError(
                f"encoder_type must be 'gru' or 'lstm', got {encoder_type!r}"
            )
        expected_arch = (
            TME_ARCHITECTURE_LSTM_V1 if encoder_type == "lstm" else TME_ARCHITECTURE_V1
        )
        if architecture_version is not None and architecture_version != expected_arch:
            raise ValueError(
                f"TME architecture_version {architecture_version!r} does not match "
                f"encoder_type {encoder_type!r} (expected {expected_arch!r})"
            )
        payload["encoder_type"] = encoder_type
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
        state_fields = (
            config.d_supervision_weight,
            config.d_ranking_margin,
            config.angular_supervision_weight,
            config.angular_ranking_margin_rad,
            config.d_aux_samples_per_class,
        )
        if config.enable_state_supervision:
            if config.d_supervision_weight < 0.0 or config.angular_supervision_weight < 0.0:
                raise ValueError("state-supervised TME requires non-negative D and angular weights")
            if config.d_supervision_weight == 0.0 and config.angular_supervision_weight == 0.0:
                raise ValueError("state-supervised TME requires at least one of D / angular weight > 0; use enable_state_supervision=false for PA-only")
            if config.d_ranking_margin < 0.0:
                raise ValueError("TME d_ranking_margin must be non-negative")
            if not 0.0 <= config.angular_ranking_margin_rad <= math.pi:
                raise ValueError("TME angular_ranking_margin_rad must be in [0, pi]")
            if config.d_aux_samples_per_class <= 0:
                raise ValueError("state-supervised TME requires positive aux samples per class")
            if (
                not math.isfinite(config.state_selection_min_d_gap)
                or config.state_selection_min_d_gap <= 0.0
            ):
                raise ValueError(
                    "state-supervised TME state_selection_min_d_gap must be finite and positive"
                )
            if (
                not math.isfinite(config.state_selection_min_raw_theta_gap_rad)
                or not 0.0
                < config.state_selection_min_raw_theta_gap_rad
                <= math.pi
            ):
                raise ValueError(
                    "state-supervised TME state_selection_min_raw_theta_gap_rad must be in (0, pi]"
                )
            if (
                not math.isfinite(config.state_selection_max_d_mannwhitney_p)
                or not 0.0 < config.state_selection_max_d_mannwhitney_p <= 1.0
            ):
                raise ValueError(
                    "state-supervised TME state_selection_max_d_mannwhitney_p must be in (0, 1]"
                )
            if (
                not math.isfinite(config.state_selection_min_d_effect_size)
                or config.state_selection_min_d_effect_size <= 0.0
            ):
                raise ValueError(
                    "state-supervised TME state_selection_min_d_effect_size "
                    "must be finite and positive"
                )
        elif any(value != 0 for value in state_fields):
            raise ValueError("PA-only TME requires all D/angular supervision fields to be zero")
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
