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
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.nn import functional as F

from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.cache.prompt_conditioned_cache import prompt_conditioned_entry_from_row
from mprisk.representation.losses import ModalitySplitRankingLoss, ProxyAnchorLoss
from mprisk.representation.relation_dataset import CONDITIONS, _reject_forbidden_fields
from mprisk.representation.relation_models import (
    REPRESENTATION_KEYS,
    SINGLE_POINT_BINARY_V1,
    TME_ARCHITECTURE_LSTM_V1,
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
    TRAJECTORY_MLP_BINARY_V1,
    build_representation_model,
    strict_l2_normalize,
)
from mprisk.utils.io import write_json

from mprisk.representation.config import (
    TRAINING_CONFIG_SCHEMA,
    REGISTERED_SPLITS,
    TrainingConfig,
    TrainingResult,
    FrozenRepresentationExportResult,
    FrozenBaselineExportResult,
    _Sample,
    load_training_config,
    _validate_config,
)
from mprisk.representation._io_utils import (
    _set_deterministic_seed,
    _resolve_device,
    _move_optimizer_state,
    _atomic_torch_save,
    _sha256,
)
from mprisk.representation.checkpoints import (
    _checkpoint_payload,
    _selection_metric_name,
    _group_checksum,
)

# canonical_rerun_v2 (20260721): added "cross_domain_test" so
# _validate_registered_splits accepts ch_sims_v2 rows BEFORE the
# exclude_prefix filter runs (Stage B bug: validator failed early on
# ch_sims rows that legitimately carry this representation_split).







def train_trajectory_encoder(
    *,
    dataset_path: str | Path,
    config: TrainingConfig,
    output_dir: str | Path,
    resume_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    exclude_prefix: str | None = None,
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
    if exclude_prefix:
        before = len(training_rows)
        # canonical_rerun_v2: exclude_prefix applies only to relation_train
        # so val can keep both Aligned (ch_sims) and Conflict (gen) labels.
        kept = []
        dropped = 0
        for row in training_rows:
            if row["sample_id"].startswith(exclude_prefix) and row["representation_split"] == "relation_train":
                dropped += 1
                continue
            kept.append(row)
        training_rows = kept
        print(
            f"[exclude_prefix={exclude_prefix!r}] dropped {dropped} train rows, "
            f"{len(training_rows)} remaining (val rows preserved)",
        )
    # canonical_rerun_v2 (20260721): also load official_test rows so every
    # epoch can be evaluated on the test split. best_checkpoint.pt is now
    # keyed on test_balanced_accuracy_ac (highest across all epochs), per
    # user spec. exclude_prefix applies to test rows too: if Stage-1 was
    # trained only on gen domain, we should not evaluate test on the
    # excluded natural-domain samples (they would not exist anyway because
    # the ch_sims rows live under aligned_calibration/cross_domain_test,
    # but the guard keeps the contract symmetric).
    test_rows = [
        row for row in rows
        if row["representation_split"] == "official_test"
        and not (exclude_prefix and row["sample_id"].startswith(exclude_prefix))
    ]
    samples = _rows_to_sample_refs(training_rows)
    _validate_prompt_contract(samples, config=config)
    train_samples, val_samples = _registered_group_split(samples)
    test_samples: list[_Sample] = _rows_to_sample_refs(test_rows) if test_rows else []
    if not test_samples:
        print(
            "[warn] no official_test rows found; best.pt will fall back to "
            "val_balanced_accuracy_ac selection",
        )
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
        encoder_type=getattr(config, "encoder_type", "gru"),
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
        if config.enable_state_supervision:
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
    unconstrained_best_path = output_root / "unconstrained_best_checkpoint.pt"
    last_path = output_root / "last_checkpoint.pt"
    config_path = output_root / "train_config.yaml"
    metrics_path = output_root / "train_metrics.json"
    log_path = output_root / "train_log.jsonl"
    start_epoch = 1
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_validation_state_separation: dict[str, float] | None = None
    unconstrained_best_score = -1.0
    unconstrained_best_epoch = 0
    unconstrained_best_validation_state_separation: dict[str, float] | None = None
    # canonical_rerun_v2 stage_c review (20260721): F1/AP reported alongside
    # balanced_accuracy; checkpoint selection still keyed on balanced_accuracy
    # subject to state feasibility (PA training does not directly yield a
    # classifier, so we keep the original selection signal).
    best_ac_f1 = 0.0
    best_ac_ap = 0.0
    best_epoch_ac_f1 = 0
    best_epoch_ac_ap = 0
    # canonical_rerun_v2 (20260721): best.pt is now keyed on
    # test_balanced_accuracy_ac (the user spec for T1/T5 frozen PA). We
    # still compute the val_* selection fields above for backward
    # compatibility / state feasibility diagnostics, but the actual
    # best_checkpoint.pt write follows best_test_score. When test_samples
    # is empty we fall back to the val selection (legacy path).
    best_test_score = -1.0
    best_test_epoch = 0
    best_test_ac_f1 = 0.0
    best_test_ac_ap = 0.0
    best_epoch_test_ac_f1 = 0
    best_epoch_test_ac_ap = 0
    best_test_preds: list[int] | None = None
    best_test_probs: list[float] | None = None
    best_test_sample_ids: list[str] | None = None
    best_test_labels: list[int] | None = None
    test_at_best_val_score = 0.0
    test_at_best_val_f1 = 0.0
    test_at_best_val_ap = 0.0
    has_test_samples = bool(test_samples)
    state_constrained_selection = (
        config.repr_key == TME_PROXY_ANCHOR_V1 and config.enable_state_supervision
    )
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
        unconstrained_best_score = float(
            checkpoint.get("unconstrained_best_score", best_score)
        )
        unconstrained_best_epoch = int(
            checkpoint.get("unconstrained_best_epoch", best_epoch)
        )
        unconstrained_best_validation_state_separation = checkpoint.get(
            "unconstrained_best_validation_state_separation",
            best_validation_state_separation,
        )
    else:
        log_path.write_text("", encoding="utf-8")
        best_path.unlink(missing_ok=True)
        unconstrained_best_path.unlink(missing_ok=True)

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
        val_loss, val_score, val_state_separation, val_f1, val_ap = _evaluate(
            model,
            objective,
            d_objective,
            val_samples,
            config=config,
            class_weights=class_weights,
        )
        # canonical_rerun_v2 (20260721): also evaluate on official_test
        # every epoch so best.pt can be keyed on test_balanced_accuracy_ac.
        # When test_samples is empty, fall back to NaN / no test-driven
        # selection (legacy val-keyed behavior takes over).
        if test_samples:
            (
                test_loss,
                test_score,
                _test_state_separation,
                test_f1,
                test_ap,
                test_sample_ids_out,
                test_labels_out,
                test_preds_out,
                test_probs_out,
            ) = _evaluate(
                model,
                objective,
                d_objective,
                test_samples,
                config=config,
                class_weights=class_weights,
                return_preds=True,
            )
        else:
            test_loss = float("nan")
            test_score = float("nan")
            test_f1 = float("nan")
            test_ap = float("nan")
            test_sample_ids_out = None
            test_labels_out = None
            test_preds_out = None
            test_probs_out = None
        checkpoint_feasibility = _state_checkpoint_feasibility(
            val_score=val_score,
            val_state_separation=val_state_separation,
            config=config,
        )
        unconstrained_improved = math.isfinite(val_score) and val_score > (
            unconstrained_best_score
            if state_constrained_selection
            else unconstrained_best_score + config.min_delta
        )
        if unconstrained_improved:
            unconstrained_best_score = val_score
            unconstrained_best_epoch = epoch
            unconstrained_best_validation_state_separation = val_state_separation
        if state_constrained_selection:
            improved = (
                bool(checkpoint_feasibility["feasible"])
                and val_score > best_score
            )
        else:
            improved = val_score > best_score + config.min_delta
        if improved:
            best_score = val_score
            best_epoch = epoch
            best_validation_state_separation = val_state_separation
            stale_epochs = 0
        elif not state_constrained_selection or best_epoch > 0:
            stale_epochs += 1
        else:
            stale_epochs = 0
        if math.isfinite(val_f1) and val_f1 > best_ac_f1:
            best_ac_f1 = val_f1
            best_epoch_ac_f1 = epoch
        if math.isfinite(val_ap) and val_ap > best_ac_ap:
            best_ac_ap = val_ap
            best_epoch_ac_ap = epoch
        # Diagnostic only: track max test metric across epochs. Not used for
        # checkpoint or best_test_preds selection.
        if has_test_samples and math.isfinite(test_score) and test_score > best_test_score:
            best_test_score = test_score
            best_test_epoch = epoch
        if has_test_samples and math.isfinite(test_f1) and test_f1 > best_test_ac_f1:
            best_test_ac_f1 = test_f1
            best_epoch_test_ac_f1 = epoch
        if has_test_samples and math.isfinite(test_ap) and test_ap > best_test_ac_ap:
            best_test_ac_ap = test_ap
            best_epoch_test_ac_ap = epoch
        stale_threshold_met = stale_epochs >= config.patience and (
            not state_constrained_selection or best_epoch > 0
        )
        # canonical_rerun_v2: disable_early_stopping keeps best/last
        # checkpoint selection intact but skips the break. The full
        # max_epochs budget runs every time.
        converged = stale_threshold_met and not config.disable_early_stopping
        log_row = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_loss,
            "val_balanced_accuracy_ac": val_score,
            "val_ac_f1": val_f1,
            "val_ac_ap": val_ap,
            "val_state_separation": val_state_separation,
            "checkpoint_feasibility": checkpoint_feasibility,
            "val_sample_count": len({sample.sample_id for sample in val_samples}),
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy_ac": best_score,
            "best_val_ac_f1": best_ac_f1,
            "best_val_ac_ap": best_ac_ap,
            "best_epoch_ac_f1": best_epoch_ac_f1,
            "best_epoch_ac_ap": best_epoch_ac_ap,
            "unconstrained_best_epoch": unconstrained_best_epoch,
            "unconstrained_best_val_balanced_accuracy_ac": unconstrained_best_score,
            # canonical_rerun_v2 (20260721): test-keyed selection fields.
            "test_loss": test_loss,
            "test_balanced_accuracy_ac": test_score,
            "test_ac_f1": test_f1,
            "test_ac_ap": test_ap,
            "test_sample_count": (
                len({sample.sample_id for sample in test_samples}) if test_samples else 0
            ),
            "best_epoch_test_balanced_accuracy_ac": best_test_epoch,
            "best_test_balanced_accuracy_ac": best_test_score,
            "best_test_ac_f1": best_test_ac_f1,
            "best_test_ac_ap": best_test_ac_ap,
            "best_epoch_test_ac_f1": best_epoch_test_ac_f1,
            "best_epoch_test_ac_ap": best_epoch_test_ac_ap,
            "stale_epochs": stale_epochs,
            "converged": converged,
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
            unconstrained_best_score=unconstrained_best_score,
            unconstrained_best_epoch=unconstrained_best_epoch,
            unconstrained_best_validation_state_separation=(
                unconstrained_best_validation_state_separation
            ),
            checkpoint_feasibility=checkpoint_feasibility,
            class_weights=class_weights,
            train_label_counts=train_label_counts,
        )
        _atomic_torch_save(last_path, checkpoint)
        if improved:
            # val-driven selection (20260723): best.pt and the per-sample test
            # preds snapshot are both keyed on val_balanced_accuracy_ac, so
            # best_checkpoint.pt and best_test_preds.pt always correspond to
            # the same epoch.
            _atomic_torch_save(
                best_path,
                {**checkpoint, "checkpoint_role": "final_selected"},
            )
            best_test_sample_ids = list(test_sample_ids_out) if test_sample_ids_out is not None else []
            best_test_labels = list(test_labels_out) if test_labels_out is not None else []
            best_test_preds = list(test_preds_out) if test_preds_out is not None else []
            best_test_probs = list(test_probs_out) if test_probs_out is not None else []
            test_at_best_val_score = test_score
            test_at_best_val_f1 = test_f1
            test_at_best_val_ap = test_ap
        if state_constrained_selection and unconstrained_improved:
            _atomic_torch_save(
                unconstrained_best_path,
                {**checkpoint, "checkpoint_role": "unconstrained_diagnostic"},
            )
        if converged:
            stop_reason = "early_stopping"
            break
    if state_constrained_selection and best_epoch == 0:
        best_path.unlink(missing_ok=True)
        raise RuntimeError(
            "state-supervised TME reached max_epochs without a feasible checkpoint: "
            f"requires val_D_gap >= {config.state_selection_min_d_gap} and "
            "val_raw_theta_gap_rad >= "
            f"{config.state_selection_min_raw_theta_gap_rad}, val_D_mannwhitney_p <= "
            f"{config.state_selection_max_d_mannwhitney_p}, and val_D_effect_size >= "
            f"{config.state_selection_min_d_effect_size} with finite selection metrics"
        )
    if state_constrained_selection:
        if best_path.is_file():
            final_payload = torch.load(best_path, map_location="cpu")
            final_payload.update(
                {
                    "unconstrained_best_score": unconstrained_best_score,
                    "unconstrained_best_epoch": unconstrained_best_epoch,
                    "unconstrained_best_validation_state_separation": (
                        unconstrained_best_validation_state_separation
                    ),
                    "unconstrained_best_checkpoint": str(unconstrained_best_path),
                }
            )
            _atomic_torch_save(best_path, final_payload)
    if (
        not state_constrained_selection
        and not best_path.is_file()
        and last_path.is_file()
    ):
        final_payload = torch.load(last_path, map_location="cpu")
        final_payload["checkpoint_role"] = "final_selected"
        _atomic_torch_save(best_path, final_payload)
    # Persist per-sample test preds/probs/labels at the val-selected best
    # epoch so downstream stages (SOUP/SWA, Figure 6/7) can load them without
    # rerunning inference. Aligned with best_checkpoint.pt by construction.
    if best_test_preds is not None:
        best_test_path = output_root / "best_test_preds.pt"
        torch.save(
            {
                "epoch": int(best_epoch),
                "sample_ids": list(best_test_sample_ids or []),
                "labels": list(best_test_labels or []),
                "preds": list(best_test_preds or []),
                "probs": list(best_test_probs or []),
                "selection_metric": "val_balanced_accuracy_ac",
                "test_at_best_val_balanced_accuracy_ac": float(test_at_best_val_score),
                "test_at_best_val_ac_f1": float(test_at_best_val_f1),
                "test_at_best_val_ac_ap": float(test_at_best_val_ap),
            },
            best_test_path,
        )
    metrics = {
        "schema": "mprisk_representation_training_metrics_v3",
        "repr_key": config.repr_key,
        "model_key": config.model_key,
        "selection_metric": _selection_metric_name(config),
        "selection_unit": "sample_id",
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy_ac": best_score,
        "best_val_ac_f1": best_ac_f1,
        "best_val_ac_ap": best_ac_ap,
        "best_epoch_ac_f1": best_epoch_ac_f1,
        "best_epoch_ac_ap": best_epoch_ac_ap,
        "best_validation_state_separation": best_validation_state_separation,
        "unconstrained_best_epoch": unconstrained_best_epoch,
        "unconstrained_best_val_balanced_accuracy_ac": unconstrained_best_score,
        "unconstrained_best_validation_state_separation": (
            unconstrained_best_validation_state_separation
        ),
        "unconstrained_best_checkpoint": (
            str(unconstrained_best_path) if state_constrained_selection else None
        ),
        # Diagnostic: best_test_* tracks the max test metric observed across
        # all epochs (NOT used for selection). test_at_best_val_* is the test
        # metric at the val-selected best epoch — these are the reported
        # headline numbers.
        "test_at_best_val_balanced_accuracy_ac": float(test_at_best_val_score),
        "test_at_best_val_ac_f1": float(test_at_best_val_f1),
        "test_at_best_val_ac_ap": float(test_at_best_val_ap),
        "best_epoch_test_balanced_accuracy_ac": int(best_test_epoch),
        "best_test_balanced_accuracy_ac": float(best_test_score),
        "best_test_ac_f1": float(best_test_ac_f1),
        "best_test_ac_ap": float(best_test_ac_ap),
        "best_epoch_test_ac_f1": int(best_epoch_test_ac_f1),
        "best_epoch_test_ac_ap": int(best_epoch_test_ac_ap),
        "test_rows": len(test_samples),
        "test_sample_count": len({sample.sample_id for sample in test_samples}),
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
                "checkpoint_selection": {
                    "min_D_gap": config.state_selection_min_d_gap,
                    "min_raw_theta_gap_rad": config.state_selection_min_raw_theta_gap_rad,
                    "min_raw_theta_gap_deg": math.degrees(
                        config.state_selection_min_raw_theta_gap_rad
                    ),
                    "max_D_mannwhitney_p": (
                        config.state_selection_max_d_mannwhitney_p
                    ),
                    "min_D_effect_size": config.state_selection_min_d_effect_size,
                    "rule": "highest_finite_val_balanced_accuracy_among_feasible_epochs",
                },
            }
            if config.repr_key == TME_PROXY_ANCHOR_V1 and config.enable_state_supervision
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
        unconstrained_best_checkpoint_path=(
            unconstrained_best_path if state_constrained_selection else None
        ),
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
    if checkpoint.get("checkpoint_role") == "unconstrained_diagnostic":
        raise ValueError("unconstrained diagnostic checkpoint cannot be exported")
    training_config = checkpoint.get("training_config")
    if (
        isinstance(training_config, dict)
        and training_config.get("enable_state_supervision") is True
        and (
            checkpoint.get("checkpoint_role") != "final_selected"
            or not checkpoint.get("checkpoint_feasibility", {}).get("feasible", False)
        )
    ):
        raise ValueError(
            "state-supervised TME export requires the final feasible checkpoint"
        )
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
        encoder_type=getattr(config, "encoder_type", "gru"),
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
    if representation_split not in {"relation_train", "relation_val", "aligned_calibration", "official_test"}:
        raise ValueError("baseline export requires a valid representation split")
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
        encoder_type=getattr(config, "encoder_type", "gru"),
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
    if repr_key == TME_PROXY_ANCHOR_V1:
        # TME checkpoints can be either GRU (V1) or LSTM (LSTM_V1). Both
        # are valid for resuming as long as the architecture_version field
        # matches one of the two known TME architectures.
        if architecture_version not in (TME_ARCHITECTURE_V1, TME_ARCHITECTURE_LSTM_V1):
            raise ValueError(
                "checkpoint architecture_version does not match its representation"
            )
        return
    expected_architecture = repr_key
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
    accepted_weight_shapes = {(2, 3 * input_dim), (2, input_dim)}
    if (
        not isinstance(input_dim, int)
        or input_dim <= 0
        or set(model_state) != {"classifier.weight", "classifier.bias"}
        or not isinstance(weight, torch.Tensor)
        or tuple(weight.shape) not in accepted_weight_shapes
        or not isinstance(bias, torch.Tensor)
        or tuple(bias.shape) != (2,)
    ):
        raise ValueError(
            "Single-Point checkpoint architecture drift: expected direct Linear(H or 3H, 2)"
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
        # canonical_rerun_v2 (20260721): cross_domain_test rows carry
        # master_split == representation_split == "cross_domain_test"
        # (see relation_dataset.py build_relation_dataset).
        "cross_domain_test": "cross_domain_test",
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
        if config.grad_clip_norm and config.grad_clip_norm > 0:
            # C-A1-R5-1: clip every trainable param the optimizer knows about,
            # which includes the Proxy Anchor proxies held inside `objective`.
            # Using model.parameters() alone silently skipped them.
            clip_params: list[nn.Parameter] = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.requires_grad:
                        clip_params.append(p)
            if clip_params:
                torch.nn.utils.clip_grad_norm_(
                    clip_params, max_norm=float(config.grad_clip_norm)
                )
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
    include_significance: bool = False,
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
    raw_theta_gap_rad = angle_conflict.mean() - angle_aligned.mean()
    summary = {
        f"{prefix}_aligned_D_mean": float(d_aligned.mean()),
        f"{prefix}_conflict_D_mean": float(d_conflict.mean()),
        f"{prefix}_D_gap": float(d_conflict.mean() - d_aligned.mean()),
        f"{prefix}_D_effect_size": _pooled_effect_size(d_aligned, d_conflict),
        f"{prefix}_D_pair_margin_satisfaction": float((d_gaps >= d_margin).float().mean()),
        f"{prefix}_aligned_split_angle_deg_mean": float(angle_aligned.mean() * degrees),
        f"{prefix}_conflict_split_angle_deg_mean": float(angle_conflict.mean() * degrees),
        f"{prefix}_split_angle_gap_deg": float(
            raw_theta_gap_rad * degrees
        ),
        f"{prefix}_raw_theta_gap_rad": float(raw_theta_gap_rad),
        f"{prefix}_raw_theta_gap_deg": float(raw_theta_gap_rad * degrees),
        f"{prefix}_split_angle_effect_size": _pooled_effect_size(
            angle_aligned, angle_conflict
        ),
        f"{prefix}_angular_pair_margin_satisfaction": float(
            (angle_gaps >= angular_margin_rad).float().mean()
        ),
    }
    if include_significance:
        d_test = mannwhitneyu(
            d_conflict.detach().cpu().numpy(),
            d_aligned.detach().cpu().numpy(),
            alternative="two-sided",
            method="auto",
        )
        summary[f"{prefix}_D_mannwhitney_p"] = float(d_test.pvalue)
    return summary


def _state_checkpoint_feasibility(
    *,
    val_score: float,
    val_state_separation: dict[str, float] | None,
    config: TrainingConfig,
) -> dict[str, Any]:
    enabled = config.repr_key == TME_PROXY_ANCHOR_V1 and config.enable_state_supervision
    if not enabled:
        return {"enabled": False, "feasible": True}
    d_gap = (
        float(val_state_separation["val_D_gap"])
        if val_state_separation is not None and "val_D_gap" in val_state_separation
        else float("nan")
    )
    raw_theta_gap_rad = (
        float(val_state_separation["val_raw_theta_gap_rad"])
        if val_state_separation is not None
        and "val_raw_theta_gap_rad" in val_state_separation
        else float("nan")
    )
    d_mannwhitney_p = (
        float(val_state_separation["val_D_mannwhitney_p"])
        if val_state_separation is not None
        and "val_D_mannwhitney_p" in val_state_separation
        else float("nan")
    )
    d_effect_size = (
        float(val_state_separation["val_D_effect_size"])
        if val_state_separation is not None
        and "val_D_effect_size" in val_state_separation
        else float("nan")
    )
    required_metrics_finite = all(
        math.isfinite(value)
        for value in (
            val_score,
            d_gap,
            raw_theta_gap_rad,
            d_mannwhitney_p,
            d_effect_size,
        )
    )
    feasible = (
        required_metrics_finite
        and d_gap >= config.state_selection_min_d_gap
        and raw_theta_gap_rad >= config.state_selection_min_raw_theta_gap_rad
        and d_mannwhitney_p <= config.state_selection_max_d_mannwhitney_p
        and d_effect_size >= config.state_selection_min_d_effect_size
    )
    return {
        "enabled": True,
        "feasible": feasible,
        "required_metrics_finite": required_metrics_finite,
        "observed_val_balanced_accuracy_ac": val_score,
        "observed_val_D_gap": d_gap,
        "observed_val_raw_theta_gap_rad": raw_theta_gap_rad,
        "observed_val_raw_theta_gap_deg": math.degrees(raw_theta_gap_rad),
        "observed_val_D_mannwhitney_p": d_mannwhitney_p,
        "observed_val_D_effect_size": d_effect_size,
        "minimum_val_D_gap": config.state_selection_min_d_gap,
        "minimum_val_raw_theta_gap_rad": config.state_selection_min_raw_theta_gap_rad,
        "minimum_val_raw_theta_gap_deg": math.degrees(
            config.state_selection_min_raw_theta_gap_rad
        ),
        "maximum_val_D_mannwhitney_p": config.state_selection_max_d_mannwhitney_p,
        "minimum_val_D_effect_size": config.state_selection_min_d_effect_size,
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
    return_preds: bool = False,
):
    """Evaluate ``model`` on ``samples``.

    Returns a 5-tuple ``(loss, balanced_acc, state_separation, f1, ap)`` by
    default.
    When ``return_preds=True`` returns a 9-tuple adding
    ``(sample_ids, labels_int, predictions_int, conflict_scores)``:

        (loss, balanced_acc, state_separation, f1, ap,
         sample_ids, labels_int, predictions_int, conflict_scores)

    so canonical_rerun_v2 callers can persist per-sample test preds aligned
    with best.pt.
    """
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
    sample_ids_out, labels, aggregate = _aggregate_sample_outputs(
        metric_samples,
        torch.cat(metric_outputs, dim=0),
        normalize=objective is not None,
    )
    predictions, similarities = _sample_level_predictions(
        aggregate, objective=objective, return_similarities=True
    )
    prediction_values = [int(value) for value in predictions.detach().cpu().numpy()]
    f1_value, ap_value, conflict_scores = _ac_aux_metrics(
        labels, aggregate, predictions,
        objective=objective, return_scores=True, similarities=similarities,
    )
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
            include_significance=True,
        )
    if return_preds:
        return (
            float(np.mean(losses)),
            _balanced_accuracy(labels, prediction_values),
            state_separation,
            f1_value,
            ap_value,
            sample_ids_out,
            [int(v) for v in labels],
            prediction_values,
            conflict_scores,
        )
    return (
        float(np.mean(losses)),
        _balanced_accuracy(labels, prediction_values),
        state_separation,
        f1_value,
        ap_value,
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
    return_similarities: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Argmax over class scores.

    By default returns just the predictions. When ``return_similarities=True``
    also returns the similarities matrix (or ``None`` when there is no
    objective, in which case ``aggregate`` itself is the logit matrix and
    callers should use it directly). The similarities are returned so the
    caller can pass them to ``_ac_aux_metrics`` and avoid recomputing the
    same ``aggregate @ proxies.T`` matmul twice (M-A1-R5-3).
    """
    if objective is None:
        preds = aggregate.argmax(dim=-1)
        if return_similarities:
            return preds, None
        return preds
    similarities = aggregate @ objective.normalized_proxies().T
    preds = similarities.argmax(dim=-1)
    if return_similarities:
        return preds, similarities
    return preds


def _balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    recalls = []
    for label in (0, 1):
        indexes = [index for index, value in enumerate(labels) if value == label]
        if not indexes:
            raise ValueError("validation must contain both A/C labels")
        recalls.append(sum(predictions[index] == label for index in indexes) / len(indexes))
    return float(sum(recalls) / len(recalls))


def _ac_aux_metrics(
    labels: list[int],
    aggregate: torch.Tensor,
    predictions: torch.Tensor,
    *,
    objective: ProxyAnchorLoss | None,
    return_scores: bool = False,
    similarities: torch.Tensor | None = None,
) -> tuple[float, float] | tuple[float, float, np.ndarray]:
    """Compute binary F1 (pos_label=1=Conflict) and AP.

    For proxy_anchor (T1/T5 frozen): aggregate is a unit vector on the 64-d
    sphere, objective.normalized_proxies() are unit class centroids; cosine
    similarity to the Conflict centroid (index 1) serves as the AP ranking
    score. Predictions come from the existing argmax over both centroids.
    For baseline cross-entropy: aggregate is logits [N,2]; softmax P(class=1)
    serves as the AP score.

    When ``return_scores=True`` the third return value is the per-sample
    conflict_score array (used by canonical_rerun_v2 to persist
    best_test_probs aligned with best.pt).

    ``similarities`` (M-A1-R5-3): if the caller already computed
    ``aggregate @ proxies.T`` for prediction, pass it in here to avoid
    recomputing the matmul. When ``objective is None`` this is ignored.
    """
    labels_np = np.asarray(labels, dtype=np.int64)
    if objective is not None:
        with torch.no_grad():
            if similarities is None:
                similarities = aggregate @ objective.normalized_proxies().T
            conflict_scores = similarities[:, 1].detach().float().cpu().numpy()
    else:
        probs = torch.softmax(aggregate, dim=-1)[:, 1]
        conflict_scores = probs.detach().float().cpu().numpy()
    pred_values = (
        [int(value) for value in predictions.detach().cpu().numpy()]
        if isinstance(predictions, torch.Tensor)
        else [int(value) for value in predictions]
    )
    f1_value = float(f1_score(labels_np, pred_values, pos_label=1, zero_division=0))
    ap_value = float(average_precision_score(labels_np, conflict_scores))
    if return_scores:
        return f1_value, ap_value, conflict_scores
    return f1_value, ap_value




def _trajectory_shape(samples: list[_Sample]) -> tuple[int, int]:
    entry = samples[0].condition_entries[0]
    return int(entry.layer_count), int(entry.hidden_dim)


def _batches(samples: list[_Sample], batch_size: int) -> list[list[_Sample]]:
    return [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]


def _training_signature(dataset_path: str | Path, config: TrainingConfig) -> str:
    config_payload = asdict(config)
    config_payload.pop("max_epochs")
    if not (
        config.repr_key == TME_PROXY_ANCHOR_V1 and config.enable_state_supervision
    ):
        config_payload.pop("state_selection_min_d_gap")
        config_payload.pop("state_selection_min_raw_theta_gap_rad")
        config_payload.pop("state_selection_max_d_mannwhitney_p")
        config_payload.pop("state_selection_min_d_effect_size")
    payload = {
        "dataset_sha256": _sha256(Path(dataset_path)),
        "config": config_payload,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()








