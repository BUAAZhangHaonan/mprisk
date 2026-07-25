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

from mprisk.representation.data import (
    _baseline_feature_definition,
    _batches,
    _load_trajectory_batch,
    _read_relation_rows,
    _registered_group_split,
    _rows_to_sample_refs,
    _trajectory_shape,
    _validate_checkpoint_architecture,
    _validate_prompt_contract,
    _validate_registered_splits,
    _vector_values,
)
from mprisk.representation.evaluation import (
    _ac_aux_metrics,
    _aggregate_sample_outputs,
    _balanced_accuracy,
    _encode_prompt_groups,
    _evaluate,
    _group_prompt_condition_z,
    _pooled_effect_size,
    _sample_level_predictions,
    _state_checkpoint_feasibility,
    _state_separation_summary,
)
from mprisk.representation.export import (
    _append_frozen_row,
    _baseline_export_row,
    _empty_frozen_bundle,
    _finalize_frozen_bundle,
    _frozen_row,
    _stream_baseline_exports,
    _stream_frozen_exports,
    export_frozen_baseline_representations,
    export_frozen_representations,
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








