"""Evaluation utilities for sample-level relation representations.

These helpers compute balanced accuracy / F1 / AP / state-separation
diagnostics on validation and test samples. They were originally defined
in ``mprisk.representation.training`` and remain re-exported from there
so existing callers (e.g. ``scripts/regenerate_val_test_preds.py`` and
``tests/test_representation/test_proxy_anchor_training_pipeline.py``)
keep working unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, f1_score
from torch import nn

from mprisk.representation.config import TrainingConfig, _Sample
from mprisk.representation.losses import ModalitySplitRankingLoss, ProxyAnchorLoss
from mprisk.representation.relation_models import TME_PROXY_ANCHOR_V1
from mprisk.representation.data import _batches, _load_trajectory_batch

__all__ = [
    "_encode_prompt_groups",
    "_group_prompt_condition_z",
    "_state_separation_summary",
    "_state_checkpoint_feasibility",
    "_pooled_effect_size",
    "_evaluate",
    "_aggregate_sample_outputs",
    "_sample_level_predictions",
    "_balanced_accuracy",
    "_ac_aux_metrics",
]


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
    from mprisk.representation.training import _batch_loss_and_outputs
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
