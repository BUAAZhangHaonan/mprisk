"""Representation loss names and torch contrastive losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

SUPPORTED_LOSSES = ("proxy_anchor", "supcon", "bce", "prompt_consistency", "tme_combined")


def prompt_consistency_loss(
    embeddings: torch.Tensor,
    sample_ids: Sequence[object] | torch.Tensor,
    view_keys: Sequence[object] | torch.Tensor,
    prompt_keys: Sequence[object] | torch.Tensor,
    *,
    temperature: float = 0.1,
    negative_budget_ratio: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Contrast different prompts for the same sample and view as positives."""
    _validate_embeddings(embeddings)
    same_sample = _pairwise_equal(sample_ids, embeddings.device, len(embeddings))
    same_view = _pairwise_equal(view_keys, embeddings.device, len(embeddings))
    different_prompt = ~_pairwise_equal(prompt_keys, embeddings.device, len(embeddings))
    positive_mask = same_sample & same_view & different_prompt
    return _masked_contrastive_loss(
        embeddings,
        positive_mask=positive_mask,
        temperature=temperature,
        negative_budget_ratio=negative_budget_ratio,
        eps=eps,
    )


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: Sequence[object] | torch.Tensor,
    *,
    temperature: float = 0.1,
    negative_budget_ratio: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Supervised contrastive loss with same-label positives."""
    _validate_embeddings(embeddings)
    positive_mask = _pairwise_equal(labels, embeddings.device, len(embeddings))
    return _masked_contrastive_loss(
        embeddings,
        positive_mask=positive_mask,
        negative_budget_ratio=negative_budget_ratio,
        temperature=temperature,
        eps=eps,
    )


def combined_trajectory_loss(
    embeddings: torch.Tensor,
    labels: Sequence[object] | torch.Tensor,
    sample_ids: Sequence[object] | torch.Tensor,
    view_keys: Sequence[object] | torch.Tensor,
    prompt_keys: Sequence[object] | torch.Tensor,
    *,
    prompt_weight: float = 1.0,
    supcon_weight: float = 1.0,
    temperature: float = 0.1,
    negative_budget_ratio: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Weighted sum of prompt consistency and supervised contrastive losses."""
    if prompt_weight < 0.0 or supcon_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if prompt_weight == 0.0 and supcon_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")

    prompt_loss = prompt_consistency_loss(
        embeddings,
        sample_ids=sample_ids,
        view_keys=view_keys,
        prompt_keys=prompt_keys,
        temperature=temperature,
        negative_budget_ratio=negative_budget_ratio,
        eps=eps,
    )
    supcon_loss = supervised_contrastive_loss(
        embeddings,
        labels=labels,
        temperature=temperature,
        negative_budget_ratio=negative_budget_ratio,
        eps=eps,
    )
    return prompt_weight * prompt_loss + supcon_weight * supcon_loss


def _masked_contrastive_loss(
    embeddings: torch.Tensor,
    *,
    positive_mask: torch.Tensor,
    temperature: float,
    negative_budget_ratio: float,
    eps: float,
) -> torch.Tensor:
    _validate_temperature(temperature)
    _validate_negative_budget_ratio(negative_budget_ratio)
    batch_size = embeddings.shape[0]
    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = positive_mask.to(device=embeddings.device, dtype=torch.bool) & ~eye
    anchor_mask = positive_mask.any(dim=1)
    if not bool(anchor_mask.any()):
        return embeddings.sum() * 0.0

    normalized = F.normalize(embeddings, p=2, dim=-1, eps=eps)
    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_mask = positive_mask | _budgeted_negative_mask(
        positive_mask=positive_mask,
        self_mask=eye,
        negative_budget_ratio=negative_budget_ratio,
    )
    masked_logits = logits.masked_fill(~denominator_mask, torch.finfo(logits.dtype).min)
    log_denominator = torch.logsumexp(masked_logits, dim=1)
    log_prob = logits - log_denominator.unsqueeze(1)
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    mean_positive_log_prob = (log_prob * positive_mask).sum(dim=1) / positive_counts
    return -mean_positive_log_prob[anchor_mask].mean()


def _budgeted_negative_mask(
    *,
    positive_mask: torch.Tensor,
    self_mask: torch.Tensor,
    negative_budget_ratio: float,
) -> torch.Tensor:
    negative_mask = ~(positive_mask | self_mask)
    if negative_budget_ratio >= 1.0:
        return negative_mask

    limited = torch.zeros_like(negative_mask)
    for row_index in range(negative_mask.shape[0]):
        negative_indices = torch.nonzero(negative_mask[row_index], as_tuple=False).flatten()
        if negative_indices.numel() == 0 or negative_budget_ratio == 0.0:
            continue
        keep_count = max(1, int(negative_indices.numel() * negative_budget_ratio))
        limited[row_index, negative_indices[:keep_count]] = True
    return limited


def _pairwise_equal(
    values: Sequence[object] | torch.Tensor,
    device: torch.device,
    expected_len: int,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.shape[0] != expected_len:
            raise ValueError("metadata length must match embedding batch size")
        flat_values = values.to(device=device).reshape(expected_len, -1)
        if flat_values.shape[1] == 1:
            return flat_values == flat_values.T
        return (flat_values[:, None, :] == flat_values[None, :, :]).all(dim=-1)

    if len(values) != expected_len:
        raise ValueError("metadata length must match embedding batch size")
    return torch.tensor(
        [[left == right for right in values] for left in values],
        dtype=torch.bool,
        device=device,
    )


def _validate_embeddings(embeddings: torch.Tensor) -> None:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, embed_dim]")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have non-empty batch and embed dimensions")


def _validate_temperature(temperature: float) -> None:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")


def _validate_negative_budget_ratio(negative_budget_ratio: float) -> None:
    if not 0.0 <= negative_budget_ratio <= 1.0:
        raise ValueError("negative_budget_ratio must be in [0.0, 1.0]")
