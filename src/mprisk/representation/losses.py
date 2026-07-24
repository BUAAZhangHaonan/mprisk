"""Proxy Anchor and state-structure objectives for the final TME representation."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from mprisk.representation.relation_models import strict_l2_normalize

SUPPORTED_LOSSES = ("proxy_anchor", "cross_entropy")
SPHERICAL_ACOS_EPS = 1e-7
STATE_DENOMINATOR_EPS = 1e-12


class ProxyAnchorLoss(torch.nn.Module):
    """Standard Proxy Anchor objective over normalized relation embeddings."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_classes: int = 2,
        alpha: float = 32.0,
        margin: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_classes != 2:
            raise ValueError(
                "TME Proxy Anchor requires a positive embed_dim and exactly two classes"
            )
        if alpha <= 0.0 or margin < 0.0:
            raise ValueError("alpha must be positive and margin must be non-negative")
        self.alpha = alpha
        self.margin = margin
        self.num_classes = num_classes
        self.proxies = torch.nn.Parameter(torch.empty(num_classes, embed_dim))
        torch.nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_ids: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        _validate_embeddings(embeddings)
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must have shape [batch]")
        if labels.dtype != torch.long:
            labels = labels.to(dtype=torch.long)
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("Proxy Anchor labels must be Aligned=0 or Conflict=1")
        normalized_embeddings = strict_l2_normalize(
            embeddings,
            stage="proxy_anchor_embeddings",
            sample_ids=sample_ids,
        )
        normalized_proxies = self.normalized_proxies()
        similarities = normalized_embeddings @ normalized_proxies.T
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=torch.bool)
        positive_classes = one_hot.any(dim=0)
        positive_terms = torch.exp(-self.alpha * (similarities - self.margin)) * one_hot
        negative_terms = torch.exp(self.alpha * (similarities + self.margin)) * ~one_hot
        positive_loss = torch.log1p(positive_terms.sum(dim=0))[positive_classes].mean()
        negative_loss = torch.log1p(negative_terms.sum(dim=0)).mean()
        return positive_loss + negative_loss

    def normalized_proxies(self) -> torch.Tensor:
        return strict_l2_normalize(
            self.proxies,
            stage="proxy_anchor_proxies",
            sample_ids=tuple(f"proxy_class_{index}" for index in range(self.num_classes)),
        )


class ModalitySplitRankingLoss(torch.nn.Module):
    """Rank Conflict above Aligned in exact D and raw spherical split angle.

    The forward value of D_for_ranking equals the paper definition
    theta / (sqrt(S_M1 + S_M2) + eps). Its denominator is detached so the
    auxiliary gradient cannot increase D by collapsing prompt dispersion.
    """

    def __init__(
        self,
        *,
        d_margin: float,
        angular_margin_rad: float,
    ) -> None:
        super().__init__()
        if d_margin < 0.0:
            raise ValueError("d_margin must be non-negative")
        if not 0.0 <= angular_margin_rad <= torch.pi:
            raise ValueError("angular_margin_rad must be in [0, pi]")
        self.d_margin = float(d_margin)
        self.angular_margin_rad = float(angular_margin_rad)

    def forward(
        self,
        condition_z: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_ids: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if condition_z.ndim != 4 or condition_z.shape[2] != 3:
            raise ValueError(
                "condition_z must have shape [sample, prompt, 3, condition_dim]"
            )
        if condition_z.shape[0] < 2 or condition_z.shape[1] < 2:
            raise ValueError("D supervision requires at least two samples and two prompts")
        if labels.ndim != 1 or labels.shape[0] != condition_z.shape[0]:
            raise ValueError("D supervision labels must have shape [sample]")
        if labels.dtype != torch.long:
            labels = labels.to(dtype=torch.long)
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("D supervision labels must be Aligned=0 or Conflict=1")
        if not bool((labels == 0).any()) or not bool((labels == 1).any()):
            raise ValueError("D supervision batches require both Aligned and Conflict")

        normalized = strict_l2_normalize(
            condition_z,
            stage="d_supervision_condition_z",
            sample_ids=sample_ids,
        )
        m1 = normalized[:, :, 0, :]
        m2 = normalized[:, :, 1, :]
        center_m1 = strict_l2_normalize(
            m1.mean(dim=1),
            stage="d_supervision_center_m1",
            sample_ids=sample_ids,
        )
        center_m2 = strict_l2_normalize(
            m2.mean(dim=1),
            stage="d_supervision_center_m2",
            sample_ids=sample_ids,
        )
        split_angle = _stable_acos((center_m1 * center_m2).sum(dim=-1))
        s_m1 = _prompt_geodesic_dispersion(m1, center_m1)
        s_m2 = _prompt_geodesic_dispersion(m2, center_m2)
        denominator = torch.sqrt(s_m1 + s_m2) + STATE_DENOMINATOR_EPS
        d_exact = split_angle / denominator
        d_for_ranking = split_angle / denominator.detach()

        conflict = labels == 1
        aligned = labels == 0
        d_pair_gaps = d_for_ranking[conflict, None] - d_for_ranking[aligned][None, :]
        angle_pair_gaps = split_angle[conflict, None] - split_angle[aligned][None, :]
        d_loss = F.relu(self.d_margin - d_pair_gaps).mean()
        angular_loss = F.relu(self.angular_margin_rad - angle_pair_gaps).mean()
        diagnostics = {
            "D": d_exact,
            "D_for_ranking": d_for_ranking,
            "split_angle_rad": split_angle,
            "S_M1": s_m1,
            "S_M2": s_m2,
            "dispersion_denominator": denominator,
            "d_pair_satisfaction": (d_pair_gaps >= self.d_margin).to(torch.float32).mean(),
            "angular_pair_satisfaction": (
                angle_pair_gaps >= self.angular_margin_rad
            ).to(torch.float32).mean(),
        }
        return d_loss, angular_loss, diagnostics


def _stable_acos(cosine: torch.Tensor) -> torch.Tensor:
    return torch.acos(cosine.clamp(-1.0 + SPHERICAL_ACOS_EPS, 1.0 - SPHERICAL_ACOS_EPS))


def _prompt_geodesic_dispersion(
    prompts: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    angles = _stable_acos((prompts * center[:, None, :]).sum(dim=-1))
    return angles.square().mean(dim=1)


def _validate_embeddings(embeddings: torch.Tensor) -> None:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, embed_dim]")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have non-empty batch and embed dimensions")
