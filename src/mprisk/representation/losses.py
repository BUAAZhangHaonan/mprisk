"""Proxy Anchor objective for the final TME representation."""

from __future__ import annotations

import torch
from torch.nn import functional as F

SUPPORTED_LOSSES = ("proxy_anchor", "cross_entropy")


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

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _validate_embeddings(embeddings)
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must have shape [batch]")
        if labels.dtype != torch.long:
            labels = labels.to(dtype=torch.long)
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("Proxy Anchor labels must be Aligned=0 or Conflict=1")
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        normalized_proxies = F.normalize(self.proxies, p=2, dim=-1)
        similarities = normalized_embeddings @ normalized_proxies.T
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=torch.bool)
        positive_classes = one_hot.any(dim=0)
        positive_terms = torch.exp(-self.alpha * (similarities - self.margin)) * one_hot
        negative_terms = torch.exp(self.alpha * (similarities + self.margin)) * ~one_hot
        positive_loss = torch.log1p(positive_terms.sum(dim=0))[positive_classes].mean()
        negative_loss = torch.log1p(negative_terms.sum(dim=0)).mean()
        return positive_loss + negative_loss

def _validate_embeddings(embeddings: torch.Tensor) -> None:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, embed_dim]")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have non-empty batch and embed dimensions")
