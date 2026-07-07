"""Torch trajectory manifold encoder modules."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerMeanPool(nn.Module):
    """Normalize each layer vector, then average across layers."""

    def __init__(self, *, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [batch, layer_count, hidden_dim]")
        normalized_layers = F.normalize(trajectories, p=2, dim=-1, eps=self.eps)
        return normalized_layers.mean(dim=1)


class MLPProjection(nn.Module):
    """Trajectory Manifold Encoder v1 projection head."""

    def __init__(
        self,
        input_dim: int | None = None,
        *,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
        normalize_output: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if pooling != "mean":
            raise ValueError("MLPProjection v1 only supports pooling='mean'")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if input_dim is not None and input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

        self.input_dim = hidden_dim if input_dim is None else input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.normalize_output = normalize_output
        self.eps = eps
        self.pool = LayerMeanPool(eps=eps)
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(trajectories)
        if pooled.shape[-1] != self.input_dim:
            raise ValueError(
                "trajectory hidden_dim "
                f"{pooled.shape[-1]} does not match input_dim {self.input_dim}"
            )
        embeddings = self.projection(pooled)
        if self.normalize_output:
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=self.eps)
        return embeddings
