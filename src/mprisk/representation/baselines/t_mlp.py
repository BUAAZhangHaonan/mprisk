"""T-MLP baseline encoder (full trajectory flattened -> 2-layer MLP).

Extracted from the original ``mprisk.representation.baselines`` module (P4-F refactor)
to keep each encoder family in its own file. Public surface stays
re-exported from ``mprisk.representation.baselines``.

Architecture is intentionally pure: no LayerNorm, no sphere normalization.
The penultimate 1024-d GELU output is the representation handed off to
downstream probes.

Trajectory contract: ``[B, L, H]``.
"""

from __future__ import annotations

import torch
from torch import nn


__all__ = ["TrajectoryMLP"]


class TrajectoryMLP(nn.Module):
    """Full trajectory flatten -> 2-layer MLP -> 2 logits.

    No LayerNorm. The penultimate 1024-d GELU output is the representation
    handed off to downstream probes.
    """

    architecture_version: str = "baseline_t_mlp"

    def __init__(
        self,
        layer_count: int,
        hidden_dim: int,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layer_count = int(layer_count)
        self.hidden_dim = int(hidden_dim)
        self.n_classes = int(n_classes)
        in_dim = self.layer_count * self.hidden_dim
        # canonical_rerun spec: encoder 1024-d + [1024 -> 128 -> 32 -> 2].
        # fc1 keeps the 1024-d encoder hidden (encode() returns it). The
        # downstream head is a 3-layer MLP, replacing the old Linear(1024, 2).
        self.fc1 = nn.Linear(in_dim, 1024)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(1024, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, self.n_classes),
        )
        # Back-compat alias (older code referenced .fc2).
        self.fc2 = self.head

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 3:
            raise ValueError(
                f"TrajectoryMLP expects [B, L, H], got shape {tuple(trajectory.shape)}"
            )
        if trajectory.shape[1] != self.layer_count or trajectory.shape[2] != self.hidden_dim:
            raise ValueError(
                f"TrajectoryMLP got shape {tuple(trajectory.shape)} but was built for "
                f"[*, {self.layer_count}, {self.hidden_dim}]"
            )
        x = trajectory.flatten(start_dim=1)
        h = self.act(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h)

    @torch.no_grad()
    def encode(self, trajectory: torch.Tensor) -> torch.Tensor:
        x = trajectory.flatten(start_dim=1)
        return self.act(self.fc1(x))  # [B, 1024]

    @torch.no_grad()
    def export_embedding(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.encode(trajectory)
