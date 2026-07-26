"""SP-MLP baseline encoders (single-point, last-layer hidden state -> MLP head).

Extracted from the original ``mprisk.representation.baselines`` module (P4-F refactor)
to keep each encoder family in its own file. The public surface stays
re-exported from ``mprisk.representation.baselines`` — callers should keep importing
from there.

Architecture is intentionally pure:
  * no LayerNorm anywhere,
  * no unit-norm / sphere normalization on the hidden representation.

Trajectory contract: ``[B, L, H]``. The penultimate representation is the
raw hidden_dim-d last-layer hidden state, handed directly to the
downstream M/N probe.
"""

from __future__ import annotations

import torch
from torch import nn


__all__ = ["SinglePointMLP", "SinglePointLayerX"]


class SinglePointMLP(nn.Module):
    """M12 last-layer hidden state -> Linear(hidden_dim, 2) -> 2 logits.

    No LayerNorm, no sphere normalization, no hidden projection. The
    penultimate representation is the raw hidden_dim-d M12 last-layer
    hidden state (e.g. 4096 for Qwen3-VL), handed directly to the
    downstream M/N probe.
    """

    architecture_version: str = "baseline_sp_mlp"

    LAST_LAYER_INDEX = -1

    def __init__(
        self,
        hidden_dim: int,
        n_classes: int = 2,
        dropout: float = 0.0,
        *,
        layer_count: int | None = None,  # accepted for API parity, unused
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_classes = int(n_classes)
        self.layer_count = layer_count
        # 3-layer MLP head: hidden_dim -> 128 -> 32 -> n_classes.
        # canonical_rerun spec: uniform head structure across baselines.
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, self.n_classes),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """trajectory: [B, L, H] -> logits [B, n_classes]."""
        if trajectory.ndim != 3:
            raise ValueError(
                f"SinglePointMLP expects [B, L, H], got shape {tuple(trajectory.shape)}"
            )
        x = trajectory[:, self.LAST_LAYER_INDEX, :]  # take last layer
        return self.classifier(x)

    @torch.no_grad()
    def encode(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Return the hidden_dim-d penultimate representation [B, hidden_dim]."""
        x = trajectory[:, self.LAST_LAYER_INDEX, :]
        return x

    # Back-compat alias for any external caller that imports the old name.
    @torch.no_grad()
    def export_embedding(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.encode(trajectory)


class SinglePointLayerX(nn.Module):
    """SinglePoint architecture with a configurable layer index.

    ``layer_idx=-1`` (default) is equivalent to ``SinglePointMLP``. Used for
    the per-layer sweep (Table 3 appendix).
    """

    architecture_version: str = "baseline_sp_layer_x"

    def __init__(
        self,
        hidden_dim: int,
        layer_idx: int = -1,
        n_classes: int = 2,
        dropout: float = 0.1,
        *,
        layer_count: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layer_idx = int(layer_idx)
        self.layer_count = layer_count
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 3:
            raise ValueError(
                f"SinglePointLayerX expects [B, L, H], got shape {tuple(trajectory.shape)}"
            )
        x = trajectory[:, self.layer_idx, :]
        return self.mlp(x)

    @torch.no_grad()
    def encode(self, trajectory: torch.Tensor) -> torch.Tensor:
        x = trajectory[:, self.layer_idx, :]
        return self.mlp[1](self.mlp[0](x))

    @torch.no_grad()
    def export_embedding(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.encode(trajectory)
