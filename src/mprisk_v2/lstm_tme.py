"""V2 LSTM-based TME.

Replaces SequentialTrajectoryEncoderV1's 1-layer GRU (hidden=128) with a
2-layer bi-LSTM (hidden=256) + 2-layer MLP projection. Roughly 10x params.

Exposes the same forward contract so it slots into SphericalTMEV1 unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from mprisk.representation import relation_models as _rm
from mprisk.representation.relation_models import (
    SphericalTMEV1,
    TME_PROXY_ANCHOR_V1,
    strict_l2_normalize,
)


class LSTMSequentialEncoderV2(nn.Module):
    """Layer-L2 → 2-layer bi-LSTM → MLP projection → sphere."""

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_hidden_dim: int = 256,
        embed_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.lstm_layers = lstm_layers
        self.sequence_hidden_dim = sequence_hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=sequence_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # bi-LSTM final hidden = 2 * sequence_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * sequence_hidden_dim, sequence_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sequence_hidden_dim, embed_dim),
        )

    @staticmethod
    def normalize_layers(trajectories: torch.Tensor) -> torch.Tensor:
        return strict_l2_normalize(trajectories, stage="tme_v2_layer_input")

    def forward(
        self,
        trajectories: torch.Tensor,
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if trajectories.ndim < 3 or trajectories.shape[-1] != self.input_dim:
            raise ValueError(
                f"trajectories must end in input_dim={self.input_dim}, "
                f"got shape {tuple(trajectories.shape)}"
            )
        leading = trajectories.shape[:-2]
        layer_count = trajectories.shape[-2]
        normalized = self.normalize_layers(trajectories)
        flat = normalized.reshape(-1, layer_count, self.input_dim)
        _output, (h_n, _c_n) = self.lstm(flat)
        # h_n shape: [num_layers * 2 (bi), batch, hidden]
        # Take final layer's forward + backward hidden states
        last_layer_fwd = h_n[-2]   # forward of top layer
        last_layer_bwd = h_n[-1]   # backward of top layer
        final_hidden = torch.cat([last_layer_fwd, last_layer_bwd], dim=-1)
        projected = self.mlp(self.dropout(final_hidden))
        projected = projected.reshape(*leading, -1)
        return strict_l2_normalize(projected, stage="tme_v2_z_projection")


class SphericalTMEV2(nn.Module):
    """LSTM-based TME with same I/O contract as SphericalTMEV1."""

    architecture_version = "tme_lstm_proxy_anchor_v2"

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_hidden_dim: int = 256,
        condition_dim: int = 128,
        relation_dim: int = 64,
        dropout: float = 0.2,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.condition_encoder = LSTMSequentialEncoderV2(
            input_dim=input_dim,
            sequence_hidden_dim=sequence_hidden_dim,
            embed_dim=condition_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
        self.relation = _rm.OrderedLinearRelationV1(relation_dim=relation_dim)

    def forward(
        self,
        trajectories: torch.Tensor,
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _rm._validate_three_condition_trajectories(trajectories)
        condition_z = self.condition_encoder(trajectories, sample_ids=sample_ids)
        relation_r = self.relation(
            condition_z[:, 0],
            condition_z[:, 1],
            condition_z[:, 2],
            sample_ids=sample_ids,
        )
        return condition_z, relation_r

    # Required by mprisk.representation.training._stream_baseline_exports:
    def forward_features(self, trajectories: torch.Tensor) -> torch.Tensor:
        z, _ = self.forward(trajectories)
        return z

    @property
    def classifier(self):  # compatibility shim
        return _NoOpClassifier()


class _NoOpClassifier(nn.Module):
    """TME-V2 doesn't use a classifier (Proxy Anchor organizes the embedding
    space directly). mprisk.representation.training._stream_baseline_exports
    calls `model.classifier(features)` for baseline exports; we route that
    through a no-op so the export stream does not crash."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def install_v2_tme_factory() -> None:
    """Replace build_representation_model's TME branch with V2 (LSTM).

    Patches both relation_models.build_representation_model AND the bound name
    in representation.training (which captured the original at import time).
    """
    original = _rm.build_representation_model

    def patched(repr_key: str, **kwargs):
        if repr_key == TME_PROXY_ANCHOR_V1:
            kwargs.setdefault("sequence_hidden_dim", kwargs.pop("hidden_dim", 256))
            kwargs.setdefault("condition_dim", 128)
            kwargs.setdefault("relation_dim", 64)
            kwargs.setdefault("dropout", 0.2)
            return SphericalTMEV2(
                input_dim=kwargs["input_dim"],
                sequence_hidden_dim=kwargs["sequence_hidden_dim"],
                condition_dim=kwargs["condition_dim"],
                relation_dim=kwargs["relation_dim"],
                dropout=kwargs["dropout"],
            )
        return original(repr_key, **kwargs)

    _rm.build_representation_model = patched

    # Patch every module that imported the name.
    import mprisk.representation.training as _training_mod
    _training_mod.build_representation_model = patched
