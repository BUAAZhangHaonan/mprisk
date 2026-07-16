"""Versioned sample-relation representation models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

SINGLE_POINT_BINARY_V1 = "single_point_binary_v1"
TRAJECTORY_MLP_BINARY_V1 = "trajectory_mlp_binary_v1"
TME_PROXY_ANCHOR_V1 = "tme_proxy_anchor_v1"
TME_ARCHITECTURE_V1 = "layer_l2_gru_linear_relation_v1"
REPRESENTATION_KEYS = (
    SINGLE_POINT_BINARY_V1,
    TRAJECTORY_MLP_BINARY_V1,
    TME_PROXY_ANCHOR_V1,
)
SPHERICAL_EPS = 1e-12


def require_nonzero_vectors(
    vectors: torch.Tensor,
    *,
    stage: str,
    sample_ids: Sequence[str] | None = None,
    eps: float = SPHERICAL_EPS,
) -> torch.Tensor:
    if vectors.ndim < 2 or vectors.shape[-1] == 0:
        raise ValueError(f"stage={stage} vectors must have a non-empty final dimension")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError(f"stage={stage} vectors must contain only finite values")
    norms = torch.linalg.vector_norm(vectors, dim=-1)
    bad = torch.nonzero(norms <= eps, as_tuple=False)
    if bad.numel() == 0:
        return norms
    vector_index = tuple(int(value) for value in bad[0].detach().cpu().numpy())
    batch_index = vector_index[0]
    sample = (
        str(sample_ids[batch_index])
        if sample_ids is not None and batch_index < len(sample_ids)
        else f"batch_index_{batch_index}"
    )
    raise ValueError(
        f"stage={stage} sample={sample} vector_index={vector_index} norm must exceed {eps}"
    )


def strict_l2_normalize(
    vectors: torch.Tensor,
    *,
    stage: str,
    sample_ids: Sequence[str] | None = None,
    eps: float = SPHERICAL_EPS,
) -> torch.Tensor:
    norms = require_nonzero_vectors(
        vectors,
        stage=stage,
        sample_ids=sample_ids,
        eps=eps,
    )
    return vectors / norms.unsqueeze(-1)


def _validate_three_condition_trajectories(trajectories: torch.Tensor) -> None:
    if trajectories.ndim != 4 or trajectories.shape[1] != 3:
        raise ValueError(
            "trajectories must have shape [batch, 3, layer_count, hidden_dim]"
        )
    if trajectories.shape[0] == 0 or trajectories.shape[2] == 0 or trajectories.shape[3] == 0:
        raise ValueError("trajectory dimensions must be non-empty")
    if not bool(torch.isfinite(trajectories).all()):
        raise ValueError("trajectories must contain only finite values")


class SinglePointBinaryClassifierV1(nn.Module):
    """Ordinary A/C classifier over the final-layer point of all conditions."""

    architecture_version = SINGLE_POINT_BINARY_V1

    def __init__(self, *, input_dim: int) -> None:
        super().__init__()
        self.penultimate_dim = 3 * input_dim
        self.classifier = nn.Linear(self.penultimate_dim, 2)

    def forward_features(self, trajectories: torch.Tensor) -> torch.Tensor:
        _validate_three_condition_trajectories(trajectories)
        return trajectories[:, :, -1, :].flatten(start_dim=1)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(trajectories))


class TrajectoryMLPBinaryClassifierV1(nn.Module):
    """Ordinary A/C classifier over the complete normalized layer trajectories."""

    architecture_version = TRAJECTORY_MLP_BINARY_V1

    def __init__(
        self,
        *,
        input_dim: int,
        layer_count: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.penultimate_dim = hidden_dim
        self.feature_projection = nn.Linear(3 * layer_count * input_dim, hidden_dim)
        self.feature_activation = nn.GELU()
        self.feature_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward_features(self, trajectories: torch.Tensor) -> torch.Tensor:
        _validate_three_condition_trajectories(trajectories)
        if trajectories.shape[2] != self.layer_count:
            raise ValueError("trajectory layer_count does not match model configuration")
        normalized = strict_l2_normalize(
            trajectories,
            stage="trajectory_mlp_layer_input",
        )
        return self.feature_dropout(
            self.feature_activation(
                self.feature_projection(normalized.flatten(start_dim=1))
            )
        )

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(trajectories))


class SequentialTrajectoryEncoderV1(nn.Module):
    """Layer-L2 + one-layer GRU + compact projection condition encoder."""

    architecture_version = TME_ARCHITECTURE_V1

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_hidden_dim: int,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or sequence_hidden_dim <= 0 or embed_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.sequence = nn.GRU(
            input_size=input_dim,
            hidden_size=sequence_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(sequence_hidden_dim, embed_dim)

    @staticmethod
    def normalize_layers(
        trajectories: torch.Tensor,
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if trajectories.shape[-1] == 0:
            raise ValueError("hidden dimension must be non-empty")
        return strict_l2_normalize(
            trajectories,
            stage="tme_layer_input",
            sample_ids=sample_ids,
        )

    def forward(
        self,
        trajectories: torch.Tensor,
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if trajectories.ndim < 3 or trajectories.shape[-1] != self.input_dim:
            raise ValueError("condition trajectories must end in the configured hidden dimension")
        leading = trajectories.shape[:-2]
        layer_count = trajectories.shape[-2]
        normalized = self.normalize_layers(trajectories, sample_ids=sample_ids)
        flat = normalized.reshape(-1, layer_count, self.input_dim)
        _sequence, hidden = self.sequence(flat)
        projected = self.projection(self.dropout(hidden[-1]))
        projected = projected.reshape(*leading, -1)
        return strict_l2_normalize(
            projected,
            stage="tme_z_projection",
            sample_ids=sample_ids,
        )


def ordered_relation_features(
    z1: torch.Tensor,
    z2: torch.Tensor,
    z12: torch.Tensor,
    *,
    sample_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """Return ordered u=[1-z1.z2, 1-z12.z1, 1-z12.z2]."""
    if z1.shape != z2.shape or z1.shape != z12.shape or z1.ndim != 2:
        raise ValueError("z1, z2, and z12 must have the same [batch, dim] shape")
    for stage, vectors in (
        ("ordered_relation_z1", z1),
        ("ordered_relation_z2", z2),
        ("ordered_relation_z12", z12),
    ):
        require_nonzero_vectors(vectors, stage=stage, sample_ids=sample_ids)
    return torch.stack(
        (
            1.0 - (z1 * z2).sum(dim=-1),
            1.0 - (z12 * z1).sum(dim=-1),
            1.0 - (z12 * z2).sum(dim=-1),
        ),
        dim=-1,
    )


class OrderedLinearRelationV1(nn.Module):
    """The only TME relation head: r=normalize(Wu+b), with no concatenation or activation."""

    def __init__(self, *, relation_dim: int) -> None:
        super().__init__()
        if relation_dim <= 0:
            raise ValueError("relation_dim must be positive")
        self.projection = nn.Linear(3, relation_dim)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z12: torch.Tensor,
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        u = ordered_relation_features(z1, z2, z12, sample_ids=sample_ids)
        return strict_l2_normalize(
            self.projection(u),
            stage="ordered_relation_r_projection",
            sample_ids=sample_ids,
        )


class SphericalTMEV1(nn.Module):
    """Three shared condition encodings followed by the ordered linear relation map."""

    architecture_version = TME_ARCHITECTURE_V1

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_hidden_dim: int,
        condition_dim: int,
        relation_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.condition_encoder = SequentialTrajectoryEncoderV1(
            input_dim=input_dim,
            sequence_hidden_dim=sequence_hidden_dim,
            embed_dim=condition_dim,
            dropout=dropout,
        )
        self.relation = OrderedLinearRelationV1(relation_dim=relation_dim)

    def forward(
        self,
        trajectories: torch.Tensor,
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_three_condition_trajectories(trajectories)
        condition_z = self.condition_encoder(trajectories, sample_ids=sample_ids)
        relation_r = self.relation(
            condition_z[:, 0],
            condition_z[:, 1],
            condition_z[:, 2],
            sample_ids=sample_ids,
        )
        return condition_z, relation_r


def build_representation_model(
    repr_key: str,
    *,
    input_dim: int,
    layer_count: int,
    hidden_dim: int,
    condition_dim: int = 64,
    relation_dim: int = 32,
    dropout: float = 0.0,
) -> nn.Module:
    if repr_key == SINGLE_POINT_BINARY_V1:
        return SinglePointBinaryClassifierV1(input_dim=input_dim)
    if repr_key == TRAJECTORY_MLP_BINARY_V1:
        return TrajectoryMLPBinaryClassifierV1(
            input_dim=input_dim,
            layer_count=layer_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if repr_key == TME_PROXY_ANCHOR_V1:
        return SphericalTMEV1(
            input_dim=input_dim,
            sequence_hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            relation_dim=relation_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown representation key: {repr_key}")
