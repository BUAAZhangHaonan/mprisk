"""Manifold-aware representation interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mprisk.representation.trajectory_encoder import Embedding, Trajectory


class ManifoldEncoder(Protocol):
    repr_key: str

    def encode(self, trajectory: Trajectory) -> Embedding:
        """Map one trajectory into a learned manifold embedding."""


@dataclass(frozen=True)
class ManifoldEncoderAdapter:
    """Interface shell for the planned trained manifold encoder."""

    repr_key: str

    def encode(self, trajectory: Trajectory) -> Embedding:
        raise NotImplementedError("Trained manifold trajectory encoders are not implemented yet")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
