"""Representation training configuration types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    embedding_dim: int = 128
    loss: str = "proxy_anchor"
    support_head: str = "radius_ball"
