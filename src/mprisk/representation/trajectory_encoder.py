"""Full-layer trajectory helpers."""

from __future__ import annotations


def l2_normalize(vector: list[float], eps: float = 1e-12) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    denom = max(norm, eps)
    return [value / denom for value in vector]


def normalize_trajectory(trajectory: list[list[float]]) -> list[list[float]]:
    return [l2_normalize(layer) for layer in trajectory]
