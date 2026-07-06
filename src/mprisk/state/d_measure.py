"""Split-strength measure."""

from __future__ import annotations


def compute_d(m1: list[float], m2: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(m1, m2, strict=True)) ** 0.5
