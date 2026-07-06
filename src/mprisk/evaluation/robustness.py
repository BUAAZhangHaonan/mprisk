"""Robustness summary helpers."""

from __future__ import annotations


def max_absolute_shift(values_a: list[float], values_b: list[float]) -> float:
    return max((abs(a - b) for a, b in zip(values_a, values_b, strict=True)), default=0.0)
