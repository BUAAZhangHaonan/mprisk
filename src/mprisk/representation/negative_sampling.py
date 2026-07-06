"""Negative sampling helpers."""

from __future__ import annotations


def negative_budget(total_positive: int, ratio: float) -> int:
    return max(0, int(round(total_positive * ratio)))
