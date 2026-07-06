"""Small statistical helpers."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
