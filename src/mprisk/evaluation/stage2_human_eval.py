"""Stage 2 human-rating aggregation."""

from __future__ import annotations


def mean_rating(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
