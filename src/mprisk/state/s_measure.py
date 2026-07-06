"""Template sensitivity measure."""

from __future__ import annotations


def compute_s(template_variances: list[float]) -> float:
    if not template_variances:
        return 0.0
    return sum(template_variances) / len(template_variances)
