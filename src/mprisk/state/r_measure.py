"""Arbitration-bias measure."""

from __future__ import annotations


def compute_r(m1_distance: float, m2_distance: float, eps: float = 1e-12) -> float:
    return (m2_distance - m1_distance) / max(m1_distance + m2_distance, eps)
