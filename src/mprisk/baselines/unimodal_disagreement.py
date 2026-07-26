"""Unimodal disagreement baseline."""

from __future__ import annotations


def disagreement_score(left: float, right: float) -> float:
    return abs(left - right)
