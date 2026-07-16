"""Classifier baseline placeholders."""

from __future__ import annotations


def threshold_score(score: float, threshold: float) -> int:
    return int(score >= threshold)
