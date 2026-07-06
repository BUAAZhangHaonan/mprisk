"""Stage 1 structured-judgment correctness."""

from __future__ import annotations


def exact_match(prediction: str, target: str) -> bool:
    return prediction.strip().lower() == target.strip().lower()
