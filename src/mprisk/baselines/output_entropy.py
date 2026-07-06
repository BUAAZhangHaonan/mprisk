"""Output entropy baseline."""

from __future__ import annotations

import math


def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    return -sum(p * math.log(max(p, eps)) for p in probabilities if p > 0)
