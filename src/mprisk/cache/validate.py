"""Hidden-state validation helpers."""

from __future__ import annotations

import math


def finite_vector(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)
