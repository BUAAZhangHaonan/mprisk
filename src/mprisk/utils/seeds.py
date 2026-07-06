"""Random seed helpers."""

from __future__ import annotations

import random


def seed_python(seed: int) -> None:
    random.seed(seed)
