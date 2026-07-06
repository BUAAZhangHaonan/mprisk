"""Aggregate state assignments."""

from __future__ import annotations

from collections import Counter

from mprisk.state.patterns import StatePattern


def count_patterns(patterns: list[StatePattern]) -> dict[str, int]:
    counts = Counter(pattern.value for pattern in patterns)
    return dict(counts)
