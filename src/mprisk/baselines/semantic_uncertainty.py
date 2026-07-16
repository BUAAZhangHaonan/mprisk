"""Semantic uncertainty baseline contract."""

from __future__ import annotations


def semantic_cluster_count(labels: list[str]) -> int:
    return len(set(labels))
