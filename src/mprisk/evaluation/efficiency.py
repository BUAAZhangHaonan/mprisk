"""Timing and cost metrics."""

from __future__ import annotations


def speedup_seconds(posthoc_seconds: float, t0_seconds: float) -> float:
    return posthoc_seconds - t0_seconds
