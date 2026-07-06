"""Four state patterns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StatePattern(StrEnum):
    CONFUSION = "Confusion"
    CONSENSUS = "Consensus"
    BALANCED = "Balanced"
    DOMINANT = "Dominant"


@dataclass(frozen=True)
class StateThresholds:
    kappa: float
    tau: float
    delta: float


def assign_state(s: float, d: float, r: float, thresholds: StateThresholds) -> StatePattern:
    if s > thresholds.kappa:
        return StatePattern.CONFUSION
    if d <= thresholds.tau:
        return StatePattern.CONSENSUS
    if abs(r) < thresholds.delta:
        return StatePattern.BALANCED
    return StatePattern.DOMINANT
