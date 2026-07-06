"""Four state patterns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class StatePattern(str, Enum):
    CONFUSION = "Confusion"
    CONSENSUS = "Consensus"
    BALANCED = "Balanced"
    DOMINANT = "Dominant"


@dataclass(frozen=True)
class StateThresholds:
    kappa: float
    tau: float
    delta: float

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> StateThresholds:
        missing = {"kappa", "tau", "delta"} - set(config)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Threshold config is missing required key(s): {names}")
        return cls(
            kappa=float(config["kappa"]),
            tau=float(config["tau"]),
            delta=float(config["delta"]),
        )


def load_thresholds_config(source: StateThresholds | Mapping[str, Any] | str | Path) -> StateThresholds:
    if isinstance(source, StateThresholds):
        return source
    if isinstance(source, Mapping):
        return StateThresholds.from_dict(source)

    raw_source = str(source)
    path = Path(raw_source)
    if path.exists():
        return StateThresholds.from_dict(json.loads(path.read_text()))
    return StateThresholds.from_dict(json.loads(raw_source))


def assign_state(
    s: float,
    d: float,
    r: float,
    thresholds: StateThresholds | Mapping[str, Any],
) -> StatePattern:
    thresholds = load_thresholds_config(thresholds)
    if s > thresholds.kappa:
        return StatePattern.CONFUSION
    if d <= thresholds.tau:
        return StatePattern.CONSENSUS
    if abs(r) < thresholds.delta:
        return StatePattern.BALANCED
    return StatePattern.DOMINANT
