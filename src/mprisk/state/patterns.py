"""Four state patterns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class StatePattern(StrEnum):
    CONFUSION = "Confusion"
    CONSENSUS = "Consensus"
    BALANCED = "Balanced"
    DOMINANT = "Dominant"


@dataclass(frozen=True)
class StateThresholds:
    kappa: float
    tau: float
    delta: float | None = None

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> StateThresholds:
        missing = {"kappa", "tau"} - set(config)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Threshold config is missing required key(s): {names}")
        return cls(
            kappa=float(config["kappa"]),
            tau=float(config["tau"]),
            delta=float(config["delta"]) if config.get("delta") is not None else None,
        )


def load_thresholds_config(
    source: StateThresholds | Mapping[str, Any] | str | Path,
) -> StateThresholds:
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
    *,
    delta_i: float | None = None,
) -> StatePattern:
    thresholds = load_thresholds_config(thresholds)
    if s > thresholds.kappa:
        return StatePattern.CONFUSION
    if d <= thresholds.tau:
        return StatePattern.CONSENSUS
    effective_delta = delta_i if delta_i is not None else thresholds.delta
    if effective_delta is None:
        raise ValueError("delta_i is required for spherical pattern assignment")
    if effective_delta < 0.0:
        raise ValueError("delta_i must be non-negative")
    if abs(r) <= effective_delta:
        return StatePattern.BALANCED
    return StatePattern.DOMINANT
