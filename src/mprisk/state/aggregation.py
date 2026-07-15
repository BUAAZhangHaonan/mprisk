"""Aggregate state assignments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from mprisk.state.patterns import StatePattern, StateThresholds, assign_state
from mprisk.state.spherical import compute_spherical_state


def count_patterns(patterns: list[StatePattern]) -> dict[str, int]:
    counts = Counter(pattern.value for pattern in patterns)
    return dict(counts)


def compute_state_row(bundle: Mapping[str, Any], thresholds: StateThresholds) -> dict[str, Any]:
    state = compute_spherical_state(bundle)
    pattern = assign_state(
        state["S_mean"], state["D"], state["R"], thresholds, delta_i=state["delta_i"]
    )

    row: dict[str, Any] = {
        "sample_id": bundle.get("sample_id"),
        "sample_type": bundle.get("sample_type"),
        "model_key": bundle.get("model_key"),
        "protocol": bundle.get("protocol"),
        "prompt_set_key": bundle.get("prompt_set_key"),
        "repr_key": bundle.get("repr_key"),
        **state,
        "pattern": pattern.value,
    }
    return row


def aggregate_state_rows(
    bundles: Sequence[Mapping[str, Any]],
    thresholds: StateThresholds,
) -> list[dict[str, Any]]:
    return [compute_state_row(bundle, thresholds) for bundle in bundles]
