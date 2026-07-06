"""Aggregate state assignments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from mprisk.state.d_measure import compute_d_for_bundle
from mprisk.state.patterns import StatePattern, StateThresholds, assign_state
from mprisk.state.r_measure import compute_r_for_bundle
from mprisk.state.s_measure import compute_s_for_bundle


def count_patterns(patterns: list[StatePattern]) -> dict[str, int]:
    counts = Counter(pattern.value for pattern in patterns)
    return dict(counts)


def compute_state_row(bundle: Mapping[str, Any], thresholds: StateThresholds) -> dict[str, Any]:
    s_scores = compute_s_for_bundle(bundle)
    d_score = compute_d_for_bundle(bundle)
    r_score = compute_r_for_bundle(bundle)
    pattern = assign_state(s_scores["S_mean"], d_score, r_score, thresholds)

    row: dict[str, Any] = {
        "sample_id": bundle.get("sample_id"),
        "sample_type": bundle.get("sample_type"),
        "model_key": bundle.get("model_key"),
        "protocol": bundle.get("protocol"),
        "prompt_set_key": bundle.get("prompt_set_key"),
        "repr_key": bundle.get("repr_key"),
        **s_scores,
        "D": d_score,
        "R": r_score,
        "pattern": pattern.value,
    }
    return row


def aggregate_state_rows(
    bundles: Sequence[Mapping[str, Any]],
    thresholds: StateThresholds,
) -> list[dict[str, Any]]:
    return [compute_state_row(bundle, thresholds) for bundle in bundles]
