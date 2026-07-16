"""Exact spherical signed Joint Lean wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mprisk.state.spherical import EPSILON, compute_spherical_state


def compute_r(
    bundle_or_distance_to_v: Mapping[str, Any] | float,
    distance_to_ta: float | None = None,
    modality_distance: float | None = None,
    eps: float = EPSILON,
) -> float:
    if isinstance(bundle_or_distance_to_v, Mapping):
        if distance_to_ta is not None or modality_distance is not None:
            raise TypeError("bundle form of compute_r does not accept distance arguments")
        return compute_r_for_bundle(bundle_or_distance_to_v)
    if distance_to_ta is None or modality_distance is None:
        raise TypeError("distance form requires distance_to_v, distance_to_ta, and M1-M2 distance")
    return (float(distance_to_ta) - float(bundle_or_distance_to_v)) / (
        float(modality_distance) + eps
    )


def compute_r_for_bundle(bundle: Mapping[str, Any]) -> float:
    return float(compute_spherical_state(bundle)["R"])


def compute_r_batch(bundles: Sequence[Mapping[str, Any]]) -> list[float]:
    return [compute_r_for_bundle(bundle) for bundle in bundles]
