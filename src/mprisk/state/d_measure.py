"""Exact spherical Modality Split wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mprisk.state.spherical import compute_spherical_state


def compute_d(bundle: Mapping[str, Any]) -> float:
    if not isinstance(bundle, Mapping):
        raise TypeError(
            "compute_d requires the full synchronized bundle because D is dispersion-normalized"
        )
    return compute_d_for_bundle(bundle)


def compute_d_for_bundle(bundle: Mapping[str, Any]) -> float:
    return float(compute_spherical_state(bundle)["D"])


def compute_d_batch(bundles: Sequence[Mapping[str, Any]]) -> list[float]:
    return [compute_d_for_bundle(bundle) for bundle in bundles]
