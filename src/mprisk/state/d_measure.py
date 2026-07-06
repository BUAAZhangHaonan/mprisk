"""Split-strength measure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mprisk.state.s_measure import compute_view_center, euclidean_distance, view_prompt_embeddings


def compute_d(m1: Sequence[float] | Mapping[str, Any], m2: Sequence[float] | None = None) -> float:
    if m2 is None:
        if isinstance(m1, Mapping):
            return compute_d_for_bundle(m1)
        raise TypeError("compute_d requires either an embedding bundle or two vectors")
    return euclidean_distance(m1, m2)


def compute_d_for_bundle(bundle: Mapping[str, Any]) -> float:
    c_m1 = compute_view_center(view_prompt_embeddings(bundle, "M1"))
    c_m2 = compute_view_center(view_prompt_embeddings(bundle, "M2"))
    return compute_d(c_m1, c_m2)


def compute_d_batch(bundles: Sequence[Mapping[str, Any]]) -> list[float]:
    return [compute_d_for_bundle(bundle) for bundle in bundles]
