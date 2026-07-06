"""Arbitration-bias measure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mprisk.state.s_measure import compute_view_center, euclidean_distance, view_prompt_embeddings


def compute_r(
    m1_distance: float | Mapping[str, Any],
    m2_distance: float | None = None,
    eps: float = 1e-12,
) -> float:
    if m2_distance is None:
        if isinstance(m1_distance, Mapping):
            return compute_r_for_bundle(m1_distance, eps=eps)
        raise TypeError("compute_r requires either an embedding bundle or two distances")
    return (m2_distance - m1_distance) / (m1_distance + m2_distance + eps)


def compute_r_for_bundle(bundle: Mapping[str, Any], eps: float = 1e-12) -> float:
    c_m1 = compute_view_center(view_prompt_embeddings(bundle, "M1"))
    c_m2 = compute_view_center(view_prompt_embeddings(bundle, "M2"))
    c_m12 = compute_view_center(view_prompt_embeddings(bundle, "M12"))
    d1 = euclidean_distance(c_m12, c_m1)
    d2 = euclidean_distance(c_m12, c_m2)
    return compute_r(d1, d2, eps=eps)


def compute_r_batch(bundles: Sequence[Mapping[str, Any]], eps: float = 1e-12) -> list[float]:
    return [compute_r_for_bundle(bundle, eps=eps) for bundle in bundles]
