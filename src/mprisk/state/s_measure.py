"""Exact spherical State Dispersion wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mprisk.state.spherical import compute_spherical_state, spherical_center, spherical_distance

VIEW_KEYS = ("M1", "M2", "M12")


def view_prompt_embeddings(bundle: Mapping[str, Any], view_key: str) -> list[list[float]]:
    embeddings = bundle.get("embeddings")
    if not isinstance(embeddings, Mapping):
        raise ValueError("Embedding bundle must contain an embeddings mapping")
    view_embeddings = embeddings.get(view_key)
    if not isinstance(view_embeddings, Mapping) or not view_embeddings:
        raise ValueError(f"Embedding bundle is missing non-empty {view_key} embeddings")
    return [[float(item) for item in vector] for vector in view_embeddings.values()]


def compute_view_center(
    view_embeddings: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
) -> list[float]:
    vectors = (
        list(view_embeddings.values())
        if isinstance(view_embeddings, Mapping)
        else list(view_embeddings)
    )
    return spherical_center(vectors)


def compute_view_dispersion(
    view_embeddings: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
) -> float:
    vectors = (
        list(view_embeddings.values())
        if isinstance(view_embeddings, Mapping)
        else list(view_embeddings)
    )
    center = spherical_center(vectors)
    return sum(spherical_distance(vector, center) ** 2 for vector in vectors) / len(vectors)


def compute_s(bundle: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(bundle, Mapping):
        raise TypeError("compute_s accepts only a synchronized spherical embedding bundle")
    return compute_s_for_bundle(bundle)


def compute_s_for_bundle(bundle: Mapping[str, Any]) -> dict[str, float]:
    state = compute_spherical_state(bundle)
    return {
        "S_M1": float(state["S_M1"]),
        "S_M2": float(state["S_M2"]),
        "S_M12": float(state["S_M12"]),
        "S_mean": float(state["S_mean"]),
    }


def compute_s_batch(bundles: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    return [compute_s_for_bundle(bundle) for bundle in bundles]
