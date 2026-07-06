"""Template sensitivity measure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


VIEW_KEYS = ("M1", "M2", "M12")


def _as_vector(value: Sequence[float]) -> list[float]:
    return [float(item) for item in value]


def euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension")
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) ** 0.5


def view_prompt_embeddings(bundle: Mapping[str, Any], view_key: str) -> list[list[float]]:
    embeddings = bundle.get("embeddings")
    if not isinstance(embeddings, Mapping):
        raise ValueError("Embedding bundle must contain an embeddings mapping")

    view_embeddings = embeddings.get(view_key)
    if not isinstance(view_embeddings, Mapping) or not view_embeddings:
        raise ValueError(f"Embedding bundle is missing non-empty {view_key} embeddings")

    vectors = [_as_vector(vector) for vector in view_embeddings.values()]
    dimension = len(vectors[0])
    if dimension == 0:
        raise ValueError(f"{view_key} embeddings must not be empty vectors")
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError(f"All {view_key} embeddings must have the same dimension")
    return vectors


def compute_view_center(view_embeddings: Mapping[str, Sequence[float]] | Sequence[Sequence[float]]) -> list[float]:
    vectors = (
        [_as_vector(vector) for vector in view_embeddings.values()]
        if isinstance(view_embeddings, Mapping)
        else [_as_vector(vector) for vector in view_embeddings]
    )
    if not vectors:
        raise ValueError("Cannot compute a center from empty embeddings")

    dimension = len(vectors[0])
    if dimension == 0:
        raise ValueError("Embedding vectors must not be empty")
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("All embeddings must have the same dimension")

    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)]


def compute_view_dispersion(
    view_embeddings: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
) -> float:
    vectors = (
        [_as_vector(vector) for vector in view_embeddings.values()]
        if isinstance(view_embeddings, Mapping)
        else [_as_vector(vector) for vector in view_embeddings]
    )
    center = compute_view_center(vectors)
    return sum(euclidean_distance(vector, center) for vector in vectors) / len(vectors)


def compute_s(template_variances: Sequence[float] | Mapping[str, Any]) -> float | dict[str, float]:
    if isinstance(template_variances, Mapping) and "embeddings" in template_variances:
        return compute_s_for_bundle(template_variances)
    if not template_variances:
        return 0.0
    return sum(template_variances) / len(template_variances)


def compute_s_for_bundle(bundle: Mapping[str, Any]) -> dict[str, float]:
    scores = {
        f"S_{view_key}": compute_view_dispersion(view_prompt_embeddings(bundle, view_key))
        for view_key in VIEW_KEYS
    }
    scores["S_mean"] = sum(scores[f"S_{view_key}"] for view_key in VIEW_KEYS) / len(VIEW_KEYS)
    return scores


def compute_s_batch(bundles: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    return [compute_s_for_bundle(bundle) for bundle in bundles]
