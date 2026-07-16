"""Full-layer trajectory helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np


Trajectory: TypeAlias = list[list[float]]
Embedding: TypeAlias = list[float]
TrajectoryBundleInput: TypeAlias = Mapping[str, Any]


class TrajectoryEncoder(Protocol):
    repr_key: str

    def encode(self, trajectory: Trajectory) -> Embedding:
        """Encode a [layer_count, hidden_dim] trajectory."""


def l2_normalize(vector: list[float], eps: float = 1e-12) -> list[float]:
    array = _as_finite_vector(vector)
    norm = float(np.linalg.norm(array))
    denom = max(norm, eps)
    return (array / denom).astype(float).tolist()


def normalize_trajectory(trajectory: Trajectory) -> Trajectory:
    array = _as_valid_trajectory(trajectory)
    normalized = _row_l2_normalize(array)
    return normalized.astype(float).tolist()


def raw_layernorm_mean(trajectory: Trajectory) -> Embedding:
    """Normalize each layer, average layers, then normalize the embedding."""
    normalized_layers = _row_l2_normalize(_as_valid_trajectory(trajectory))
    embedding = normalized_layers.mean(axis=0)
    return _l2_normalize_array(embedding).astype(float).tolist()


def raw_layernorm_flat(trajectory: Trajectory) -> Embedding:
    """Normalize each layer, flatten all layers, then normalize the embedding."""
    normalized_layers = _row_l2_normalize(_as_valid_trajectory(trajectory))
    embedding = normalized_layers.reshape(-1)
    return _l2_normalize_array(embedding).astype(float).tolist()


@dataclass(frozen=True)
class RawTrajectoryEncoder:
    """Non-training trajectory encoder for first-pass S/D/R consumers."""

    repr_key: str

    def encode(self, trajectory: Trajectory) -> Embedding:
        if self.repr_key == "raw_layernorm_mean":
            return raw_layernorm_mean(trajectory)
        if self.repr_key == "raw_layernorm_flat":
            return raw_layernorm_flat(trajectory)
        raise ValueError(f"Unknown raw trajectory representation: {self.repr_key}")


def encode_trajectory_bundle(
    bundle: TrajectoryBundleInput | object,
    *,
    encoder: TrajectoryEncoder,
) -> dict[str, Embedding] | dict[str, dict[str, Embedding]]:
    """Encode a bundle and require all embeddings in the bundle to share shape."""
    normalized_bundle = _normalize_bundle_input(bundle)
    encoded = _encode_bundle_mapping(normalized_bundle, encoder)
    shapes = _collect_embedding_shapes(encoded)
    if len(shapes) != 1:
        raise ValueError("All trajectories in the same bundle must have the same embedding shape")
    return encoded


def _as_valid_trajectory(trajectory: Trajectory) -> np.ndarray:
    try:
        array = np.asarray(trajectory, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory must have shape [layer_count, hidden_dim]") from exc
    if array.ndim != 2:
        raise ValueError("trajectory must have shape [layer_count, hidden_dim]")
    layer_count, hidden_dim = array.shape
    if layer_count == 0 or hidden_dim == 0:
        raise ValueError("trajectory must have non-empty layer and hidden dimensions")
    if not np.isfinite(array).all():
        raise ValueError("trajectory must contain only finite values")
    return array


def _as_finite_vector(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("vector must contain only finite values")
    return array


def _row_l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def _l2_normalize_array(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(array))
    return array / max(norm, eps)


def _normalize_bundle_input(bundle: TrajectoryBundleInput | object) -> TrajectoryBundleInput:
    if isinstance(bundle, Mapping):
        return bundle
    required_attrs = ("m1_trajectory", "m2_trajectory", "m12_trajectory")
    if all(hasattr(bundle, attr) for attr in required_attrs):
        return {
            "M1": getattr(bundle, "m1_trajectory"),
            "M2": getattr(bundle, "m2_trajectory"),
            "M12": getattr(bundle, "m12_trajectory"),
        }
    raise ValueError("bundle must be a mapping or expose m1_trajectory/m2_trajectory/m12_trajectory")


def _encode_bundle_mapping(
    bundle: TrajectoryBundleInput,
    encoder: TrajectoryEncoder,
) -> dict[str, Embedding] | dict[str, dict[str, Embedding]]:
    encoded: dict[str, Embedding] | dict[str, dict[str, Embedding]] = {}
    for key, value in bundle.items():
        if isinstance(value, Mapping):
            encoded[key] = _encode_bundle_mapping(value, encoder)  # type: ignore[assignment]
        else:
            encoded[key] = encoder.encode(value)  # type: ignore[assignment, arg-type]
    return encoded


def _collect_embedding_shapes(
    encoded: Mapping[str, Embedding] | Mapping[str, Mapping[str, Embedding]],
) -> set[tuple[int]]:
    shapes: set[tuple[int]] = set()
    for value in encoded.values():
        if isinstance(value, Mapping):
            shapes.update(_collect_embedding_shapes(value))
        else:
            shapes.add((len(value),))
    return shapes
