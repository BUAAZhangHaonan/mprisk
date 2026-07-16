from __future__ import annotations

import math

import pytest

from mprisk.representation.adapters import get_trajectory_encoder
from mprisk.representation.trajectory_encoder import (
    raw_layernorm_flat,
    raw_layernorm_mean,
)


def assert_unit_norm(vector: list[float]) -> None:
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_raw_layernorm_mean_normalizes_layers_then_mean_then_embedding() -> None:
    embedding = raw_layernorm_mean([[3.0, 4.0], [0.0, 2.0]])

    assert embedding == pytest.approx([0.316227766, 0.948683298])
    assert_unit_norm(embedding)


def test_raw_layernorm_flat_normalizes_layers_then_concatenates_then_embedding() -> None:
    embedding = raw_layernorm_flat([[3.0, 4.0], [0.0, 2.0]])

    assert embedding == pytest.approx([0.424264069, 0.565685425, 0.0, 0.707106781])
    assert_unit_norm(embedding)


@pytest.mark.parametrize(
    "trajectory",
    [
        [],
        [[]],
        [[1.0, 2.0], [3.0]],
        [[1.0, float("nan")]],
        [[1.0, float("inf")]],
    ],
)
def test_raw_embedding_rejects_invalid_trajectories(trajectory: list[list[float]]) -> None:
    encoder = get_trajectory_encoder("raw_layernorm_mean")

    with pytest.raises(ValueError):
        encoder.encode(trajectory)


def test_get_trajectory_encoder_supports_raw_repr_keys() -> None:
    trajectory = [[1.0, 0.0], [0.0, 1.0]]

    mean_encoder = get_trajectory_encoder("raw_layernorm_mean")
    flat_encoder = get_trajectory_encoder("raw_layernorm_flat")

    assert mean_encoder.encode(trajectory) == pytest.approx([0.707106781, 0.707106781])
    assert flat_encoder.encode(trajectory) == pytest.approx([0.707106781, 0.0, 0.0, 0.707106781])


def test_get_trajectory_encoder_rejects_unknown_repr_key() -> None:
    with pytest.raises(ValueError, match="Unknown trajectory representation"):
        get_trajectory_encoder("trained_manifold")
