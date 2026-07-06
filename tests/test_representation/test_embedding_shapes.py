from __future__ import annotations

import math

import pytest

from mprisk.representation.adapters import get_trajectory_encoder
from mprisk.representation.trajectory_encoder import encode_trajectory_bundle


def _trajectory(scale: float) -> list[list[float]]:
    return [
        [1.0 * scale, 2.0 * scale, 3.0 * scale],
        [4.0 * scale, 5.0 * scale, 6.0 * scale],
    ]


@pytest.mark.parametrize(
    ("repr_key", "expected_dim"),
    [
        ("raw_layernorm_mean", 3),
        ("raw_layernorm_flat", 6),
    ],
)
def test_m1_m2_m12_embeddings_have_consistent_shape(repr_key: str, expected_dim: int) -> None:
    encoder = get_trajectory_encoder(repr_key)
    embeddings = encode_trajectory_bundle(
        {
            "M1": _trajectory(1.0),
            "M2": _trajectory(2.0),
            "M12": _trajectory(3.0),
        },
        encoder=encoder,
    )

    shapes = {condition: len(embedding) for condition, embedding in embeddings.items()}
    assert shapes == {"M1": expected_dim, "M2": expected_dim, "M12": expected_dim}
    for embedding in embeddings.values():
        assert all(math.isfinite(value) for value in embedding)


def test_protocol_view_embeddings_have_consistent_shape() -> None:
    encoder = get_trajectory_encoder("raw_layernorm_flat")

    embeddings = encode_trajectory_bundle(
        {
            "P1": {
                "M1": _trajectory(1.0),
                "M2": _trajectory(2.0),
                "M12": _trajectory(3.0),
            },
            "P2": {
                "M1": _trajectory(4.0),
                "M2": _trajectory(5.0),
                "M12": _trajectory(6.0),
            },
        },
        encoder=encoder,
    )

    lengths = {
        condition: len(embedding)
        for protocol_embeddings in embeddings.values()
        for condition, embedding in protocol_embeddings.items()
    }
    assert set(lengths.values()) == {6}


def test_encode_trajectory_bundle_rejects_inconsistent_shapes() -> None:
    encoder = get_trajectory_encoder("raw_layernorm_mean")

    with pytest.raises(ValueError, match="same embedding shape"):
        encode_trajectory_bundle(
            {
                "M1": [[1.0, 2.0]],
                "M2": [[1.0, 2.0, 3.0]],
                "M12": [[1.0, 2.0]],
            },
            encoder=encoder,
        )
