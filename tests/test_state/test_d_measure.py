from __future__ import annotations

import math

import pytest

from mprisk.state.d_measure import compute_d, compute_d_for_bundle


def _unit(angle: float) -> list[float]:
    return [math.cos(angle), math.sin(angle)]


def _symmetric(center: float, spread: float) -> dict[str, list[float]]:
    return {"p1": _unit(center - spread), "p2": _unit(center + spread)}


def _bundle() -> dict[str, object]:
    return {
        "sample_id": "case-1",
        "sample_type": "Conflict",
        "embeddings": {
            "M1": _symmetric(0.0, math.pi / 6),
            "M2": _symmetric(math.pi / 2, math.pi / 12),
            "M12": _symmetric(math.pi / 6, math.pi / 18),
        },
    }


def test_compute_d_is_geodesic_split_normalized_by_dispersion() -> None:
    expected = (math.pi / 2) / math.sqrt((math.pi / 6) ** 2 + (math.pi / 12) ** 2)
    assert compute_d_for_bundle(_bundle()) == pytest.approx(expected)
    assert compute_d(_bundle()) == pytest.approx(expected)


def test_compute_d_rejects_legacy_two_vector_distance() -> None:
    with pytest.raises(TypeError, match="full synchronized bundle"):
        compute_d([1.0, 0.0])  # type: ignore[arg-type]
