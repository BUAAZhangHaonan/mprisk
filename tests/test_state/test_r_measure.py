from __future__ import annotations

import math

import pytest

from mprisk.state.r_measure import compute_r, compute_r_for_bundle


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


def test_compute_r_uses_m1_m2_geodesic_denominator_and_v_positive() -> None:
    assert compute_r_for_bundle(_bundle()) == pytest.approx(1 / 3)
    assert compute_r(_bundle()) == pytest.approx(1 / 3)
    assert compute_r(math.pi / 6, math.pi / 3, math.pi / 2) == pytest.approx(1 / 3)


def test_compute_r_rejects_old_two_distance_denominator() -> None:
    with pytest.raises(TypeError, match="M1-M2 distance"):
        compute_r(1.0, 3.0)
