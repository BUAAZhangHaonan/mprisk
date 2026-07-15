from __future__ import annotations

import math

import pytest

from mprisk.state.s_measure import compute_s, compute_s_for_bundle


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


def test_compute_s_is_exact_squared_geodesic_dispersion() -> None:
    scores = compute_s_for_bundle(_bundle())
    assert scores["S_M1"] == pytest.approx((math.pi / 6) ** 2)
    assert scores["S_M2"] == pytest.approx((math.pi / 12) ** 2)
    assert scores["S_M12"] == pytest.approx((math.pi / 18) ** 2)
    assert compute_s(_bundle()) == scores


def test_compute_s_rejects_legacy_scalar_variance_api() -> None:
    with pytest.raises(TypeError, match="spherical embedding bundle"):
        compute_s([1.0, 2.0, 3.0])  # type: ignore[arg-type]
