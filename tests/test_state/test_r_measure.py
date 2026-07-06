from __future__ import annotations

import math

import pytest

from mprisk.state.r_measure import compute_r, compute_r_for_bundle


def _bundle() -> dict[str, object]:
    return {
        "sample_id": "case-1",
        "embeddings": {
            "M1": {"p1": [0.0, 0.0], "p2": [0.0, 2.0]},
            "M2": {"p1": [3.0, 0.0], "p2": [3.0, 4.0]},
            "M12": {"p1": [1.0, 1.0], "p2": [1.0, 3.0]},
        },
    }


def test_compute_r_for_bundle_uses_m12_relative_to_m1_and_m2() -> None:
    d1 = math.sqrt(2.0)
    d2 = 2.0

    assert compute_r_for_bundle(_bundle()) == pytest.approx((d2 - d1) / (d1 + d2))


def test_compute_r_accepts_embedding_bundle_for_new_api() -> None:
    d1 = math.sqrt(2.0)
    d2 = 2.0

    assert compute_r(_bundle()) == pytest.approx((d2 - d1) / (d1 + d2))


def test_compute_r_keeps_legacy_distance_ratio() -> None:
    assert compute_r(1.0, 3.0) == pytest.approx(0.5)
