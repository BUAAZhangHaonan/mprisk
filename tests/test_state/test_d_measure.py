from __future__ import annotations

import math

import pytest

from mprisk.state.d_measure import compute_d, compute_d_for_bundle


def _bundle() -> dict[str, object]:
    return {
        "sample_id": "case-1",
        "embeddings": {
            "M1": {"p1": [0.0, 0.0], "p2": [0.0, 2.0]},
            "M2": {"p1": [3.0, 0.0], "p2": [3.0, 4.0]},
            "M12": {"p1": [1.0, 1.0], "p2": [1.0, 3.0]},
        },
    }


def test_compute_d_for_bundle_uses_m1_m2_centers() -> None:
    assert compute_d_for_bundle(_bundle()) == pytest.approx(math.sqrt(10.0))


def test_compute_d_accepts_embedding_bundle_for_new_api() -> None:
    assert compute_d(_bundle()) == pytest.approx(math.sqrt(10.0))


def test_compute_d_keeps_legacy_vector_distance() -> None:
    assert compute_d([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)
