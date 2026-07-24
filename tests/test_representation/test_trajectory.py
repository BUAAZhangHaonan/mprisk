from __future__ import annotations

import math

from mprisk.representation.trajectory_encoder import l2_normalize


def test_l2_normalize_unit_norm() -> None:
    vector = l2_normalize([3.0, 4.0])
    assert math.isclose(float(vector[0]), 0.6, abs_tol=1e-6)
    assert math.isclose(float(vector[1]), 0.8, abs_tol=1e-6)
