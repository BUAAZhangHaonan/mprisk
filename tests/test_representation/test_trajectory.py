from __future__ import annotations

from mprisk.representation.trajectory_encoder import l2_normalize


def test_l2_normalize_unit_norm() -> None:
    vector = l2_normalize([3.0, 4.0])
    assert vector == [0.6, 0.8]
