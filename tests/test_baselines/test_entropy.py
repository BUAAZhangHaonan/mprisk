from __future__ import annotations

import pytest

from mprisk.baselines.output_entropy import entropy


def test_entropy_zero_for_certain_distribution() -> None:
    assert entropy([1.0, 0.0]) == pytest.approx(0.0, abs=1e-12)
