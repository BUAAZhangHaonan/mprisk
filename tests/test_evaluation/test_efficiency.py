from __future__ import annotations

from mprisk.evaluation.efficiency import speedup_seconds


def test_speedup_seconds() -> None:
    assert speedup_seconds(10.0, 4.0) == 6.0
