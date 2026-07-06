from __future__ import annotations

from mprisk.data.splits import assign_split


def test_assign_split_is_stable() -> None:
    assert assign_split("sample-1") == assign_split("sample-1")
