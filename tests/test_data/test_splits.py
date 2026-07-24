from __future__ import annotations

from mprisk.data.splits import assign_split


def test_assign_split_is_stable() -> None:
    first = assign_split("sample-1")
    second = assign_split("sample-1")
    assert first == second
    assert first in {"train", "val", "test"}
