"""Sample filtering helpers."""

from __future__ import annotations


def filter_by_sample_type(records: list[dict[str, object]], sample_type: str) -> list[dict[str, object]]:
    return [record for record in records if record.get("sample_type") == sample_type]
