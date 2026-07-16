"""Merge annotation tables with sample manifests."""

from __future__ import annotations


def merge_by_sample_id(
    samples: list[dict[str, object]], annotations: list[dict[str, object]]
) -> list[dict[str, object]]:
    annotation_by_id = {row["sample_id"]: row for row in annotations if "sample_id" in row}
    merged: list[dict[str, object]] = []
    for sample in samples:
        row = dict(sample)
        row["annotation"] = annotation_by_id.get(sample.get("sample_id"))
        merged.append(row)
    return merged
