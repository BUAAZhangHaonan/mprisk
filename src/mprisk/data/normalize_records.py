"""Normalize dataset rows into the project sample contract."""

from __future__ import annotations


REQUIRED_SAMPLE_FIELDS = (
    "sample_id",
    "dataset_key",
    "source_id",
    "available_modalities",
    "sample_type",
    "split",
    "labels",
)


def has_required_sample_fields(record: dict[str, object]) -> bool:
    return all(field in record for field in REQUIRED_SAMPLE_FIELDS)
