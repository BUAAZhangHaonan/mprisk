"""Full-cache manifest schema helpers."""

from __future__ import annotations

REQUIRED_CACHE_FIELDS = (
    "model_key",
    "protocol",
    "dataset_key",
    "split",
    "condition",
    "artifact_uri",
)


def validate_cache_manifest_entry(entry: dict[str, object]) -> bool:
    return all(field in entry for field in REQUIRED_CACHE_FIELDS)
