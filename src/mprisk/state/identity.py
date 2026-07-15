"""Strict identity binding for spherical calibration and scoring artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CALIBRATION_IDENTITY_FIELDS = (
    "model_key",
    "protocol",
    "prompt_set_key",
    "prompt_set_artifact_sha256",
    "repr_key",
    "encoder_checkpoint_sha256",
    "split_assignment_sha256",
    "embedding_manifest_sha256",
)
SOURCE_IDENTITY_FIELDS = CALIBRATION_IDENTITY_FIELDS[:-1]
SHA256_IDENTITY_FIELDS = frozenset(
    {
        "prompt_set_artifact_sha256",
        "encoder_checkpoint_sha256",
        "split_assignment_sha256",
        "embedding_manifest_sha256",
    }
)


def homogeneous_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] = CALIBRATION_IDENTITY_FIELDS,
) -> dict[str, str]:
    if not rows:
        raise ValueError("identity validation requires at least one row")
    identity: dict[str, str] = {}
    for field in fields:
        values = {str(row.get(field, "")) for row in rows}
        if len(values) != 1:
            raise ValueError(f"identity field {field} must be homogeneous")
        value = next(iter(values))
        if not value:
            raise ValueError(f"identity field {field} must be non-empty")
        if field in SHA256_IDENTITY_FIELDS and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"identity field {field} must be lowercase sha256")
        identity[field] = value
    return identity


def require_matching_identity(
    rows: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]
) -> None:
    actual = homogeneous_identity(rows)
    expected_identity = homogeneous_identity([expected])
    for field in CALIBRATION_IDENTITY_FIELDS:
        if actual[field] != expected_identity[field]:
            raise ValueError(
                f"identity mismatch for {field}: scores={actual[field]} "
                f"calibration={expected_identity[field]}"
            )
