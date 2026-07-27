"""Strict atomic JSONL validation receipts for resumable artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

JSONL_RECEIPT_SCHEMA = "mprisk_jsonl_validation_receipt_v1"
SPHERICAL_EMBEDDING_REQUIRED_FIELDS = (
    "sample_id",
    "sample_type",
    "model_key",
    "protocol",
    "prompt_set_key",
    "calibration_split",
    "representation_split",
    "split_assignment_sha256",
    "repr_key",
    "encoder_checkpoint_sha256",
    "prompt_set_artifact_sha256",
    "embeddings",
    "relations",
    "sample_relation_feature",
    "prompt_count",
)
SPHERICAL_EMBEDDING_IDENTITY_FIELDS = ("sample_id",)


def receipt_path_for(path: str | Path) -> Path:
    return Path(path).with_suffix(".receipt.json")


def validate_jsonl(
    path: str | Path,
    *,
    required_fields: Sequence[str],
    identity_fields: Sequence[str],
    expected_rows: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every UTF-8 line and return deterministic receipt content."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    required = tuple(required_fields)
    identity = tuple(identity_fields)
    if not required or not identity or not set(identity).issubset(required):
        raise ValueError("JSONL required/identity field contracts are invalid")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    byte_offset = 0
    with manifest_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line_offset = byte_offset
            byte_offset += len(raw_line)
            if not raw_line.strip():
                raise ValueError(
                    f"{manifest_path}:line={line_number}:byte={line_offset}: blank JSONL line"
                )
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"{manifest_path}:line={line_number}:byte={line_offset + error.start}: "
                    "invalid UTF-8"
                ) from error
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                error_byte = line_offset + len(line[: error.pos].encode("utf-8"))
                raise ValueError(
                    f"{manifest_path}:line={line_number}:byte={error_byte}: "
                    f"invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{manifest_path}:line={line_number}:byte={line_offset}: "
                    "row must be a JSON object"
                )
            missing = [field for field in required if field not in value]
            if missing:
                raise ValueError(
                    f"{manifest_path}:line={line_number}:byte={line_offset}: "
                    f"missing required fields: {missing}"
                )
            row_identity = tuple(value[field] for field in identity)
            try:
                json.dumps(row_identity, ensure_ascii=True, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{manifest_path}:line={line_number}:byte={line_offset}: "
                    "identity fields must be JSON serializable"
                ) from error
            rows.append(value)
            identities.append(row_identity)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"{manifest_path}: row count mismatch: expected {expected_rows}, observed {len(rows)}"
        )
    encoded_identities = [
        json.dumps(item, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for item in identities
    ]
    if len(encoded_identities) != len(set(encoded_identities)):
        raise ValueError(f"{manifest_path}: identity fields are not unique: {list(identity)}")
    receipt = {
        "schema_name": JSONL_RECEIPT_SCHEMA,
        "artifact_path": str(manifest_path),
        "artifact_sha256": _sha256(manifest_path),
        "artifact_bytes": manifest_path.stat().st_size,
        "row_count": len(rows),
        "required_fields": list(required),
        "identity_fields": list(identity),
        "identity_sha256": _sha256_text(
            json.dumps(
                encoded_identities,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }
    return rows, receipt


def publish_jsonl_receipt(
    path: str | Path,
    *,
    required_fields: Sequence[str],
    identity_fields: Sequence[str],
    expected_rows: int,
    bindings: Mapping[str, Any],
    receipt_path: str | Path | None = None,
) -> Path:
    """Validate an atomically published JSONL artifact and atomically attest it."""
    artifact_path = Path(path).expanduser().resolve()
    _, receipt = validate_jsonl(
        artifact_path,
        required_fields=required_fields,
        identity_fields=identity_fields,
        expected_rows=expected_rows,
    )
    try:
        encoded_bindings = json.loads(
            json.dumps(bindings, ensure_ascii=True, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise TypeError("JSONL receipt bindings must be JSON serializable") from error
    if not isinstance(encoded_bindings, dict) or not encoded_bindings:
        raise ValueError("JSONL receipt bindings must be a non-empty mapping")
    receipt["bindings"] = encoded_bindings
    destination = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else receipt_path_for(artifact_path)
    )
    _atomic_json(destination, receipt)
    return destination


def read_validated_jsonl(
    path: str | Path,
    *,
    required_fields: Sequence[str],
    identity_fields: Sequence[str],
    expected_bindings: Mapping[str, Any] | None = None,
    receipt_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Read only an artifact whose current bytes still pass its receipt."""
    artifact_path = Path(path).expanduser().resolve()
    receipt_file = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else receipt_path_for(artifact_path)
    )
    if not receipt_file.is_file():
        raise FileNotFoundError(f"JSONL validation receipt is missing: {receipt_file}")
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSONL validation receipt: {receipt_file}: {error}") from error
    if not isinstance(receipt, dict) or receipt.get("schema_name") != JSONL_RECEIPT_SCHEMA:
        raise ValueError(f"Unsupported JSONL validation receipt: {receipt_file}")
    rows, observed = validate_jsonl(
        artifact_path,
        required_fields=required_fields,
        identity_fields=identity_fields,
        expected_rows=receipt.get("row_count"),
    )
    for field in (
        "artifact_path",
        "artifact_sha256",
        "artifact_bytes",
        "row_count",
        "required_fields",
        "identity_fields",
        "identity_sha256",
    ):
        if receipt.get(field) != observed[field]:
            raise ValueError(f"JSONL receipt mismatch for {field}: {artifact_path}")
    if expected_bindings is not None and receipt.get("bindings") != dict(expected_bindings):
        raise ValueError(f"JSONL receipt binding mismatch: {artifact_path}")
    if not isinstance(receipt.get("bindings"), dict) or not receipt["bindings"]:
        raise ValueError(f"JSONL receipt has no bindings: {receipt_file}")
    return rows


def write_atomic_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write complete ASCII-safe JSONL bytes with fsync and atomic replacement."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
