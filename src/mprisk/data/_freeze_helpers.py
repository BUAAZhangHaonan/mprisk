"""Shared validation and serialization helpers for data freeze scripts.

These helpers were originally private to ``generated_archive_freeze`` and
re-exported (with underscore prefix) by ``archetype_canonical_meanings``.
They are gathered here as public functions so that the three freeze/archive
scripts (``generated_archive_freeze``, ``archetype_canonical_meanings``,
``delivery``) can share a single source of truth.

Naming convention follows :mod:`mprisk.utils.io`: functions are public
(no underscore prefix). Callers that want a short local alias can use
``as _name`` on import, matching the existing style in this package.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mprisk.utils.io import canonical_json, sha256_file


__all__ = [
    "artifact_payload",
    "hash_media",
    "json_bytes",
    "jsonl_bytes",
    "literal_assignment",
    "optional_text",
    "read_jsonl_strict",
    "require_executable",
    "require_file",
    "write_immutable_outputs",
]


def optional_text(row: dict[str, Any], field: str) -> str | None:
    """Return a stripped non-empty string for *field*, or ``None`` if absent.

    Raises ``TypeError`` if the value is present but not a string.
    """
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    stripped = value.strip()
    return stripped or None


def require_file(path: Path, *, label: str) -> None:
    """Require *path* to be a non-empty regular file (no symlink)."""
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} must not be empty: {path}")


def require_executable(path: Path, *, label: str) -> None:
    """Require *path* to be a regular file that is executable by the current user."""
    require_file(path, label=label)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{label} is not executable: {path}")


def read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file requiring every line to be a non-blank JSON object.

    Unlike :func:`mprisk.utils.io.read_jsonl`, blank lines are rejected.
    """
    require_file(path, label="source index")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number}: row must be an object")
            rows.append(payload)
    return rows


def hash_media(path: Path, cache: dict[Path, tuple[str, int]]) -> tuple[str, int]:
    """Return ``(sha256, size_bytes)`` for *path*, caching into *cache*."""
    if path not in cache:
        cache[path] = (sha256_file(path), path.stat().st_size)
    return cache[path]


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize *rows* as canonical JSONL bytes (one object per line)."""
    return ("".join(canonical_json(row) + "\n" for row in rows)).encode("utf-8")


def json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize *payload* as pretty-printed JSON bytes (sorted keys, trailing newline)."""
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def artifact_payload(path: Path, content: bytes) -> dict[str, Any]:
    """Build the ``{path, bytes, sha256}`` provenance payload for *content*."""
    return {
        "path": path.as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def literal_assignment(path: Path, name: str) -> Any:
    """Return the Python literal assigned to ``name = ...`` at module top level."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} assignment in {path}")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} in {path} must be a literal") from exc


def write_immutable_outputs(outputs: dict[Path, bytes]) -> None:
    """Atomically write freeze artifacts, refusing to overwrite divergent content.

    Each output is written via a temporary file in the same directory,
    ``fsync``'ed, then ``os.replace``'d into place. If a target already
    exists with **different** bytes, the whole call raises ``ValueError``;
    existing identical files are left untouched (idempotent).
    """
    conflicts = [
        path for path, content in outputs.items() if path.exists() and path.read_bytes() != content
    ]
    if conflicts:
        raise ValueError(
            "Immutable freeze outputs already exist with different content: "
            + ", ".join(str(path) for path in conflicts)
        )
    for path, content in outputs.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
