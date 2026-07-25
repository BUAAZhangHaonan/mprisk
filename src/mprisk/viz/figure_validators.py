"""Pure-data and PDF validators for figure inputs and exports."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mprisk.utils.io import sha256_file as _sha256
from .figure_constants import FORBIDDEN_PDF_TEXT


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).casefold() == "true":
        return True
    if str(value).casefold() == "false":
        return False
    raise ValueError(f"expected true/false value, got {value!r}")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.casefold())


def _require_columns(rows: list[dict[str, Any]], columns: set[str]) -> None:
    missing = columns - set(rows[0])
    if missing:
        raise ValueError(f"figure input is missing columns: {', '.join(sorted(missing))}")


def _required_text(spec: Mapping[str, Any], field: str) -> str:
    value = spec.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"figure field {field} must be non-empty text")
    return value


def _validate_pdf_open(path: Path) -> None:
    completed = subprocess.run(
        ["pdfinfo", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"PDF validation failed for {path}: {completed.stderr.strip()}")


def _validate_pdf_text(path: Path) -> None:
    completed = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"PDF text extraction failed for {path}: {completed.stderr.strip()}")
    normalized = completed.stdout.casefold()
    matches = [term for term in FORBIDDEN_PDF_TEXT if term in normalized]
    if matches:
        raise ValueError(f"PDF contains forbidden text: {', '.join(matches)}")


