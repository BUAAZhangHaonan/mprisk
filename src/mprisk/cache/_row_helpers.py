"""Shared row-parsing helpers for cache manifest loaders."""

from __future__ import annotations

from typing import Any


def field_present(value: Any) -> bool:
    """Return True when a row value is neither None nor an empty string."""
    return value is not None and value != ""


def optional_string(value: Any) -> str | None:
    """Coerce a row value to ``str`` if present, else ``None``."""
    if not field_present(value):
        return None
    return str(value)
