"""Post-hoc full-response analysis baseline."""

from __future__ import annotations


def response_length(text: str) -> int:
    return len(text.split())
