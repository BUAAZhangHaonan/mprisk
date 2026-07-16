"""Text helpers."""

from __future__ import annotations


def squash_whitespace(text: str) -> str:
    return " ".join(text.split())
