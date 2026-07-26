"""Shared parsing helpers for generated answers."""

from __future__ import annotations


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().split())
