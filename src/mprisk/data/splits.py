"""Deterministic split helpers."""

from __future__ import annotations

import hashlib


def stable_bucket(key: str, modulo: int = 100) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def assign_split(group_key: str) -> str:
    bucket = stable_bucket(group_key)
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"
