"""Build stable prompt cache keys."""

from __future__ import annotations

from mprisk.utils.hashing import stable_hash


def prompt_cache_key(model_key: str, prompt_id: str, sample_id: str) -> str:
    return stable_hash({"model_key": model_key, "prompt_id": prompt_id, "sample_id": sample_id})
