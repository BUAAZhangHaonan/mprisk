"""KV-cache helper placeholders."""

from __future__ import annotations


def cache_reuse_supported(model_family: str) -> bool:
    return model_family in {"qwen_vl", "qwen_omni", "internvl"}
