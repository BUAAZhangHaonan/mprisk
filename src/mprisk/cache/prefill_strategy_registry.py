"""Registry for exact prompt-prefix prefill strategies."""

from __future__ import annotations

from typing import Any, TypeAlias

from mprisk.cache.kv_prefill import QwenVlPromptKvPrefillExtractor

PromptKvExtractorFactory: TypeAlias = type[QwenVlPromptKvPrefillExtractor]

PROMPT_KV_EXTRACTOR_REGISTRY: dict[str, PromptKvExtractorFactory] = {
    "qwen_vl": QwenVlPromptKvPrefillExtractor,
}


def get_prompt_kv_extractor(family: str) -> PromptKvExtractorFactory:
    """Return the model-native exact KV extractor for ``family``.

    Unsupported families fail explicitly. They must not be routed through a
    visually similar cache contract because position and multimodal-prefix
    semantics differ across model families.
    """
    try:
        return PROMPT_KV_EXTRACTOR_REGISTRY[family]
    except KeyError as exc:
        supported = ", ".join(sorted(PROMPT_KV_EXTRACTOR_REGISTRY))
        raise ValueError(
            f"No exact prompt-prefix KV extractor for family {family!r}; "
            f"supported families: {supported}"
        ) from exc


def create_prompt_kv_extractor(wrapper: Any, **kwargs: Any) -> Any:
    family = getattr(wrapper, "family", None)
    if not isinstance(family, str) or not family:
        raise ValueError("Model wrapper must expose a non-empty family")
    return get_prompt_kv_extractor(family)(wrapper, **kwargs)
