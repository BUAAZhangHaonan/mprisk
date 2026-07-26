"""Registry for exact prompt-prefix prefill strategies."""

from __future__ import annotations

from typing import Any, TypeAlias

from mprisk.cache.kv_prefill import PREFILL_STRATEGY, QwenVlPromptKvPrefillExtractor

PromptKvExtractorFactory: TypeAlias = type[QwenVlPromptKvPrefillExtractor]

PROMPT_KV_EXTRACTOR_REGISTRY: dict[str, tuple[str, PromptKvExtractorFactory]] = {
    PREFILL_STRATEGY: ("qwen_vl", QwenVlPromptKvPrefillExtractor),
}


def get_prompt_kv_extractor(strategy: str, *, family: str) -> PromptKvExtractorFactory:
    """Return the model-native exact KV extractor for ``strategy`` and ``family``.

    Unsupported families fail explicitly. They must not be routed through a
    visually similar cache contract because position and multimodal-prefix
    semantics differ across model families.
    """
    try:
        registered_family, factory = PROMPT_KV_EXTRACTOR_REGISTRY[strategy]
    except KeyError as exc:
        supported = ", ".join(sorted(PROMPT_KV_EXTRACTOR_REGISTRY))
        raise ValueError(
            f"Unknown prompt-prefix KV strategy {strategy!r}; supported strategies: {supported}"
        ) from exc
    if family != registered_family:
        raise ValueError(
            f"Prefill strategy {strategy!r} requires family {registered_family!r}, "
            f"got {family!r}"
        )
    return factory


def create_prompt_kv_extractor(strategy: str, wrapper: Any, **kwargs: Any) -> Any:
    family = getattr(wrapper, "family", None)
    if not isinstance(family, str) or not family:
        raise ValueError("Model wrapper must expose a non-empty family")
    return get_prompt_kv_extractor(strategy, family=family)(wrapper, **kwargs)
