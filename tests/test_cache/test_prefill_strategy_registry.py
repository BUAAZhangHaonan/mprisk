from __future__ import annotations

from types import SimpleNamespace

import pytest

from mprisk.cache.kv_prefill import QwenVlPromptKvPrefillExtractor
from mprisk.cache.prefill_strategy_registry import (
    create_prompt_kv_extractor,
    get_prompt_kv_extractor,
)


def test_registry_exposes_only_exact_qwen_vl_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr("mprisk.cache.kv_prefill.site.ENABLE_USER_SITE", False)
    wrapper = SimpleNamespace(family="qwen_vl")
    extractor = create_prompt_kv_extractor(wrapper, verbose=False)
    assert isinstance(extractor, QwenVlPromptKvPrefillExtractor)
    assert get_prompt_kv_extractor("qwen_vl") is QwenVlPromptKvPrefillExtractor


@pytest.mark.parametrize("family", ["internvl", "qwen_omni"])
def test_registry_rejects_unimplemented_family_contracts(family: str) -> None:
    with pytest.raises(ValueError, match="No exact prompt-prefix KV extractor"):
        get_prompt_kv_extractor(family)
