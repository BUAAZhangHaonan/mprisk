from __future__ import annotations

from mprisk.prompts.prompt_cache_builder import (
    build_prompt_cache_manifest_row,
    prompt_cache_key,
)
from mprisk.prompts.template_bank import PromptTemplate


def test_prompt_cache_key_is_stable_and_sample_independent() -> None:
    key_a = prompt_cache_key(
        "qwen2-vl",
        "va_aux_v1_t01",
        sample_id="sample-a",
        prompt_set_key="va_aux_v1",
        protocol="va",
    )
    key_b = prompt_cache_key(
        "qwen2-vl",
        "va_aux_v1_t01",
        sample_id="sample-b",
        prompt_set_key="va_aux_v1",
        protocol="va",
    )

    assert key_a == key_b
    assert key_a == prompt_cache_key(
        "qwen2-vl",
        "va_aux_v1_t01",
        prompt_set_key="va_aux_v1",
        protocol="va",
    )


def test_prompt_cache_key_changes_by_prompt_contract_fields() -> None:
    base = prompt_cache_key(
        "qwen2-vl",
        "va_aux_v1_t01",
        prompt_set_key="va_aux_v1",
        protocol="va",
    )

    assert base != prompt_cache_key(
        "qwen2-vl",
        "va_aux_v1_t02",
        prompt_set_key="va_aux_v1",
        protocol="va",
    )
    assert base != prompt_cache_key(
        "qwen2-vl",
        "va_aux_v1_t01",
        prompt_set_key="vt_primary_v1",
        protocol="va",
    )
    assert base != prompt_cache_key(
        "qwen2-vl",
        "va_aux_v1_t01",
        prompt_set_key="va_aux_v1",
        protocol="vt",
    )


def test_build_prompt_cache_manifest_row() -> None:
    template = PromptTemplate(
        prompt_id="it_aux_v1_t01",
        template_text="Judge emotion from this input: {sample_text}",
        role="user",
        enabled=True,
    )

    row = build_prompt_cache_manifest_row(
        model_key="qwen2-vl",
        prompt_set_key="it_aux_v1",
        protocol="it",
        template=template,
    )

    assert row == {
        "model_key": "qwen2-vl",
        "prompt_set_key": "it_aux_v1",
        "prompt_id": "it_aux_v1_t01",
        "protocol": "it",
        "cache_key": prompt_cache_key(
            "qwen2-vl",
            "it_aux_v1_t01",
            prompt_set_key="it_aux_v1",
            protocol="it",
        ),
    }
