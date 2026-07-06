"""Build stable prompt cache keys."""

from __future__ import annotations

from mprisk.prompts.template_bank import PromptTemplate
from mprisk.utils.hashing import stable_hash


def prompt_cache_key(
    model_key: str,
    prompt_id: str,
    sample_id: str | None = None,
    *,
    prompt_set_key: str = "",
    protocol: str = "",
) -> str:
    return stable_hash(
        {
            "model_key": model_key,
            "prompt_set_key": prompt_set_key,
            "prompt_id": prompt_id,
            "protocol": protocol,
        }
    )


def build_prompt_cache_manifest_row(
    *,
    model_key: str,
    prompt_set_key: str,
    protocol: str,
    template: PromptTemplate,
) -> dict[str, str]:
    return {
        "model_key": model_key,
        "prompt_set_key": prompt_set_key,
        "prompt_id": template.prompt_id,
        "protocol": protocol,
        "cache_key": prompt_cache_key(
            model_key,
            template.prompt_id,
            prompt_set_key=prompt_set_key,
            protocol=protocol,
        ),
    }
