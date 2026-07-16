"""Compile prompts for model wrappers."""

from __future__ import annotations

from mprisk.prompts.template_bank import PromptTemplate


def compile_prompt(template: PromptTemplate, sample: dict[str, object]) -> str:
    try:
        return template.render(**sample)
    except KeyError as exc:
        missing_field = exc.args[0]
        raise ValueError(
            f"Prompt {template.prompt_id} requires missing sample field {missing_field!r}"
        ) from exc
