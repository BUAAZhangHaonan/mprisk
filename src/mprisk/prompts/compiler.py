"""Compile prompts for model wrappers."""

from __future__ import annotations

from mprisk.prompts.template_bank import PromptTemplate


def compile_prompt(template: PromptTemplate, sample: dict[str, object]) -> str:
    return template.render(**sample)
