"""Prompt equivalence set helpers."""

from __future__ import annotations

from mprisk.prompts.template_bank import PromptTemplate


def select_first_k(templates: list[PromptTemplate], k: int) -> list[PromptTemplate]:
    return templates[:k]
