"""Prompt template bank contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    protocol: str
    text: str

    def render(self, **values: object) -> str:
        return self.text.format(**values)
