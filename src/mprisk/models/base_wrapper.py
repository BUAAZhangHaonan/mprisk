"""Base model wrapper contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOutput:
    answer_text: str
    parsed_answer: str | None = None


class BaseModelWrapper:
    model_key: str
    family: str

    def generate(self, prompt: str) -> ModelOutput:
        raise NotImplementedError

    def extract_prefill(self, prompt: str) -> dict[str, object]:
        raise NotImplementedError
