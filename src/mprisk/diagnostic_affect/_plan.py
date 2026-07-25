"""Diagnostic Affect Description plan primitives and constants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mprisk.models.base_wrapper import GenerationRequest, GenerationResult

CANONICAL_DIAGNOSTIC_AFFECT_PROMPT = (
    "Based on the complete input, describe the person's overall emotional state in one concise "
    "sentence. Do not address the person, give advice, or explain your reasoning."
)
CONFIG_SCHEMA = "mprisk_diagnostic_affect_description_config_v2"
OUTPUT_SCHEMA = "mprisk_diagnostic_affect_description"
PROVENANCE_SCHEMA = "mprisk_diagnostic_affect_description_provenance_v2"
SIGNATURE_SCHEMA = "mprisk_diagnostic_affect_description_signature_v2"
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_SUPPORTED_PROTOCOLS = frozenset({"VT", "VA"})
_SUPPORTED_CONDITIONS = frozenset({"M12"})


@dataclass(frozen=True)
class DiagnosticAffectDescriptionTask:
    task_id: str
    request: GenerationRequest
    input_sha256: str
    media_sha256: str
    prompt_sha256: str


@dataclass(frozen=True)
class DiagnosticAffectDescriptionPlan:
    tasks: list[DiagnosticAffectDescriptionTask]
    signature: dict[str, Any]
    counts: dict[str, int]


def _request_payload(request: GenerationRequest) -> dict[str, Any]:
    return {
        "sample_id": request.sample_id,
        "model_key": request.model_key,
        "protocol": request.protocol,
        "condition": request.condition,
        "messages": list(request.messages),
        "media_paths": dict(request.media_paths),
        "use_audio_in_video": request.use_audio_in_video,
        "generation_kwargs": dict(request.generation_kwargs),
    }


def _result_payload(result: GenerationResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "token_ids": list(result.token_ids),
        "eos_token_ids": list(result.eos_token_ids),
        "finish_reason": result.finish_reason,
        "input_token_count": result.input_token_count,
    }
