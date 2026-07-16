"""Model-wrapper contracts for generation and prefill-state extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

SUPPORTED_PREFILL_PROTOCOLS = frozenset({"vt", "va", "vta"})
SUPPORTED_PREFILL_CONDITIONS = frozenset({"M1", "M2", "M12"})


@dataclass(frozen=True)
class ModelOutput:
    answer_text: str
    parsed_answer: str | None = None


@dataclass(frozen=True)
class GenerationRequest:
    """One explicit conditioned text-generation request."""

    sample_id: str
    model_key: str
    protocol: str
    condition: str
    messages: Sequence[Mapping[str, Any]]
    media_paths: Mapping[str, str]
    use_audio_in_video: bool
    generation_kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.model_key:
            raise ValueError("Generation request identifiers must be non-empty")
        if self.protocol.lower() not in SUPPORTED_PREFILL_PROTOCOLS:
            raise ValueError(f"Unsupported generation protocol: {self.protocol!r}")
        if self.condition.upper() not in SUPPORTED_PREFILL_CONDITIONS:
            raise ValueError(f"Unsupported generation condition: {self.condition!r}")
        if not self.messages:
            raise ValueError("Generation request messages must not be empty")
        required = {"do_sample", "num_beams", "max_new_tokens"}
        if set(self.generation_kwargs) != required:
            raise ValueError(
                "Generation kwargs must contain only do_sample, num_beams, max_new_tokens"
            )
        if (
            self.generation_kwargs["do_sample"] is not False
            or self.generation_kwargs["num_beams"] != 1
        ):
            raise ValueError("Generation must be greedy with do_sample=False and num_beams=1")
        if (
            not isinstance(self.generation_kwargs["max_new_tokens"], int)
            or self.generation_kwargs["max_new_tokens"] <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        object.__setattr__(self, "protocol", self.protocol.lower())
        object.__setattr__(self, "condition", self.condition.upper())
        object.__setattr__(self, "messages", tuple(dict(message) for message in self.messages))
        object.__setattr__(self, "media_paths", dict(self.media_paths))
        object.__setattr__(self, "generation_kwargs", dict(self.generation_kwargs))


@dataclass(frozen=True)
class GenerationResult:
    """Raw newly generated text and tokens for one conditioned request."""

    request: GenerationRequest
    text: str
    token_ids: Sequence[int]
    eos_token_ids: Sequence[int]
    finish_reason: Literal["eos", "max_new_tokens"]
    input_token_count: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Generated text must be a string")
        if self.input_token_count <= 0:
            raise ValueError("input_token_count must be positive")
        token_ids = tuple(int(token_id) for token_id in self.token_ids)
        eos_token_ids = tuple(int(token_id) for token_id in self.eos_token_ids)
        if not token_ids:
            raise ValueError("Generated token_ids must not be empty")
        if self.finish_reason not in {"eos", "max_new_tokens"}:
            raise ValueError("finish_reason must be eos or max_new_tokens")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "eos_token_ids", eos_token_ids)
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class PrefillRequest:
    """One explicit multimodal conditioning request for a single sample."""

    sample_id: str
    model_key: str
    protocol: str
    condition: str
    dataset_key: str
    split: str
    messages: Sequence[Mapping[str, Any]]
    media_paths: Mapping[str, str]
    use_audio_in_video: bool
    prompt_set_key: str = "adhoc"
    prompt_id: str = "adhoc"

    def __post_init__(self) -> None:
        protocol = self.protocol.lower()
        condition = self.condition.upper()
        if protocol not in SUPPORTED_PREFILL_PROTOCOLS:
            raise ValueError(f"Unsupported prefill protocol: {self.protocol!r}")
        if condition not in SUPPORTED_PREFILL_CONDITIONS:
            raise ValueError(f"Unsupported prefill condition: {self.condition!r}")
        if (
            not self.sample_id
            or not self.model_key
            or not self.dataset_key
            or not self.split
            or not self.prompt_set_key
            or not self.prompt_id
        ):
            raise ValueError("Prefill request identifiers must be non-empty")
        if not isinstance(self.use_audio_in_video, bool):
            raise TypeError("use_audio_in_video must be an explicit bool")
        if not self.messages:
            raise ValueError("Prefill request messages must not be empty")
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "messages", tuple(dict(message) for message in self.messages))
        object.__setattr__(self, "media_paths", dict(self.media_paths))


@dataclass(frozen=True)
class PrefillResult:
    """All transformer-block states at the token that predicts the first reply token."""

    request: PrefillRequest
    trajectory: np.ndarray
    token_count: int
    t0_token_index: int
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        trajectory = np.asarray(self.trajectory, dtype=np.float32)
        if trajectory.ndim != 2:
            raise ValueError("Prefill trajectory must have shape [layer_count, hidden_dim]")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not 0 <= self.t0_token_index < self.token_count:
            raise ValueError("t0_token_index must address the conditioning sequence")
        if not np.isfinite(trajectory).all():
            raise ValueError("Prefill trajectory must contain only finite values")
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def layer_count(self) -> int:
        return int(self.trajectory.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.trajectory.shape[1])


class BaseModelWrapper:
    model_key: str
    family: str

    def generate(self, prompt: str) -> ModelOutput:
        raise NotImplementedError

    def generate_conditioned(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    def extract_prefill(self, request: PrefillRequest) -> PrefillResult:
        raise NotImplementedError
