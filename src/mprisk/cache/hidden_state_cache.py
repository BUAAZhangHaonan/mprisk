"""Hidden-state cache entry types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HiddenStateEntry:
    sample_id: str
    model_key: str
    protocol: str
    condition: str
    layer_count: int
    hidden_dim: int
