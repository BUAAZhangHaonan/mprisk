"""Hidden-state cache entry types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONDITION_ALIASES = {
    "m1": "M1",
    "M1": "M1",
    "m2": "M2",
    "M2": "M2",
    "m12": "M12",
    "M12": "M12",
}


def normalize_condition(condition: str) -> str:
    normalized = CONDITION_ALIASES.get(condition)
    if normalized is None:
        normalized = CONDITION_ALIASES.get(condition.upper())
    if normalized is None:
        raise ValueError(f"Unsupported cache condition: {condition!r}")
    return normalized


def normalize_protocol(protocol: str) -> str:
    return protocol.lower()


@dataclass(frozen=True)
class HiddenStateEntry:
    sample_id: str
    model_key: str
    protocol: str
    condition: str
    dataset_key: str
    split: str
    shard_path: str
    index_in_shard: int
    layer_count: int
    hidden_dim: int
    token_count: int
    cache_root: Path | str
    checksum: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol", normalize_protocol(self.protocol))
        object.__setattr__(self, "condition", normalize_condition(self.condition))
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        object.__setattr__(self, "index_in_shard", int(self.index_in_shard))
        object.__setattr__(self, "layer_count", int(self.layer_count))
        object.__setattr__(self, "hidden_dim", int(self.hidden_dim))
        object.__setattr__(self, "token_count", int(self.token_count))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.sample_id, self.model_key, self.protocol, self.condition)

    @property
    def shard_file(self) -> Path:
        path = Path(self.shard_path)
        if path.is_absolute():
            return path
        return Path(self.cache_root) / path
