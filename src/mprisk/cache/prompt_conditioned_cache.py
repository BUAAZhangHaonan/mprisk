"""Prompt-conditioned hidden-state cache manifest contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mprisk.cache.hidden_state_cache import (
    HiddenStateEntry,
    normalize_condition,
    normalize_protocol,
)
from mprisk.utils.io import write_jsonl


REQUIRED_PROMPT_CONDITIONED_FIELDS = (
    "sample_id",
    "model_key",
    "protocol",
    "condition",
    "prompt_set_key",
    "prompt_id",
    "shard_path",
    "index_in_shard",
    "layer_count",
    "hidden_dim",
    "token_count",
    "t0_token_index",
    "cache_root",
)

ENTRY_FIELDS = set(REQUIRED_PROMPT_CONDITIONED_FIELDS) | {
    "artifact_uri",
    "checksum",
    "metadata",
}


@dataclass(frozen=True)
class PromptConditionedStateEntry:
    sample_id: str
    model_key: str
    protocol: str
    condition: str
    prompt_set_key: str
    prompt_id: str
    shard_path: str
    index_in_shard: int
    layer_count: int
    hidden_dim: int
    token_count: int
    t0_token_index: int
    cache_root: Path | str
    checksum: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "model_key", str(self.model_key))
        object.__setattr__(self, "protocol", normalize_protocol(str(self.protocol)))
        object.__setattr__(self, "condition", normalize_condition(str(self.condition)))
        object.__setattr__(self, "prompt_set_key", str(self.prompt_set_key))
        object.__setattr__(self, "prompt_id", str(self.prompt_id))
        object.__setattr__(self, "shard_path", str(self.shard_path))
        object.__setattr__(self, "index_in_shard", int(self.index_in_shard))
        object.__setattr__(self, "layer_count", int(self.layer_count))
        object.__setattr__(self, "hidden_dim", int(self.hidden_dim))
        object.__setattr__(self, "token_count", int(self.token_count))
        object.__setattr__(self, "t0_token_index", int(self.t0_token_index))
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if self.checksum is not None:
            object.__setattr__(self, "checksum", str(self.checksum))

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.sample_id,
            self.model_key,
            self.protocol,
            self.condition,
            self.prompt_set_key,
            self.prompt_id,
        )

    @property
    def shard_file(self) -> Path:
        path = Path(self.shard_path)
        if path.is_absolute():
            return path
        return self.cache_root / path

    def to_hidden_state_entry(self) -> HiddenStateEntry:
        metadata = dict(self.metadata or {})
        metadata["t0_token_index"] = self.t0_token_index
        return HiddenStateEntry(
            sample_id=self.sample_id,
            model_key=self.model_key,
            protocol=self.protocol,
            condition=self.condition,
            dataset_key=str(metadata.get("dataset_key") or "prompt_conditioned"),
            split=str(metadata.get("split") or "unknown"),
            shard_path=self.shard_path,
            index_in_shard=self.index_in_shard,
            layer_count=self.layer_count,
            hidden_dim=self.hidden_dim,
            token_count=self.token_count,
            cache_root=self.cache_root,
            checksum=self.checksum,
            metadata=metadata,
        )

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model_key": self.model_key,
            "protocol": self.protocol,
            "condition": self.condition,
            "prompt_set_key": self.prompt_set_key,
            "prompt_id": self.prompt_id,
            "shard_path": self.shard_path,
            "index_in_shard": self.index_in_shard,
            "layer_count": self.layer_count,
            "hidden_dim": self.hidden_dim,
            "token_count": self.token_count,
            "t0_token_index": self.t0_token_index,
            "cache_root": str(self.cache_root),
            "checksum": self.checksum,
            "metadata": dict(self.metadata or {}),
        }


class PromptConditionedManifest:
    """JSONL-backed lookup for prompted hidden states."""

    def __init__(self, entries: Iterable[PromptConditionedStateEntry | dict[str, Any]]) -> None:
        self.entries = [coerce_prompt_conditioned_entry(entry) for entry in entries]
        self._entries_by_key: dict[
            tuple[str, str, str, str, str, str], PromptConditionedStateEntry
        ] = {}
        for entry in self.entries:
            self._entries_by_key.setdefault(entry.key, entry)

    def lookup(
        self,
        *,
        sample_id: str,
        model_key: str,
        protocol: str,
        condition: str,
        prompt_set_key: str,
        prompt_id: str,
    ) -> PromptConditionedStateEntry | None:
        return self._entries_by_key.get(
            prompt_conditioned_key(
                sample_id=sample_id,
                model_key=model_key,
                protocol=protocol,
                condition=condition,
                prompt_set_key=prompt_set_key,
                prompt_id=prompt_id,
            )
        )


def prompt_conditioned_key(
    *,
    sample_id: str,
    model_key: str,
    protocol: str,
    condition: str,
    prompt_set_key: str,
    prompt_id: str,
) -> tuple[str, str, str, str, str, str]:
    return (
        str(sample_id),
        str(model_key),
        normalize_protocol(str(protocol)),
        normalize_condition(str(condition)),
        str(prompt_set_key),
        str(prompt_id),
    )


def coerce_prompt_conditioned_entry(
    entry: PromptConditionedStateEntry | dict[str, Any],
    *,
    default_cache_root: str | Path | None = None,
) -> PromptConditionedStateEntry:
    if isinstance(entry, PromptConditionedStateEntry):
        return entry
    return prompt_conditioned_entry_from_row(entry, default_cache_root=default_cache_root)


def prompt_conditioned_entry_from_row(
    row: dict[str, Any],
    *,
    default_cache_root: str | Path | None = None,
) -> PromptConditionedStateEntry:
    if not isinstance(row, dict):
        raise ValueError("prompt-conditioned cache row must be a JSON object")
    clean = dict(row)
    if "shard_path" not in clean and _present(clean.get("artifact_uri")):
        clean["shard_path"] = clean["artifact_uri"]
    if "cache_root" not in clean and default_cache_root is not None:
        clean["cache_root"] = default_cache_root
    for field in REQUIRED_PROMPT_CONDITIONED_FIELDS:
        if not _present(clean.get(field)):
            raise ValueError(f"prompt-conditioned cache row missing required field {field}")

    metadata = _metadata_from_row(clean)
    return PromptConditionedStateEntry(
        sample_id=clean["sample_id"],
        model_key=clean["model_key"],
        protocol=clean["protocol"],
        condition=clean["condition"],
        prompt_set_key=clean["prompt_set_key"],
        prompt_id=clean["prompt_id"],
        shard_path=clean["shard_path"],
        index_in_shard=clean["index_in_shard"],
        layer_count=clean["layer_count"],
        hidden_dim=clean["hidden_dim"],
        token_count=clean["token_count"],
        t0_token_index=clean["t0_token_index"],
        cache_root=clean["cache_root"],
        checksum=_optional_string(clean.get("checksum")),
        metadata=metadata,
    )


def read_prompt_conditioned_entries(path: str | Path) -> list[PromptConditionedStateEntry]:
    manifest_path = Path(path)
    entries: list[PromptConditionedStateEntry] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                entries.append(prompt_conditioned_entry_from_row(row))
            except ValueError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: {exc}") from exc
    return entries


def load_prompt_conditioned_manifest(path: str | Path) -> PromptConditionedManifest:
    return PromptConditionedManifest(read_prompt_conditioned_entries(path))


def write_prompt_conditioned_manifest(
    path: str | Path,
    entries: Iterable[PromptConditionedStateEntry | dict[str, Any]],
) -> Path:
    rows = [coerce_prompt_conditioned_entry(entry).to_manifest_row() for entry in entries]
    return write_jsonl(path, rows)


def _metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            key: value
            for key, value in row.items()
            if key not in ENTRY_FIELDS and _present(value)
        }
    )
    return metadata


def _optional_string(value: Any) -> str | None:
    if not _present(value):
        return None
    return str(value)


def _present(value: Any) -> bool:
    return value is not None and value != ""
