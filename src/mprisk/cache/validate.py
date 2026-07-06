"""Hidden-state validation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from mprisk.cache.cache_manifest import DEFAULT_CONDITIONS, FullCacheManifest
from mprisk.cache.hidden_state_cache import HiddenStateEntry


def finite_vector(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


@dataclass(frozen=True)
class CacheValidationError:
    code: str
    message: str
    key: tuple[str, ...]


@dataclass(frozen=True)
class CacheValidationReport:
    errors: list[CacheValidationError]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_full_cache_manifest(manifest: FullCacheManifest) -> CacheValidationReport:
    errors: list[CacheValidationError] = []
    errors.extend(_duplicate_key_errors(manifest.entries))
    errors.extend(_group_completeness_and_shape_errors(manifest.entries))
    errors.extend(_missing_shard_errors(manifest.entries))
    return CacheValidationReport(errors=errors)


def _duplicate_key_errors(entries: Iterable[HiddenStateEntry]) -> list[CacheValidationError]:
    by_key: dict[tuple[str, str, str, str], int] = {}
    for entry in entries:
        by_key[entry.key] = by_key.get(entry.key, 0) + 1
    return [
        CacheValidationError(
            code="duplicate_key",
            message=f"Duplicate full-cache entry for {key}",
            key=key,
        )
        for key, count in by_key.items()
        if count > 1
    ]


def _group_completeness_and_shape_errors(
    entries: Iterable[HiddenStateEntry],
) -> list[CacheValidationError]:
    grouped: dict[tuple[str, str, str], list[HiddenStateEntry]] = {}
    for entry in entries:
        group_key = (entry.sample_id, entry.model_key, entry.protocol)
        grouped.setdefault(group_key, []).append(entry)

    errors: list[CacheValidationError] = []
    for group_key, group_entries in grouped.items():
        present_conditions = {entry.condition for entry in group_entries}
        for condition in DEFAULT_CONDITIONS:
            if condition not in present_conditions:
                errors.append(
                    CacheValidationError(
                        code="missing_condition",
                        message=f"Missing condition {condition} for {group_key}",
                        key=(*group_key, condition),
                    )
                )

        layer_counts = {entry.layer_count for entry in group_entries}
        if len(layer_counts) > 1:
            errors.append(
                CacheValidationError(
                    code="inconsistent_layer_count",
                    message=(
                        f"Inconsistent layer_count values for {group_key}: "
                        f"{sorted(layer_counts)}"
                    ),
                    key=group_key,
                )
            )

        hidden_dims = {entry.hidden_dim for entry in group_entries}
        if len(hidden_dims) > 1:
            errors.append(
                CacheValidationError(
                    code="inconsistent_hidden_dim",
                    message=(
                        f"Inconsistent hidden_dim values for {group_key}: "
                        f"{sorted(hidden_dims)}"
                    ),
                    key=group_key,
                )
            )
    return errors


def _missing_shard_errors(entries: Iterable[HiddenStateEntry]) -> list[CacheValidationError]:
    return [
        CacheValidationError(
            code="missing_shard_file",
            message=f"Shard file does not exist: {entry.shard_file}",
            key=entry.key,
        )
        for entry in entries
        if not entry.shard_file.exists()
    ]
