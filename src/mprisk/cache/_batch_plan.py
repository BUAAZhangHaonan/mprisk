"""Batch plan dataclasses and shared constants for prefill extraction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WrapperFactory = Callable[..., Any]
CONDITIONS = ("M1", "M2", "M12")
DEFAULT_ASSET_CONFIG = Path("configs/assets/model_assets.yaml")


@dataclass(frozen=True)
class BatchTask:
    task_id: str
    sample_id: str
    prompt_set_key: str
    prompt_id: str
    prompt_text: str | None
    condition: str
    row: dict[str, Any]


@dataclass(frozen=True)
class BatchPlan:
    tasks: list[BatchTask]
    prompt_ids: tuple[str, ...]
    unresolved_prompt_variables: tuple[str, ...]
    rows: list[dict[str, Any]]
    signature: dict[str, Any]


@dataclass(frozen=True)
class RecoveredArtifact:
    entry: dict[str, Any]
    provenance: dict[str, Any]
