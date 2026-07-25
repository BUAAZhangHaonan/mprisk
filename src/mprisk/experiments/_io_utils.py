"""Low-level IO helpers for downstream experiments (hashing, yaml, csv, paths).

This module sits at the bottom of the experiments subpackage DAG. It only
imports from :mod:`mprisk.utils.io` (a true leaf with no ``mprisk.*`` imports),
so there is no circular-import risk. ``CacheJob``/``DownstreamPlan`` are
imported under :data:`typing.TYPE_CHECKING` only, for type annotations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from mprisk.utils.io import sha256_file

if TYPE_CHECKING:
    from mprisk.experiments.jobs import CacheJob, DownstreamPlan


def _training_config_path(plan: "DownstreamPlan", job: "CacheJob", repr_key: str) -> Path:
    path = plan.config_root / f"seed{job.seed}" / f"{job.model_key}_{repr_key}.yaml"
    if not path.is_file():
        raise ValueError(f"missing immutable training config: {path}")
    return path

    path = plan.config_root / f"seed{job.seed}" / f"{job.model_key}_{repr_key}.yaml"
    if not path.is_file():
        raise ValueError(f"missing immutable training config: {path}")
    return path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def _sha256(path: Path | str) -> str:
    """Delegate to the canonical :func:`mprisk.utils.io.sha256_file`.

    Kept as a thin wrapper so existing ``experiments`` callers keep working;
    new code should call ``sha256_file`` directly.
    """
    return sha256_file(path)


def _json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def _one(rows: list[dict[str, Any]], field: str) -> str:
    values = {str(row.get(field, "")) for row in rows}
    if len(values) != 1 or not next(iter(values)):
        raise ValueError(f"official paper inputs require homogeneous {field}")
    return next(iter(values))
