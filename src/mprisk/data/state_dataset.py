"""Build state-data manifests from final labels and cache entries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mprisk.cache.cache_manifest import DEFAULT_CONDITIONS, load_full_cache_manifest
from mprisk.cache.hidden_state_cache import HiddenStateEntry
from mprisk.cache.prefill_extract import t0_token_index
from mprisk.data.manifests import FinalManifestRow, read_final_manifest
from mprisk.data.protocol_views import VIEW_KEYS, normalize_protocol
from mprisk.utils.io import write_json, write_jsonl


@dataclass(frozen=True)
class StateDatasetBuildResult:
    manifest_path: Path
    summary_path: Path
    missing_path: Path
    resolved_count: int
    missing_count: int


def build_state_dataset(
    *,
    manifest_paths: Iterable[str | Path],
    cache_root: str | Path = ".",
    model_key: str,
    protocol: str,
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
) -> StateDatasetBuildResult:
    normalized_protocol = normalize_protocol(protocol)
    output_root = Path(output_dir or Path("outputs/state_data") / model_key / normalized_protocol)
    rows = _load_label_rows(manifest_paths, protocol=normalized_protocol)
    cache_manifest = load_full_cache_manifest(
        cache_root,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
    )
    resolutions = cache_manifest.resolve_m_conditions(
        [row.sample_id for row in rows],
        model_key=model_key,
        protocol=normalized_protocol,
    )

    state_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for row in rows:
        resolution = resolutions[row.sample_id]
        if not resolution.ok:
            missing_rows.append(_missing_row(row, model_key, resolution.missing_conditions))
            continue
        state_rows.append(_state_row(row, model_key, resolution.entries))

    manifest_output = write_jsonl(output_root / "state_dataset_manifest.jsonl", state_rows)
    summary_output = write_json(
        output_root / "state_dataset_summary.json",
        {
            "model_key": model_key,
            "protocol": normalized_protocol,
            "input_rows": len(rows),
            "resolved_rows": len(state_rows),
            "missing_cache_rows": len(missing_rows),
            "output_manifest": str(manifest_output),
            "missing_cache_rows_path": str(output_root / "missing_cache_rows.jsonl"),
        },
    )
    missing_output = write_jsonl(output_root / "missing_cache_rows.jsonl", missing_rows)
    return StateDatasetBuildResult(
        manifest_path=manifest_output,
        summary_path=summary_output,
        missing_path=missing_output,
        resolved_count=len(state_rows),
        missing_count=len(missing_rows),
    )


def _load_label_rows(
    manifest_paths: Iterable[str | Path],
    *,
    protocol: str,
) -> list[FinalManifestRow]:
    rows: list[FinalManifestRow] = []
    seen: set[str] = set()
    for path in manifest_paths:
        for row in read_final_manifest(path, protocol=protocol, use_in_main=True):
            if row.sample_id in seen:
                continue
            seen.add(row.sample_id)
            rows.append(row)
    return rows


def _state_row(
    row: FinalManifestRow,
    model_key: str,
    entries: dict[str, HiddenStateEntry],
) -> dict[str, Any]:
    m1_entry = entries["M1"]
    m2_entry = entries["M2"]
    m12_entry = entries["M12"]
    _require_consistent_entry_shape((m1_entry, m2_entry, m12_entry), row.sample_id)
    extras = row.model_dump()
    return {
        "sample_id": row.sample_id,
        "sample_type": row.sample_type,
        "source_dataset": row.source_dataset,
        "protocol": row.protocol,
        "model_key": model_key,
        "target_label": _target_label(row),
        "view_labels": _view_labels(row),
        "dominant_modality": extras.get("dominant_modality", "unclear"),
        "m1_entry": entry_to_manifest(m1_entry),
        "m2_entry": entry_to_manifest(m2_entry),
        "m12_entry": entry_to_manifest(m12_entry),
        "trajectory_meta": {
            "layer_count": m1_entry.layer_count,
            "hidden_dim": m1_entry.hidden_dim,
            "t0_token_index": t0_token_index(m1_entry),
        },
    }


def _missing_row(
    row: FinalManifestRow,
    model_key: str,
    missing_conditions: list[str],
) -> dict[str, Any]:
    return {
        "sample_id": row.sample_id,
        "sample_type": row.sample_type,
        "source_dataset": row.source_dataset,
        "protocol": row.protocol,
        "model_key": model_key,
        "missing_conditions": missing_conditions,
    }


def _target_label(row: FinalManifestRow) -> str | None:
    return row.views.M12.get("label") or row.views.M12.get("joint_label")


def _view_labels(row: FinalManifestRow) -> dict[str, dict[str, Any]]:
    return {
        view_key: _view_label(getattr(row.views, view_key))
        for view_key in VIEW_KEYS
    }


def _view_label(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": view.get("label") or view.get("joint_label"),
        "specific_affect": view.get("specific_affect"),
        "is_clear": view.get("is_clear", False),
    }


def _require_consistent_entry_shape(entries: tuple[HiddenStateEntry, ...], sample_id: str) -> None:
    missing = [condition for condition in DEFAULT_CONDITIONS if condition not in {e.condition for e in entries}]
    if missing:
        raise ValueError(f"Missing cache entries for {sample_id}: {', '.join(missing)}")
    shapes = {(entry.layer_count, entry.hidden_dim, t0_token_index(entry)) for entry in entries}
    if len(shapes) != 1:
        raise ValueError(
            f"Cache entry shape metadata differs for {sample_id}; expected shared trajectory_meta"
        )


def entry_to_manifest(entry: HiddenStateEntry) -> dict[str, Any]:
    return {
        "sample_id": entry.sample_id,
        "model_key": entry.model_key,
        "protocol": entry.protocol,
        "condition": entry.condition,
        "dataset_key": entry.dataset_key,
        "split": entry.split,
        "shard_path": entry.shard_path,
        "index_in_shard": entry.index_in_shard,
        "layer_count": entry.layer_count,
        "hidden_dim": entry.hidden_dim,
        "token_count": entry.token_count,
        "cache_root": str(entry.cache_root),
        "checksum": entry.checksum,
        "metadata": entry.metadata or {},
    }


def read_state_dataset_manifest(path: str | Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
