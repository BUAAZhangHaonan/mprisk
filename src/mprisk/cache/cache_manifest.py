"""Full-cache manifest schema, loading, and resolution helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mprisk.cache.hidden_state_cache import (
    HiddenStateEntry,
    normalize_condition,
    normalize_protocol,
)

DEFAULT_CONDITIONS = ("M1", "M2", "M12")
DEFAULT_MANIFEST_PATH = Path("outputs/full_cache/manifests/unified_full_cache_manifest.json")
DEFAULT_LEDGER_PATH = Path("outputs/full_cache/manifests/extraction_ledger.csv")
DEFAULT_REPORTS_DIR = Path("outputs/state_data/reports")

REQUIRED_CACHE_FIELDS = (
    "model_key",
    "protocol",
    "dataset_key",
    "split",
    "condition",
)

ENTRY_FIELDS = {
    "sample_id",
    "model_key",
    "protocol",
    "condition",
    "dataset_key",
    "split",
    "shard_path",
    "artifact_uri",
    "index_in_shard",
    "layer_count",
    "hidden_dim",
    "token_count",
    "cache_root",
    "checksum",
    "metadata",
}


class CacheResolutionError(ValueError):
    """Raised when one or more samples cannot resolve all M conditions."""

    def __init__(self, message: str, resolutions: dict[str, "CacheResolution"]):
        super().__init__(message)
        self.resolutions = resolutions


@dataclass(frozen=True)
class CacheResolution:
    sample_id: str
    entries: dict[str, HiddenStateEntry]
    missing_conditions: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_conditions


class FullCacheManifest:
    """Manifest-backed lookup layer for full-cache hidden-state shards."""

    def __init__(
        self,
        entries: Iterable[HiddenStateEntry],
        *,
        cache_root: str | Path,
        manifest_path: str | Path,
        ledger_path: str | Path,
        ledger_rows: list[dict[str, str]] | None = None,
        raw_manifest: dict[str, Any] | None = None,
    ) -> None:
        self.entries = list(entries)
        self.cache_root = Path(cache_root)
        self.manifest_path = Path(manifest_path)
        self.ledger_path = Path(ledger_path)
        self.ledger_rows = ledger_rows or []
        self.raw_manifest = raw_manifest or {}
        self._entries_by_key: dict[tuple[str, str, str, str], list[HiddenStateEntry]] = {}
        for entry in self.entries:
            self._entries_by_key.setdefault(entry.key, []).append(entry)

    def query(
        self,
        sample_id: str,
        model_key: str,
        protocol: str,
        condition: str,
    ) -> HiddenStateEntry | None:
        key = (sample_id, model_key, normalize_protocol(protocol), normalize_condition(condition))
        entries = self._entries_by_key.get(key)
        if not entries:
            return None
        return entries[0]

    def resolve_m_conditions(
        self,
        sample_ids: Iterable[str],
        model_key: str,
        protocol: str,
        conditions: Iterable[str] = DEFAULT_CONDITIONS,
    ) -> dict[str, CacheResolution]:
        normalized_conditions = [normalize_condition(condition) for condition in conditions]
        resolutions: dict[str, CacheResolution] = {}
        for sample_id in sample_ids:
            entries = {
                condition: entry
                for condition in normalized_conditions
                if (
                    entry := self.query(
                        sample_id=sample_id,
                        model_key=model_key,
                        protocol=protocol,
                        condition=condition,
                    )
                )
                is not None
            }
            missing = [condition for condition in normalized_conditions if condition not in entries]
            resolutions[sample_id] = CacheResolution(
                sample_id=sample_id,
                entries=entries,
                missing_conditions=missing,
            )
        return resolutions

    def require_m_conditions(
        self,
        sample_ids: Iterable[str],
        model_key: str,
        protocol: str,
        conditions: Iterable[str] = DEFAULT_CONDITIONS,
    ) -> dict[str, CacheResolution]:
        resolutions = self.resolve_m_conditions(
            sample_ids,
            model_key=model_key,
            protocol=protocol,
            conditions=conditions,
        )
        missing = [
            f"{sample_id}: {','.join(resolution.missing_conditions)}"
            for sample_id, resolution in resolutions.items()
            if resolution.missing_conditions
        ]
        if missing:
            raise CacheResolutionError(
                "Missing full-cache conditions for " + "; ".join(missing),
                resolutions,
            )
        return resolutions


def validate_cache_manifest_entry(entry: dict[str, object]) -> bool:
    has_shard_location = "artifact_uri" in entry or "shard_path" in entry
    return has_shard_location and all(field in entry for field in REQUIRED_CACHE_FIELDS)


def load_full_cache_manifest(
    cache_root: str | Path = ".",
    *,
    manifest_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
) -> FullCacheManifest:
    root = Path(cache_root)
    manifest_file = _resolve_path(root, manifest_path or DEFAULT_MANIFEST_PATH)
    ledger_file = _resolve_path(root, ledger_path or DEFAULT_LEDGER_PATH)

    raw_manifest = _read_manifest(manifest_file)
    manifest_entries = list(raw_manifest.get("entries") or [])
    ledger_rows = _read_ledger(ledger_file)
    source_entries = manifest_entries if manifest_entries else ledger_rows
    entries = [
        _entry_from_row(row, cache_root=root)
        for row in source_entries
        if _can_materialize_entry(row)
    ]

    return FullCacheManifest(
        entries,
        cache_root=root,
        manifest_path=manifest_file,
        ledger_path=ledger_file,
        ledger_rows=ledger_rows,
        raw_manifest=raw_manifest,
    )


def write_cache_resolution_summary(
    resolutions: dict[str, CacheResolution],
    *,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> dict[str, Path]:
    report_dir = Path(reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "cache_resolution_summary.json"
    markdown_path = report_dir / "cache_resolution_summary.md"

    samples = [_resolution_to_dict(resolution) for resolution in resolutions.values()]
    summary = {
        "total_samples": len(samples),
        "resolved_samples": sum(1 for sample in samples if sample["ok"]),
        "missing_samples": sum(1 for sample in samples if not sample["ok"]),
        "samples": samples,
    }

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_resolution_summary_markdown(summary), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _resolve_path(root: Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"entries": []}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Full-cache manifest must be a JSON object: {path}")
    entries = data.get("entries")
    if entries is not None and not isinstance(entries, list):
        raise ValueError(f"Full-cache manifest entries must be a list: {path}")
    return data


def _read_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _can_materialize_entry(row: dict[str, Any]) -> bool:
    return all(
        _present(row.get(field))
        for field in (
            "sample_id",
            "model_key",
            "protocol",
            "condition",
            "dataset_key",
            "split",
            "index_in_shard",
            "layer_count",
            "hidden_dim",
            "token_count",
        )
    ) and _present(row.get("shard_path") or row.get("artifact_uri"))


def _entry_from_row(row: dict[str, Any], *, cache_root: Path) -> HiddenStateEntry:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            key: value
            for key, value in row.items()
            if key not in ENTRY_FIELDS and _present(value)
        }
    )
    shard_path = row.get("shard_path") or row.get("artifact_uri")
    return HiddenStateEntry(
        sample_id=str(row["sample_id"]),
        model_key=str(row["model_key"]),
        protocol=str(row["protocol"]),
        condition=str(row["condition"]),
        dataset_key=str(row["dataset_key"]),
        split=str(row["split"]),
        shard_path=str(shard_path),
        index_in_shard=int(row["index_in_shard"]),
        layer_count=int(row["layer_count"]),
        hidden_dim=int(row["hidden_dim"]),
        token_count=int(row["token_count"]),
        cache_root=row.get("cache_root") or cache_root,
        checksum=_optional_string(row.get("checksum")),
        metadata=metadata,
    )


def _optional_string(value: Any) -> str | None:
    if not _present(value):
        return None
    return str(value)


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _resolution_to_dict(resolution: CacheResolution) -> dict[str, Any]:
    return {
        "sample_id": resolution.sample_id,
        "ok": resolution.ok,
        "present_conditions": [
            condition for condition in DEFAULT_CONDITIONS if condition in resolution.entries
        ],
        "missing_conditions": resolution.missing_conditions,
        "entries": [
            _entry_to_dict(entry)
            for condition in DEFAULT_CONDITIONS
            if (entry := resolution.entries.get(condition)) is not None
        ],
    }


def _entry_to_dict(entry: HiddenStateEntry) -> dict[str, Any]:
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
        "checksum": entry.checksum,
        "metadata": entry.metadata,
    }


def _resolution_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cache Resolution Summary",
        "",
        f"- Total samples: {summary['total_samples']}",
        f"- Resolved samples: {summary['resolved_samples']}",
        f"- Missing samples: {summary['missing_samples']}",
        "",
        "| sample_id | ok | present_conditions | missing_conditions |",
        "| --- | --- | --- | --- |",
    ]
    for sample in summary["samples"]:
        lines.append(
            "| {sample_id} | {ok} | {present} | {missing} |".format(
                sample_id=sample["sample_id"],
                ok=str(sample["ok"]).lower(),
                present=",".join(sample["present_conditions"]) or "-",
                missing=",".join(sample["missing_conditions"]) or "-",
            )
        )
    return "\n".join(lines) + "\n"
