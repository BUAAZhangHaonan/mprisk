"""Versioned prompt-context v2 inputs and deterministic eight-row pilot."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mprisk.data.generated_archive_freeze import _canonical_json, _sha256

CONTEXT_PRIORITY = ("setting", "trigger", "ltx2_prompt")
PILOT_ROW_SCHEMA = "mprisk_gt_prompt_context_v2_pilot_row"
PILOT_SAMPLE_IDS = (
    "gen:accept_a_svt:S0097",
    "gen:accept_a_svt:S0098",
    "gen:accept_a_va:S0047",
    "gen:accept_a_va:S0048",
    "gen:accept_c_svt:S0001",
    "gen:accept_c_svt:S0002",
    "gen:accept_c_va:S0012",
    "gen:accept_c_va:S0013",
)
_TEMPLATE_TRIGGER = re.compile(r"T[1-4]")


def resolve_context(source_row: dict[str, Any]) -> tuple[str, str]:
    """Resolve context with the explicit v2 priority and no implicit fallback."""
    setting = _optional_text(source_row.get("setting"))
    if setting:
        return setting, "setting"
    trigger = _optional_text(source_row.get("trigger"))
    if trigger and _TEMPLATE_TRIGGER.fullmatch(trigger) is None:
        return trigger, "trigger"
    raw_prompt = _optional_text(source_row.get("ltx2_prompt"))
    if raw_prompt:
        return raw_prompt, "ltx2_prompt"
    raise ValueError("Prompt-context v2 requires setting, natural trigger, or ltx2_prompt")


def build_prompt_context_v2_pilot(
    repo_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the deterministic A/C by VT/VA two-per-cell pilot in memory."""
    root = Path(repo_root).resolve()
    input_root = root / "data/frozen/generated_round1_v1"
    archive_path = input_root / "archive_manifest.jsonl"
    assignment_path = input_root / "archetype_semantic_assignments_v1.jsonl"
    dictionary_path = input_root / "archetype_canonical_meanings_v1.jsonl"
    archive_rows = _read_jsonl(archive_path)
    assignments = _index(_read_jsonl(assignment_path), "sample_id")
    meanings = _index(_read_jsonl(dictionary_path), "archetype_semantic_id")
    if len(archive_rows) != 652 or len(assignments) != 652:
        raise ValueError("Prompt-context v2 source must contain exactly 652 archive assignments")

    context_source_counts: Counter[str] = Counter()
    recovered: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for archive_row in archive_rows:
        source_row = archive_row.get("source_row")
        if not isinstance(source_row, dict):
            raise TypeError("Archive row source_row must be an object")
        _, context_source = resolve_context(source_row)
        context_source_counts[context_source] += 1
        if context_source == "ltx2_prompt":
            key = (_text(archive_row, "data_type"), _text(archive_row, "protocol"))
            recovered.setdefault(key, []).append(archive_row)
    if dict(context_source_counts) != {"setting": 126, "trigger": 36, "ltx2_prompt": 490}:
        raise ValueError(
            f"Unexpected prompt-context v2 source counts: {dict(context_source_counts)}"
        )

    selected: list[dict[str, Any]] = []
    for key in (("A", "VT"), ("A", "VA"), ("C", "VT"), ("C", "VA")):
        candidates = sorted(recovered.get(key, []), key=lambda row: _text(row, "sample_id"))
        if len(candidates) < 2:
            raise ValueError(f"Prompt-context v2 pilot cell {key!r} has fewer than two rows")
        selected.extend(candidates[:2])
    if tuple(_text(row, "sample_id") for row in selected) != PILOT_SAMPLE_IDS:
        raise ValueError("Prompt-context v2 deterministic pilot IDs changed")

    pilot_rows = [
        _pilot_row(
            archive_row=row,
            assignment=assignments[_text(row, "sample_id")],
            meanings=meanings,
            assignment_path=assignment_path,
        )
        for row in selected
    ]
    manifest_bytes = _jsonl_bytes(pilot_rows)
    provenance = {
        "schema_name": "mprisk_gt_prompt_context_v2_pilot_provenance",
        "protocol_version": "prompt_context_v2",
        "context_priority": list(CONTEXT_PRIORITY),
        "source_count": len(archive_rows),
        "context_source_counts": dict(context_source_counts),
        "pilot_count": len(pilot_rows),
        "pilot_sample_ids": list(PILOT_SAMPLE_IDS),
        "selection": "lexicographic_sample_id_first_2_per_data_type_protocol_cell",
        "source_artifacts": {
            "archive_manifest": {"path": str(archive_path), "sha256": _sha256(archive_path)},
            "assignments": {"path": str(assignment_path), "sha256": _sha256(assignment_path)},
            "dictionary": {"path": str(dictionary_path), "sha256": _sha256(dictionary_path)},
        },
        "pilot_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    return pilot_rows, provenance


def write_prompt_context_v2_pilot(
    repo_root: str | Path,
    *,
    manifest_path: str | Path,
    provenance_path: str | Path,
) -> tuple[Path, Path]:
    """Write or verify byte-identical frozen prompt-context v2 pilot artifacts."""
    root = Path(repo_root).resolve()
    rows, provenance = build_prompt_context_v2_pilot(root)
    manifest = _resolve_under_root(root, manifest_path)
    provenance_file = _resolve_under_root(root, provenance_path)
    _write_immutable(manifest, _jsonl_bytes(rows))
    _write_immutable(
        provenance_file,
        (json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )
    return manifest, provenance_file


def _pilot_row(
    *,
    archive_row: dict[str, Any],
    assignment: dict[str, Any],
    meanings: dict[str, dict[str, Any]],
    assignment_path: Path,
) -> dict[str, Any]:
    sample_id = _text(archive_row, "sample_id")
    if assignment.get("sample_id") != sample_id:
        raise ValueError(f"Prompt-context v2 assignment mismatch: {sample_id}")
    source_row_sha256 = _text(archive_row, "source_row_sha256")
    if assignment.get("source_row_sha256") != source_row_sha256:
        raise ValueError(f"Prompt-context v2 source hash mismatch: {sample_id}")
    semantic_id = _text(assignment, "archetype_semantic_id")
    meaning = meanings.get(semantic_id)
    if meaning is None or meaning.get("data_type") != archive_row.get("data_type"):
        raise ValueError(f"Prompt-context v2 dictionary mismatch: {sample_id}")
    context_text, context_source = resolve_context(archive_row["source_row"])
    if context_source != "ltx2_prompt":
        raise ValueError(
            f"Prompt-context v2 pilot row was not recovered from ltx2_prompt: {sample_id}"
        )
    media = archive_row.get("media")
    if not isinstance(media, dict):
        raise TypeError(f"Prompt-context v2 media must be an object: {sample_id}")
    media_path = Path(_text(media, "model_input_path"))
    media_sha256 = _text(media, "model_input_sha256")
    if not media_path.is_file() or _sha256(media_path) != media_sha256:
        raise ValueError(f"Prompt-context v2 media hash mismatch: {sample_id}")
    assignment_sha256 = hashlib.sha256(_canonical_json(assignment).encode()).hexdigest()
    return {
        "schema_name": PILOT_ROW_SCHEMA,
        "protocol_version": "prompt_context_v2",
        "sample_id": sample_id,
        "source_archive": _text(archive_row, "source_archive"),
        "data_type": _text(archive_row, "data_type"),
        "protocol": _text(archive_row, "protocol"),
        "archetype": {
            "id": semantic_id,
            "name": _text(meaning, "canonical_name"),
            "canonical_meaning": _text(meaning, "canonical_meaning"),
        },
        "dialogue": _text(archive_row, "dialogue_text"),
        "context_text": context_text,
        "context_source": context_source,
        "surface_emotion": meaning.get("surface_emotion"),
        "media": {"path": str(media_path), "sha256": media_sha256},
        "source_assignment": {
            "path": str(assignment_path),
            "schema_name": _text(assignment, "schema_name"),
            "dictionary_id": _text(assignment, "dictionary_id"),
            "assignment_source": _text(assignment, "assignment_source"),
            "source_row_sha256": source_row_sha256,
            "assignment_sha256": assignment_sha256,
        },
        "source_row_sha256": source_row_sha256,
    }


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _text(row: dict[str, Any], key: str) -> str:
    value = _optional_text(row.get(key))
    if value is None:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"JSONL rows must be objects: {path}")
    return rows


def _index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = _text(row, key)
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode()


def _resolve_under_root(root: Path, path: str | Path) -> Path:
    value = Path(path)
    resolved = value.resolve() if value.is_absolute() else (root / value).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Prompt-context v2 output escapes repository: {resolved}")
    return resolved


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Frozen prompt-context v2 artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
