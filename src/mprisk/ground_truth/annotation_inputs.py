"""Strict GT Description annotation inputs and deterministic pilot construction."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from mprisk.data.generated_archive_freeze import _canonical_json, _sha256

GT_INPUT_SCHEMA_VERSION = "gt_annotation_input_v1"
GT_ANNOTATION_INPUT_SCHEMA = "mprisk_gt_annotation_input_v1"
SCENARIO_CONTEXT_SOURCE_PRIORITY = ("setting", "trigger", "source_prompt")
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
_SOURCE_CLASS_TO_SAMPLE_TYPE = {"A": "Conflict", "C": "Aligned"}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GTAnnotationArchetype(_StrictModel):
    id: str
    name: str
    canonical_meaning: str


class GTAnnotationMedia(_StrictModel):
    path: str
    sha256: str


class GTAnnotationSourceAssignment(_StrictModel):
    path: str
    schema_name: str
    dictionary_id: str
    assignment_source: str
    source_row_sha256: str
    assignment_sha256: str


class GTAnnotationSourceProvenance(_StrictModel):
    source_archive: str
    source_class_code: Literal["A", "C"]
    source_row_sha256: str
    source_assignment: GTAnnotationSourceAssignment


class GTAnnotationInput(_StrictModel):
    """One strict model input for generating exactly one GT_DESCRIPTION field."""

    schema_name: Literal["mprisk_gt_annotation_input_v1"]
    gt_input_schema_version: Literal["gt_annotation_input_v1"]
    sample_id: str
    sample_type: Literal["Conflict", "Aligned"]
    protocol: Literal["VT", "VA"]
    archetype: GTAnnotationArchetype
    dialogue: str
    scenario_context: str
    scenario_context_source: Literal["setting", "trigger", "source_prompt"]
    surface_emotion: str | None
    media: GTAnnotationMedia
    source_provenance: GTAnnotationSourceProvenance


def resolve_scenario_context(source_row: dict[str, Any]) -> tuple[str, str]:
    """Resolve one scenario context using the fixed annotation-input policy."""
    setting = _optional_text(source_row.get("setting"))
    if setting:
        return setting, "setting"
    trigger = _optional_text(source_row.get("trigger"))
    if trigger and _TEMPLATE_TRIGGER.fullmatch(trigger) is None:
        return trigger, "trigger"
    source_prompt = _optional_text(source_row.get("ltx2_prompt"))
    if source_prompt:
        return source_prompt, "source_prompt"
    raise ValueError("GT annotation input requires setting, natural trigger, or source prompt")


def build_gt_annotation_input_pilot(
    repo_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the deterministic two-per-sample-type/protocol pilot in memory."""
    root = Path(repo_root).resolve()
    input_root = root / "data/frozen/generated_round1_v1"
    archive_path = input_root / "archive_manifest.jsonl"
    assignment_path = input_root / "archetype_semantic_assignments_v1.jsonl"
    dictionary_path = input_root / "archetype_canonical_meanings_v1.jsonl"
    archive_rows = _read_jsonl(archive_path)
    assignments = _index(_read_jsonl(assignment_path), "sample_id")
    meanings = _index(_read_jsonl(dictionary_path), "archetype_semantic_id")
    if len(archive_rows) != 652 or len(assignments) != 652:
        raise ValueError("GT annotation input source must contain exactly 652 archive assignments")

    source_counts: Counter[str] = Counter()
    source_prompt_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for archive_row in archive_rows:
        source_row = archive_row.get("source_row")
        if not isinstance(source_row, dict):
            raise TypeError("Archive row source_row must be an object")
        _, scenario_source = resolve_scenario_context(source_row)
        source_counts[scenario_source] += 1
        if scenario_source == "source_prompt":
            source_code = _text(archive_row, "data_type")
            protocol = _text(archive_row, "protocol")
            source_prompt_rows.setdefault((source_code, protocol), []).append(archive_row)
    expected_counts = {"setting": 126, "trigger": 36, "source_prompt": 490}
    if dict(source_counts) != expected_counts:
        raise ValueError(f"Unexpected scenario-context source counts: {dict(source_counts)}")

    selected: list[dict[str, Any]] = []
    for source_code, protocol in (("A", "VT"), ("A", "VA"), ("C", "VT"), ("C", "VA")):
        candidates = sorted(
            source_prompt_rows.get((source_code, protocol), []),
            key=lambda row: _text(row, "sample_id"),
        )
        if len(candidates) < 2:
            raise ValueError(
                f"GT annotation pilot cell {(source_code, protocol)!r} has fewer than two rows"
            )
        selected.extend(candidates[:2])
    if tuple(_text(row, "sample_id") for row in selected) != PILOT_SAMPLE_IDS:
        raise ValueError("GT annotation input deterministic pilot IDs changed")

    pilot_rows = [
        _annotation_input_row(
            archive_row=row,
            assignment=assignments[_text(row, "sample_id")],
            meanings=meanings,
            assignment_path=assignment_path,
            repo_root=root,
        )
        for row in selected
    ]
    manifest_bytes = _jsonl_bytes(pilot_rows)
    provenance = {
        "schema_name": "mprisk_gt_annotation_input_pilot_provenance_v1",
        "gt_input_schema_version": GT_INPUT_SCHEMA_VERSION,
        "scenario_context_source_priority": list(SCENARIO_CONTEXT_SOURCE_PRIORITY),
        "source_count": len(archive_rows),
        "scenario_context_source_counts": dict(source_counts),
        "pilot_count": len(pilot_rows),
        "pilot_sample_ids": list(PILOT_SAMPLE_IDS),
        "selection": "lexicographic_first_2_per_sample_type_protocol_cell",
        "source_artifacts": {
            "archive_manifest": {
                "path": _repository_locator(root, archive_path),
                "sha256": _sha256(archive_path),
            },
            "assignments": {
                "path": _repository_locator(root, assignment_path),
                "sha256": _sha256(assignment_path),
            },
            "dictionary": {
                "path": _repository_locator(root, dictionary_path),
                "sha256": _sha256(dictionary_path),
            },
        },
        "pilot_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    return pilot_rows, provenance


def write_gt_annotation_input_pilot(
    repo_root: str | Path,
    *,
    manifest_path: str | Path,
    provenance_path: str | Path,
) -> tuple[Path, Path]:
    """Write or verify byte-identical immutable GT annotation-input artifacts."""
    root = Path(repo_root).resolve()
    rows, provenance = build_gt_annotation_input_pilot(root)
    manifest = _resolve_under_root(root, manifest_path)
    provenance_file = _resolve_under_root(root, provenance_path)
    _write_immutable(manifest, _jsonl_bytes(rows))
    _write_immutable(
        provenance_file,
        (json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )
    return manifest, provenance_file


def _annotation_input_row(
    *,
    archive_row: dict[str, Any],
    assignment: dict[str, Any],
    meanings: dict[str, dict[str, Any]],
    assignment_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    sample_id = _text(archive_row, "sample_id")
    if assignment.get("sample_id") != sample_id:
        raise ValueError(f"GT annotation assignment mismatch: {sample_id}")
    source_row_sha256 = _text(archive_row, "source_row_sha256")
    if assignment.get("source_row_sha256") != source_row_sha256:
        raise ValueError(f"GT annotation source hash mismatch: {sample_id}")
    semantic_id = _text(assignment, "archetype_semantic_id")
    source_class_code = _text(archive_row, "data_type")
    if source_class_code not in _SOURCE_CLASS_TO_SAMPLE_TYPE:
        raise ValueError(f"Unsupported source class code: {source_class_code!r}")
    meaning = meanings.get(semantic_id)
    if meaning is None or meaning.get("data_type") != source_class_code:
        raise ValueError(f"GT annotation dictionary mismatch: {sample_id}")
    scenario_context, scenario_source = resolve_scenario_context(archive_row["source_row"])
    if scenario_source != "source_prompt":
        raise ValueError(f"GT annotation pilot row is not source-prompt recovered: {sample_id}")
    media = archive_row.get("media")
    if not isinstance(media, dict):
        raise TypeError(f"GT annotation media must be an object: {sample_id}")
    media_path = Path(_text(media, "model_input_path"))
    media_sha256 = _text(media, "model_input_sha256")
    if not media_path.is_file() or _sha256(media_path) != media_sha256:
        raise ValueError(f"GT annotation media hash mismatch: {sample_id}")
    assignment_sha256 = hashlib.sha256(_canonical_json(assignment).encode()).hexdigest()
    payload = {
        "schema_name": GT_ANNOTATION_INPUT_SCHEMA,
        "gt_input_schema_version": GT_INPUT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "sample_type": _SOURCE_CLASS_TO_SAMPLE_TYPE[source_class_code],
        "protocol": _text(archive_row, "protocol"),
        "archetype": {
            "id": semantic_id,
            "name": _text(meaning, "canonical_name"),
            "canonical_meaning": _text(meaning, "canonical_meaning"),
        },
        "dialogue": _text(archive_row, "dialogue_text"),
        "scenario_context": scenario_context,
        "scenario_context_source": scenario_source,
        "surface_emotion": meaning.get("surface_emotion"),
        "media": {
            "path": _archive_media_locator(archive_row),
            "sha256": media_sha256,
        },
        "source_provenance": {
            "source_archive": _text(archive_row, "source_archive"),
            "source_class_code": source_class_code,
            "source_row_sha256": source_row_sha256,
            "source_assignment": {
                "path": _repository_locator(repo_root, assignment_path),
                "schema_name": _text(assignment, "schema_name"),
                "dictionary_id": _text(assignment, "dictionary_id"),
                "assignment_source": _text(assignment, "assignment_source"),
                "source_row_sha256": source_row_sha256,
                "assignment_sha256": assignment_sha256,
            },
        },
    }
    return GTAnnotationInput.model_validate(payload).model_dump(mode="json")


def _repository_locator(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Repository artifact escapes root: {path}") from exc
    return f"repo://{relative.as_posix()}"


def _archive_media_locator(archive_row: dict[str, Any]) -> str:
    source_row = archive_row.get("source_row")
    if not isinstance(source_row, dict):
        raise TypeError("Archive row source_row must be an object")
    files = source_row.get("files")
    if not isinstance(files, dict):
        raise TypeError("Archive source row files must be an object")
    relative_value = _text(files, "primary").replace("\\", "/")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Archive media path must be relative: {relative_value!r}")
    source_archive = _text(archive_row, "source_archive")
    return f"archive://{source_archive}/{relative.as_posix()}"


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
        raise ValueError(f"GT annotation output escapes repository: {resolved}")
    return resolved


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Immutable GT annotation artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
