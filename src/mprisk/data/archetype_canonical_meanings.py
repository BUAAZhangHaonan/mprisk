"""Build and verify the frozen canonical archetype-meaning dictionary."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from mprisk.config.loader import load_yaml
from mprisk.data.generated_archive_freeze import (
    _artifact_payload,
    _canonical_json,
    _json_bytes,
    _jsonl_bytes,
    _literal_assignment,
    _read_jsonl_strict,
    _sha256,
    _write_immutable_outputs,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DictionaryConfig(StrictModel):
    schema_name: Literal["mprisk_archetype_canonical_dictionary_config_v1"]
    dictionary_id: Literal["archetype_canonical_meanings_v1"]
    freeze_root: Path
    dictionary_file: Literal["archetype_canonical_meanings_v1.jsonl"]
    assignments_file: Literal["archetype_semantic_assignments_v1.jsonl"]
    review_queue_file: Literal["archetype_canonical_review_queue_v1.jsonl"]
    provenance_file: Literal["archetype_canonical_meanings_v1.provenance.json"]
    max_description_words: int
    recorded_name_aliases: dict[str, str]
    recorded_surface_aliases: dict[str, str]

    @field_validator("freeze_root")
    @classmethod
    def freeze_root_must_be_relative(cls, value: Path) -> Path:
        if value.is_absolute():
            raise ValueError("freeze_root must be relative to the repository")
        return value

    @field_validator("max_description_words")
    @classmethod
    def word_limit_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_description_words must be positive")
        return value


@dataclass(frozen=True)
class DictionaryResult:
    dictionary_path: Path
    assignments_path: Path
    review_queue_path: Path
    provenance_path: Path
    dictionary_count: int
    assignment_count: int
    review_count: int


@dataclass(frozen=True)
class _PreparedArtifacts:
    outputs: dict[Path, bytes]
    result: DictionaryResult


def load_dictionary_config(path: str | Path) -> DictionaryConfig:
    return DictionaryConfig.model_validate(load_yaml(path))


def build_archetype_canonical_meanings(
    *, repo_root: str | Path, config_path: str | Path
) -> DictionaryResult:
    prepared = _prepare_artifacts(repo_root=repo_root, config_path=config_path)
    _write_immutable_outputs(prepared.outputs)
    if prepared.result.review_count:
        raise ValueError(
            f"Canonical meanings require review for {prepared.result.review_count} archetype(s); "
            f"see {prepared.result.review_queue_path}"
        )
    return prepared.result


def verify_archetype_canonical_meanings(
    *, repo_root: str | Path, config_path: str | Path
) -> DictionaryResult:
    prepared = _prepare_artifacts(repo_root=repo_root, config_path=config_path)
    missing = [path for path in prepared.outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing canonical dictionary artifacts: " + ", ".join(map(str, missing))
        )
    mismatches = [
        path for path, expected in prepared.outputs.items() if path.read_bytes() != expected
    ]
    if mismatches:
        raise ValueError(
            "Canonical dictionary artifact mismatch: " + ", ".join(map(str, mismatches))
        )
    if prepared.result.review_count:
        raise ValueError(
            f"Canonical dictionary is blocked by {prepared.result.review_count} review item(s)"
        )
    return prepared.result


def normalize_source_description(value: Any, *, max_words: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("official desc must be a non-empty string")
    meaning = " ".join(value.split())
    if not meaning.startswith("Person "):
        raise ValueError("official desc must be a scene-independent Person statement")
    boundaries = re.findall(r"[.!?](?=\s|$)", meaning)
    if boundaries:
        if len(boundaries) != 1 or meaning[-1] not in ".!?":
            raise ValueError("official desc must contain exactly one sentence")
    else:
        meaning += "."
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", meaning)
    if not 4 <= len(words) <= max_words:
        raise ValueError(
            f"official desc must contain 4-{max_words} English words, got {len(words)}"
        )
    if any(character in meaning for character in ("\n", "\r", '"')):
        raise ValueError("official desc must not contain dialogue or line breaks")
    return meaning


def _prepare_artifacts(*, repo_root: str | Path, config_path: str | Path) -> _PreparedArtifacts:
    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_dictionary_config(config_file)
    freeze_root = (root / config.freeze_root).resolve()
    _require_within_repo(freeze_root, root)
    freeze_provenance_path = freeze_root / "provenance.json"
    freeze_provenance = _read_json_object(freeze_provenance_path)
    if freeze_provenance.get("freeze_id") != "generated_round1_v1":
        raise ValueError("Canonical dictionary requires generated_round1_v1")
    _verify_freeze_artifacts(root, freeze_provenance)
    freeze_provenance_sha = _sha256(freeze_provenance_path)

    source_info = freeze_provenance.get("official_sources")
    if not isinstance(source_info, dict):
        raise TypeError("freeze provenance official_sources must be an object")
    archetype_source = _verified_source(source_info, "archetypes_glm")
    c_template_source = _verified_source(source_info, "c_emotion_variants")
    official_archetypes = _literal_assignment(archetype_source[0], "ARCHETYPES_GLM")
    official_c_templates = _literal_assignment(c_template_source[0], "EMOTION_VARIANTS")
    _validate_official_definitions(official_archetypes, official_c_templates)

    snapshot_rows = _read_jsonl_strict(freeze_root / "archive_manifest.jsonl")
    eligible_rows = _read_jsonl_strict(freeze_root / "gt_eligible.jsonl")
    eligible_by_id = _index_unique(eligible_rows, "sample_id", source="gt_eligible")
    snapshot_by_id = _index_unique(snapshot_rows, "sample_id", source="archive_manifest")
    if set(eligible_by_id) - set(snapshot_by_id):
        raise ValueError("GT-eligible rows must be a subset of the frozen snapshot")

    assignments, grouped = _build_assignments(
        snapshot_rows=snapshot_rows,
        eligible_by_id=eligible_by_id,
        official_archetypes=official_archetypes,
        official_c_templates=official_c_templates,
        recorded_name_aliases=config.recorded_name_aliases,
        recorded_surface_aliases=config.recorded_surface_aliases,
    )
    dictionary_rows, review_rows = _build_dictionary_rows(
        grouped=grouped,
        eligible_by_id=eligible_by_id,
        official_archetypes=official_archetypes,
        official_c_templates=official_c_templates,
        archetype_source=archetype_source,
        c_template_source=c_template_source,
        freeze_provenance_sha=freeze_provenance_sha,
        dictionary_id=config.dictionary_id,
        max_words=config.max_description_words,
    )
    _validate_coverage(assignments, dictionary_rows, review_rows, eligible_by_id)

    dictionary_bytes = _jsonl_bytes(dictionary_rows)
    assignments_bytes = _jsonl_bytes(assignments)
    review_bytes = _jsonl_bytes(review_rows)
    relative_artifacts = {
        "dictionary": config.freeze_root / config.dictionary_file,
        "assignments": config.freeze_root / config.assignments_file,
        "review_queue": config.freeze_root / config.review_queue_file,
    }
    artifacts = {
        "dictionary": _artifact_payload(relative_artifacts["dictionary"], dictionary_bytes),
        "assignments": _artifact_payload(relative_artifacts["assignments"], assignments_bytes),
        "review_queue": _artifact_payload(relative_artifacts["review_queue"], review_bytes),
    }
    provenance = {
        "schema_name": "mprisk_archetype_canonical_dictionary_provenance_v1",
        "dictionary_id": config.dictionary_id,
        "status": "review_required" if review_rows else "complete",
        "builder": "mprisk.data.archetype_canonical_meanings.build_archetype_canonical_meanings",
        "config": {"path": str(config_file), "sha256": _sha256(config_file)},
        "freeze_binding": {
            "path": str(freeze_provenance_path),
            "sha256": freeze_provenance_sha,
            "freeze_id": freeze_provenance["freeze_id"],
            "archive_manifest_sha256": freeze_provenance["artifacts"]["archive_manifest"]["sha256"],
            "gt_eligible_sha256": freeze_provenance["artifacts"]["gt_eligible"]["sha256"],
        },
        "official_sources": {
            "archetypes_glm": {
                "path": str(archetype_source[0]),
                "sha256": archetype_source[1],
            },
            "c_emotion_variants": {
                "path": str(c_template_source[0]),
                "sha256": c_template_source[1],
            },
        },
        "counts": {
            "dictionary": len(dictionary_rows),
            "assignments": len(assignments),
            "review_queue": len(review_rows),
            "snapshot_by_data_type": dict(Counter(row["data_type"] for row in assignments)),
            "dictionary_by_data_type": dict(Counter(row["data_type"] for row in dictionary_rows)),
            "gt_eligible_assignments": sum(row["gt_eligible"] for row in assignments),
        },
        "artifacts": artifacts,
        "dictionary_primary_key": ["archetype_semantic_id"],
        "assignment_primary_key": ["sample_id"],
    }
    provenance_bytes = _json_bytes(provenance)
    outputs = {
        freeze_root / config.dictionary_file: dictionary_bytes,
        freeze_root / config.assignments_file: assignments_bytes,
        freeze_root / config.review_queue_file: review_bytes,
        freeze_root / config.provenance_file: provenance_bytes,
    }
    result = DictionaryResult(
        dictionary_path=freeze_root / config.dictionary_file,
        assignments_path=freeze_root / config.assignments_file,
        review_queue_path=freeze_root / config.review_queue_file,
        provenance_path=freeze_root / config.provenance_file,
        dictionary_count=len(dictionary_rows),
        assignment_count=len(assignments),
        review_count=len(review_rows),
    )
    return _PreparedArtifacts(outputs=outputs, result=result)


def _build_assignments(
    *,
    snapshot_rows: list[dict[str, Any]],
    eligible_by_id: dict[str, dict[str, Any]],
    official_archetypes: dict[Any, Any],
    official_c_templates: list[Any],
    recorded_name_aliases: dict[str, str],
    recorded_surface_aliases: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    official_c_emotions = {
        item["emotion"] for item in official_c_templates if isinstance(item, dict)
    }
    assignments: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in snapshot_rows:
        source_row = row.get("source_row")
        if not isinstance(source_row, dict):
            raise TypeError(f"{row.get('sample_id')}: source_row must be an object")
        data_type = row.get("data_type")
        true_emotion = _true_emotion(source_row)
        if data_type == "A":
            archetype_id = source_row.get("archetype_id")
            if isinstance(archetype_id, bool) or not isinstance(archetype_id, int):
                raise TypeError(f"{row.get('sample_id')}: A archetype_id must be an integer")
            definition = official_archetypes.get(archetype_id)
            if not isinstance(definition, dict) or definition.get("type") != "A":
                raise ValueError(f"{row.get('sample_id')}: missing official A archetype definition")
            if definition.get("gt") != true_emotion:
                raise ValueError(f"{row.get('sample_id')}: A true emotion conflicts with source")
            recorded_name = _required_text(source_row, "archetype_name")
            canonical_name = definition.get("name")
            if (
                recorded_name != canonical_name
                and recorded_name_aliases.get(recorded_name) != canonical_name
            ):
                raise ValueError(f"{row.get('sample_id')}: A archetype name conflicts with source")
            recorded_surface = _required_text(source_row, "surface_emotion")
            canonical_surface = definition.get("surface")
            if (
                recorded_surface != canonical_surface
                and recorded_surface_aliases.get(recorded_surface) != canonical_surface
            ):
                raise ValueError(f"{row.get('sample_id')}: A surface emotion conflicts with source")
            semantic_id = f"A:{archetype_id:03d}"
            assignment_source = "recorded_archetype_id"
        elif data_type == "C":
            if true_emotion not in official_c_emotions:
                raise ValueError(
                    f"{row.get('sample_id')}: C emotion is absent from EMOTION_VARIANTS"
                )
            c_matches = [
                archetype_id
                for archetype_id, definition in official_archetypes.items()
                if isinstance(definition, dict)
                and definition.get("type") == "C"
                and definition.get("gt") == true_emotion
            ]
            if len(c_matches) != 1:
                raise ValueError(
                    f"{row.get('sample_id')}: expected one official C emotion ID, "
                    f"got {len(c_matches)}"
                )
            semantic_id = f"C:{int(c_matches[0]):03d}"
            assignment_source = "official_c_emotion"
            _validate_c_exact_tuple_if_present(
                source_row, official_c_templates, row.get("sample_id")
            )
        else:
            raise ValueError(f"Unsupported generated data_type: {data_type!r}")
        sample_id = _required_text(row, "sample_id")
        eligible = eligible_by_id.get(sample_id)
        if eligible is not None and eligible.get("source_row_sha256") != row.get(
            "source_row_sha256"
        ):
            raise ValueError(f"{sample_id}: eligible/source snapshot hashes differ")
        assignment = {
            "schema_name": "mprisk_archetype_semantic_assignment_v1",
            "dictionary_id": "archetype_canonical_meanings_v1",
            "sample_id": sample_id,
            "source_archive": row["source_archive"],
            "original_variant_id": row["original_variant_id"],
            "data_type": data_type,
            "archetype_semantic_id": semantic_id,
            "true_emotion": true_emotion,
            "gt_eligible": eligible is not None,
            "assignment_source": assignment_source,
            "source_row_sha256": row["source_row_sha256"],
        }
        assignments.append(assignment)
        grouped[semantic_id].append(assignment)
    return assignments, grouped


def _build_dictionary_rows(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    eligible_by_id: dict[str, dict[str, Any]],
    official_archetypes: dict[Any, Any],
    official_c_templates: list[Any],
    archetype_source: tuple[Path, str],
    c_template_source: tuple[Path, str],
    freeze_provenance_sha: str,
    dictionary_id: str,
    max_words: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for semantic_id in sorted(grouped):
        assignments = grouped[semantic_id]
        data_type = assignments[0]["data_type"]
        true_emotions = {row["true_emotion"] for row in assignments}
        if len(true_emotions) != 1:
            raise ValueError(f"{semantic_id}: assignments contain conflicting true emotions")
        true_emotion = next(iter(true_emotions))
        if data_type == "A":
            archetype_id = int(semantic_id.split(":", maxsplit=1)[1])
            definition = official_archetypes[archetype_id]
            source_path, source_sha = archetype_source
            canonical_name = _required_text(definition, "name")
            surface_emotion = _required_text(definition, "surface")
            source_kind = "ARCHETYPES_GLM.source_defined"
            source_definition = definition
            try:
                meaning = normalize_source_description(definition.get("desc"), max_words=max_words)
            except ValueError as exc:
                review.append(
                    _review_row(
                        dictionary_id=dictionary_id,
                        semantic_id=semantic_id,
                        data_type=data_type,
                        source_path=source_path,
                        source_sha=source_sha,
                        reason=str(exc),
                        source_definition=definition,
                    )
                )
                continue
        else:
            source_path, source_sha = c_template_source
            canonical_name = true_emotion
            surface_emotion = None
            source_kind = "EMOTION_VARIANTS.exact_tuple_emotion"
            source_definition = [
                item
                for item in official_c_templates
                if isinstance(item, dict) and item.get("emotion") == true_emotion
            ]
            if not source_definition:
                raise ValueError(f"{semantic_id}: no official C templates")
            meaning = f"The modalities consistently express {true_emotion.replace('_', ' ')}."
        eligible_count = sum(row["sample_id"] in eligible_by_id for row in assignments)
        input_hash = hashlib.sha256(
            _canonical_json(
                {
                    "freeze_provenance_sha256": freeze_provenance_sha,
                    "semantic_id": semantic_id,
                    "source_definition": source_definition,
                    "assignments": [
                        {
                            "sample_id": row["sample_id"],
                            "source_row_sha256": row["source_row_sha256"],
                            "gt_eligible": row["gt_eligible"],
                        }
                        for row in assignments
                    ],
                }
            ).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "schema_name": "mprisk_archetype_canonical_meaning_v1",
                "dictionary_id": dictionary_id,
                "archetype_semantic_id": semantic_id,
                "canonical_name": canonical_name,
                "canonical_meaning": meaning,
                "true_emotion": true_emotion,
                "surface_emotion": surface_emotion,
                "data_type": data_type,
                "source_kind": source_kind,
                "source_path": str(source_path),
                "source_sha256": source_sha,
                "input_hash": input_hash,
                "status": "source_defined",
                "snapshot_sample_count": len(assignments),
                "gt_eligible_sample_count": eligible_count,
            }
        )
    return rows, review


def _review_row(
    *,
    dictionary_id: str,
    semantic_id: str,
    data_type: str,
    source_path: Path,
    source_sha: str,
    reason: str,
    source_definition: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_name": "mprisk_archetype_canonical_review_v1",
        "dictionary_id": dictionary_id,
        "archetype_semantic_id": semantic_id,
        "data_type": data_type,
        "status": "needs_review",
        "reason": reason,
        "source_kind": "ARCHETYPES_GLM.source_defined",
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "source_definition": source_definition,
    }


def _validate_coverage(
    assignments: list[dict[str, Any]],
    dictionary_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    eligible_by_id: dict[str, dict[str, Any]],
) -> None:
    if len(assignments) != len({row["sample_id"] for row in assignments}):
        raise ValueError("Semantic assignments must contain unique sample_id values")
    dictionary_ids = [row["archetype_semantic_id"] for row in dictionary_rows]
    review_ids = [row["archetype_semantic_id"] for row in review_rows]
    if len(dictionary_ids) != len(set(dictionary_ids)):
        raise ValueError("Canonical dictionary keys must be unique")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Canonical review keys must be unique")
    if set(dictionary_ids) & set(review_ids):
        raise ValueError("Dictionary and review queue keys must be disjoint")
    assignment_ids = {row["archetype_semantic_id"] for row in assignments}
    if assignment_ids != set(dictionary_ids) | set(review_ids):
        raise ValueError("Dictionary plus review queue must cover every semantic assignment")
    assigned_eligible = {row["sample_id"] for row in assignments if row["gt_eligible"]}
    if assigned_eligible != set(eligible_by_id):
        raise ValueError("Semantic assignments must cover every GT-eligible sample exactly once")
    for row in dictionary_rows:
        if row["data_type"] == "A" and row["surface_emotion"] is None:
            raise ValueError("A meanings require an official surface emotion")
        if row["data_type"] == "C" and row["surface_emotion"] is not None:
            raise ValueError("C meanings must not invent a surface emotion")
        if len(re.findall(r"[.!?](?=\s|$)", row["canonical_meaning"])) != 1:
            raise ValueError("Each canonical meaning must contain exactly one sentence")


def _validate_official_definitions(archetypes: Any, templates: Any) -> None:
    if not isinstance(archetypes, dict) or not archetypes:
        raise TypeError("ARCHETYPES_GLM must be a non-empty literal dictionary")
    if not isinstance(templates, list) or not templates:
        raise TypeError("EMOTION_VARIANTS must be a non-empty literal list")
    c_template_keys: list[tuple[Any, Any, Any]] = []
    for item in templates:
        if not isinstance(item, dict):
            raise TypeError("EMOTION_VARIANTS items must be objects")
        c_template_keys.append((item.get("emotion"), item.get("setting"), item.get("dialogue")))
    if len(c_template_keys) != len(set(c_template_keys)):
        raise ValueError("EMOTION_VARIANTS exact tuple keys must be unique")


def _validate_c_exact_tuple_if_present(
    source_row: dict[str, Any], templates: list[Any], sample_id: Any
) -> None:
    setting = source_row.get("setting")
    if setting is None:
        return
    if not isinstance(setting, str) or not setting.strip():
        raise TypeError(f"{sample_id}: C setting must be a non-empty string or null")
    dialogue = _required_text(source_row, "dialogue_text")
    emotion = _true_emotion(source_row)
    matches = [
        item
        for item in templates
        if isinstance(item, dict)
        and item.get("emotion") == emotion
        and item.get("setting") == setting.strip()
        and item.get("dialogue") == dialogue
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{sample_id}: expected one exact EMOTION_VARIANTS tuple, got {len(matches)}"
        )


def _verified_source(source_info: dict[str, Any], key: str) -> tuple[Path, str]:
    payload = source_info.get(key)
    if not isinstance(payload, dict):
        raise TypeError(f"official_sources.{key} must be an object")
    path = Path(_required_text(payload, "path"))
    expected_sha = _required_text(payload, "sha256")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Frozen official source is unavailable: {path}")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"Frozen official source hash mismatch for {key}")
    return path, expected_sha


def _verify_freeze_artifacts(root: Path, provenance: dict[str, Any]) -> None:
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("freeze provenance artifacts must be an object")
    for key in ("archive_manifest", "gt_eligible"):
        payload = artifacts.get(key)
        if not isinstance(payload, dict):
            raise TypeError(f"freeze provenance artifact {key} must be an object")
        path = (root / _required_text(payload, "path")).resolve()
        _require_within_repo(path, root)
        if not path.is_file() or _sha256(path) != _required_text(payload, "sha256"):
            raise ValueError(f"Frozen upstream artifact mismatch: {key}")


def _index_unique(
    rows: list[dict[str, Any]], field: str, *, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _required_text(row, field)
        if key in indexed:
            raise ValueError(f"Duplicate {field} in {source}: {key}")
        indexed[key] = row
    return indexed


def _true_emotion(row: dict[str, Any]) -> str:
    value = row.get("gt_emotion") or row.get("emotion")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source row requires gt_emotion or emotion")
    return value.strip()


def _required_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _require_within_repo(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc
