"""Immutable snapshot builder for the generated round-one source archives."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from mprisk.config.loader import load_yaml

ARCHIVE_ORDER = ("accept_a_svt", "accept_a_va", "accept_c_svt", "accept_c_va")
TEMPLATE_TRIGGER = re.compile(r"^T[1-4]$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArchiveSpec(StrictModel):
    data_type: Literal["A", "C"]
    sample_type: Literal["Conflict", "Aligned"]
    protocol: Literal["VT", "VA"]
    media_field: Literal["silent", "primary"]
    expected_count: int
    expected_gt_eligible: int


class SilentCopyOverride(StrictModel):
    source_archive: Literal["accept_a_svt", "accept_a_va", "accept_c_svt", "accept_c_va"]
    sample_id: str


class FreezeConfig(StrictModel):
    schema_name: Literal["mprisk_generated_archive_freeze_config_v1"]
    freeze_id: Literal["generated_round1_v1"]
    source_root: Path
    output_root: Path
    external_media_root: Path
    official_archetypes_path: Path
    official_c_templates_path: Path
    ffmpeg_path: Path
    ffprobe_path: Path
    archives: dict[str, ArchiveSpec]
    silent_copy_overrides: list[SilentCopyOverride]

    @field_validator(
        "source_root",
        "external_media_root",
        "official_archetypes_path",
        "official_c_templates_path",
        "ffmpeg_path",
        "ffprobe_path",
    )
    @classmethod
    def external_paths_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("source and tool paths must be absolute")
        return value

    @model_validator(mode="after")
    def archive_contract_must_be_exact(self) -> FreezeConfig:
        if set(self.archives) != set(ARCHIVE_ORDER):
            raise ValueError(f"archives must contain exactly: {', '.join(ARCHIVE_ORDER)}")
        if self.output_root.is_absolute():
            raise ValueError("output_root must be relative to the repository")
        overrides = [(row.source_archive, row.sample_id) for row in self.silent_copy_overrides]
        if len(overrides) != len(set(overrides)):
            raise ValueError("silent_copy_overrides must be unique")
        return self


@dataclass(frozen=True)
class FreezeResult:
    archive_manifest_path: Path
    gt_eligible_path: Path
    provenance_path: Path
    total_count: int
    gt_eligible_count: int


def load_freeze_config(path: str | Path) -> FreezeConfig:
    return FreezeConfig.model_validate(load_yaml(path))


def freeze_generated_round1(
    *,
    repo_root: str | Path,
    config_path: str | Path,
) -> FreezeResult:
    """Freeze all four accepted archives without modifying the previous delivery."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_freeze_config(config_file)
    output_root = (root / config.output_root).resolve()
    _require_within(output_root, root, label="output_root")
    _require_file(config.official_archetypes_path, label="official_archetypes_path")
    _require_file(config.official_c_templates_path, label="official_c_templates_path")
    _require_executable(config.ffmpeg_path, label="ffmpeg_path")
    _require_executable(config.ffprobe_path, label="ffprobe_path")

    official_archetypes = _literal_assignment(
        config.official_archetypes_path,
        "ARCHETYPES_GLM",
    )
    official_c_templates = _literal_assignment(
        config.official_c_templates_path,
        "EMOTION_VARIANTS",
    )
    _validate_official_sources(official_archetypes, official_c_templates)
    overrides = {(row.source_archive, row.sample_id) for row in config.silent_copy_overrides}

    frozen_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    index_provenance: dict[str, dict[str, Any]] = {}
    media_hash_cache: dict[Path, tuple[str, int]] = {}
    seen_keys: set[tuple[str, str]] = set()

    for source_archive in ARCHIVE_ORDER:
        spec = config.archives[source_archive]
        archive_root = (config.source_root / source_archive).resolve()
        _require_within(archive_root, config.source_root.resolve(), label=source_archive)
        index_path = archive_root / "index.jsonl"
        rows = _read_jsonl_strict(index_path)
        if len(rows) != spec.expected_count:
            raise ValueError(
                f"{source_archive} count mismatch: expected {spec.expected_count}, got {len(rows)}"
            )
        index_provenance[source_archive] = {
            "path": str(index_path),
            "sha256": _sha256(index_path),
            "rows": len(rows),
        }
        for source_line, source_row in enumerate(rows, start=1):
            frozen_row, eligible_row = _freeze_source_row(
                config=config,
                source_archive=source_archive,
                spec=spec,
                archive_root=archive_root,
                source_line=source_line,
                source_row=source_row,
                official_archetypes=official_archetypes,
                official_c_templates=official_c_templates,
                silent_override=(source_archive, _required_text(source_row, "sample_id"))
                in overrides,
                media_hash_cache=media_hash_cache,
            )
            key = (frozen_row["source_archive"], frozen_row["original_variant_id"])
            if key in seen_keys:
                raise ValueError(f"Duplicate generated round-one key: {key!r}")
            seen_keys.add(key)
            frozen_rows.append(frozen_row)
            if eligible_row is not None:
                eligible_rows.append(eligible_row)

    _validate_final_counts(config, frozen_rows, eligible_rows)
    _validate_overrides_consumed(overrides, frozen_rows)

    archive_bytes = _jsonl_bytes(frozen_rows)
    eligible_bytes = _jsonl_bytes(eligible_rows)
    artifacts = {
        "archive_manifest": _artifact_payload(
            config.output_root / "archive_manifest.jsonl", archive_bytes
        ),
        "gt_eligible": _artifact_payload(config.output_root / "gt_eligible.jsonl", eligible_bytes),
    }
    provenance = {
        "schema_name": "mprisk_generated_archive_provenance_v1",
        "freeze_id": config.freeze_id,
        "builder": "mprisk.data.generated_archive_freeze.freeze_generated_round1",
        "config": {"path": str(config_file), "sha256": _sha256(config_file)},
        "official_sources": {
            "archetypes_glm": {
                "path": str(config.official_archetypes_path),
                "sha256": _sha256(config.official_archetypes_path),
            },
            "c_emotion_variants": {
                "path": str(config.official_c_templates_path),
                "sha256": _sha256(config.official_c_templates_path),
            },
        },
        "source_indexes": index_provenance,
        "counts": _counts_payload(frozen_rows, eligible_rows),
        "media": {
            "files_hashed": len(media_hash_cache),
            "bytes_hashed": sum(size for _, size in media_hash_cache.values()),
            "large_media_committed": False,
            "silent_copy_overrides": [
                {"source_archive": archive, "sample_id": sample_id}
                for archive, sample_id in sorted(overrides)
            ],
        },
        "artifacts": artifacts,
        "primary_key": ["source_archive", "original_variant_id"],
    }
    provenance_bytes = _json_bytes(provenance)
    outputs = {
        output_root / "archive_manifest.jsonl": archive_bytes,
        output_root / "gt_eligible.jsonl": eligible_bytes,
        output_root / "provenance.json": provenance_bytes,
    }
    _write_immutable_outputs(outputs)
    return FreezeResult(
        archive_manifest_path=output_root / "archive_manifest.jsonl",
        gt_eligible_path=output_root / "gt_eligible.jsonl",
        provenance_path=output_root / "provenance.json",
        total_count=len(frozen_rows),
        gt_eligible_count=len(eligible_rows),
    )


def select_natural_context(row: dict[str, Any]) -> tuple[str, str] | None:
    setting = _optional_text(row, "setting")
    trigger = _optional_text(row, "trigger")
    if setting:
        return setting, "setting"
    if trigger and TEMPLATE_TRIGGER.fullmatch(trigger) is None:
        return trigger, "trigger"
    return None


def _freeze_source_row(
    *,
    config: FreezeConfig,
    source_archive: str,
    spec: ArchiveSpec,
    archive_root: Path,
    source_line: int,
    source_row: dict[str, Any],
    official_archetypes: dict[Any, Any],
    official_c_templates: list[Any],
    silent_override: bool,
    media_hash_cache: dict[Path, tuple[str, int]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sample_id = _required_text(source_row, "sample_id")
    original_variant_id = _required_text(source_row, "original_variant_id")
    if source_row.get("bucket") != "accept":
        raise ValueError(f"{source_archive}:{source_line}: bucket must be 'accept'")
    if source_row.get("data_type") != spec.data_type:
        raise ValueError(f"{source_archive}:{source_line}: data_type does not match config")
    expected_conflict_type = "silent_video_text" if spec.protocol == "VT" else "video_audio"
    conflict_type = source_row.get("conflict_type")
    if conflict_type is not None and conflict_type != expected_conflict_type:
        raise ValueError(f"{source_archive}:{source_line}: conflict_type does not match config")

    files = source_row.get("files")
    if not isinstance(files, dict):
        raise TypeError(f"{source_archive}:{source_line}: files must be an object")
    primary_path = _resolve_source_media(archive_root, files.get("primary"), label="primary")
    if silent_override:
        if spec.protocol != "VT" or spec.media_field != "silent":
            raise ValueError("silent-copy overrides are only valid for VT archives")
        model_path = (
            config.external_media_root / source_archive / f"{sample_id}.silent.mp4"
        ).resolve()
        _require_within(
            model_path,
            config.external_media_root.resolve(),
            label="silent-copy output",
        )
        _ensure_deterministic_silent_copy(
            source=primary_path,
            target=model_path,
            ffmpeg_path=config.ffmpeg_path,
            ffprobe_path=config.ffprobe_path,
        )
        derivation = "ffmpeg_stream_copy_no_audio"
    elif (
        spec.protocol == "VT"
        and spec.media_field == "silent"
        and files.get("silent") is None
        and str(files.get("primary", "")).endswith(".silent.mp4")
    ):
        model_path = primary_path
        _verify_video_without_audio(model_path, config.ffprobe_path)
        derivation = "recorded_primary_silent"
    else:
        model_path = _resolve_source_media(
            archive_root,
            files.get(spec.media_field),
            label=spec.media_field,
        )
        derivation = "recorded_silent" if spec.media_field == "silent" else "recorded_primary"

    primary_sha, primary_size = _hash_media(primary_path, media_hash_cache)
    model_sha, model_size = _hash_media(model_path, media_hash_cache)
    source_row_sha = hashlib.sha256(_canonical_json(source_row).encode("utf-8")).hexdigest()
    namespaced_sample_id = f"gen:{source_archive}:{sample_id}"
    frozen_row = {
        "schema_name": "mprisk_generated_archive_row_v1",
        "freeze_id": config.freeze_id,
        "sample_id": namespaced_sample_id,
        "source_archive": source_archive,
        "source_line": source_line,
        "source_sample_id": sample_id,
        "original_variant_id": original_variant_id,
        "data_type": spec.data_type,
        "sample_type": spec.sample_type,
        "protocol": spec.protocol,
        "dialogue_text": _optional_text(source_row, "dialogue_text"),
        "setting_text": _optional_text(source_row, "setting"),
        "trigger_text": _optional_text(source_row, "trigger"),
        "media": {
            "primary_path": str(primary_path),
            "primary_sha256": primary_sha,
            "primary_bytes": primary_size,
            "model_input_path": str(model_path),
            "model_input_sha256": model_sha,
            "model_input_bytes": model_size,
            "derivation": derivation,
        },
        "source_row_sha256": source_row_sha,
        "source_row": source_row,
    }

    context = select_natural_context(source_row)
    dialogue = frozen_row["dialogue_text"]
    anchor = _resolve_anchor(
        source_archive=source_archive,
        source_line=source_line,
        row=source_row,
        official_archetypes=official_archetypes,
        official_c_templates=official_c_templates,
    )
    if not dialogue or context is None or anchor is None:
        return frozen_row, None
    context_text, context_source = context
    eligible_row = {
        "schema_name": "mprisk_generated_gt_eligible_v1",
        "freeze_id": config.freeze_id,
        "sample_id": namespaced_sample_id,
        "source_archive": source_archive,
        "original_variant_id": original_variant_id,
        "data_type": spec.data_type,
        "sample_type": spec.sample_type,
        "protocol": spec.protocol,
        "dialogue_text": dialogue,
        "setting_text": frozen_row["setting_text"],
        "trigger_text": frozen_row["trigger_text"],
        "context_text": context_text,
        "context_source": context_source,
        "anchor": anchor,
        "model_input_path": str(model_path),
        "model_input_sha256": model_sha,
        "source_row_sha256": source_row_sha,
    }
    return frozen_row, eligible_row


def _resolve_anchor(
    *,
    source_archive: str,
    source_line: int,
    row: dict[str, Any],
    official_archetypes: dict[Any, Any],
    official_c_templates: list[Any],
) -> dict[str, Any] | None:
    archetype_id = row.get("archetype_id")
    archetype_name = _optional_text(row, "archetype_name")
    has_id = archetype_id is not None
    has_name = archetype_name is not None
    if has_id != has_name:
        raise ValueError(
            f"{source_archive}:{source_line}: recorded archetype_id/name must be a complete pair"
        )
    if has_id:
        if isinstance(archetype_id, bool) or not isinstance(archetype_id, int):
            raise TypeError(f"{source_archive}:{source_line}: archetype_id must be an integer")
        return {
            "source_kind": "recorded",
            "archetype_id": archetype_id,
            "archetype_name": archetype_name,
            "emotion": _required_anchor_emotion(row, source_archive, source_line),
            "surface_emotion": _optional_text(row, "surface_emotion"),
        }
    if row.get("data_type") != "C":
        return None
    emotion = _required_anchor_emotion(row, source_archive, source_line)
    setting = _optional_text(row, "setting")
    dialogue = _optional_text(row, "dialogue_text")
    if not setting or not dialogue:
        return None
    template_matches = [
        item
        for item in official_c_templates
        if isinstance(item, dict)
        and item.get("emotion") == emotion
        and item.get("setting") == setting
        and item.get("dialogue") == dialogue
    ]
    if len(template_matches) != 1:
        raise ValueError(
            f"{source_archive}:{source_line}: expected one official C template match, "
            f"got {len(template_matches)}"
        )
    archetype_matches = [
        (key, item)
        for key, item in official_archetypes.items()
        if isinstance(item, dict) and item.get("type") == "C" and item.get("gt") == emotion
    ]
    if len(archetype_matches) != 1:
        raise ValueError(
            f"{source_archive}:{source_line}: expected one official C emotion mapping, "
            f"got {len(archetype_matches)}"
        )
    official_id, official_archetype = archetype_matches[0]
    template = template_matches[0]
    return {
        "source_kind": "official_template",
        "archetype_id": official_id,
        "archetype_name": official_archetype["name"],
        "emotion": emotion,
        "surface_emotion": None,
        "template_variant": template["v"],
        "template_lookup": {
            "emotion": emotion,
            "setting": setting,
            "dialogue_text": dialogue,
        },
    }


def _required_anchor_emotion(row: dict[str, Any], archive: str, line: int) -> str:
    value = _optional_text(row, "gt_emotion") or _optional_text(row, "emotion")
    if not value:
        raise ValueError(f"{archive}:{line}: anchor emotion is missing")
    return value


def _validate_official_sources(archetypes: Any, templates: Any) -> None:
    if not isinstance(archetypes, dict) or not archetypes:
        raise TypeError("ARCHETYPES_GLM must be a non-empty dictionary literal")
    if not isinstance(templates, list) or not templates:
        raise TypeError("EMOTION_VARIANTS must be a non-empty list literal")
    c_emotions = [
        item.get("gt")
        for item in archetypes.values()
        if isinstance(item, dict) and item.get("type") == "C"
    ]
    duplicates = [emotion for emotion, count in Counter(c_emotions).items() if count != 1]
    if duplicates:
        raise ValueError("Official C archetype emotions must be unique: " + ", ".join(duplicates))
    keys: list[tuple[Any, Any, Any]] = []
    for item in templates:
        if not isinstance(item, dict):
            raise TypeError("Each EMOTION_VARIANTS item must be an object")
        keys.append((item.get("emotion"), item.get("setting"), item.get("dialogue")))
    if len(keys) != len(set(keys)):
        raise ValueError("Official C template lookup keys must be unique")


def _validate_final_counts(
    config: FreezeConfig,
    frozen_rows: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
) -> None:
    expected_total = sum(spec.expected_count for spec in config.archives.values())
    if len(frozen_rows) != expected_total:
        raise ValueError(
            f"Frozen total mismatch: expected {expected_total}, got {len(frozen_rows)}"
        )
    frozen_counts = Counter(row["source_archive"] for row in frozen_rows)
    eligible_counts = Counter(row["source_archive"] for row in eligible_rows)
    for archive in ARCHIVE_ORDER:
        spec = config.archives[archive]
        if frozen_counts[archive] != spec.expected_count:
            raise ValueError(f"{archive} frozen count mismatch")
        if eligible_counts[archive] != spec.expected_gt_eligible:
            raise ValueError(
                f"{archive} GT-eligible count mismatch: expected "
                f"{spec.expected_gt_eligible}, got {eligible_counts[archive]}"
            )


def _validate_overrides_consumed(
    overrides: set[tuple[str, str]], frozen_rows: list[dict[str, Any]]
) -> None:
    derived = {
        (row["source_archive"], row["source_sample_id"])
        for row in frozen_rows
        if row["media"]["derivation"] == "ffmpeg_stream_copy_no_audio"
    }
    if derived != overrides:
        raise ValueError(f"Silent-copy override mismatch: expected {overrides}, got {derived}")


def _counts_payload(
    frozen_rows: list[dict[str, Any]], eligible_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "total": len(frozen_rows),
        "gt_eligible": len(eligible_rows),
        "ineligible": len(frozen_rows) - len(eligible_rows),
        "by_archive": dict(Counter(row["source_archive"] for row in frozen_rows)),
        "gt_eligible_by_archive": dict(Counter(row["source_archive"] for row in eligible_rows)),
        "gt_eligible_by_protocol": dict(Counter(row["protocol"] for row in eligible_rows)),
        "gt_eligible_by_data_type": dict(Counter(row["data_type"] for row in eligible_rows)),
        "gt_eligible_context_source": dict(Counter(row["context_source"] for row in eligible_rows)),
        "gt_eligible_anchor_source": dict(
            Counter(row["anchor"]["source_kind"] for row in eligible_rows)
        ),
    }


def _ensure_deterministic_silent_copy(
    *,
    source: Path,
    target: Path,
    ffmpeg_path: Path,
    ffprobe_path: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".tmp.mp4", dir=target.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        command = [
            str(ffmpeg_path),
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        _verify_video_without_audio(temporary, ffprobe_path)
        if target.exists():
            _verify_video_without_audio(target, ffprobe_path)
            if _sha256(target) != _sha256(temporary):
                raise ValueError(
                    f"Existing silent copy differs from deterministic output: {target}"
                )
        else:
            os.replace(temporary, target)
        _verify_video_without_audio(target, ffprobe_path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_video_without_audio(path: Path, ffprobe_path: Path) -> None:
    completed = subprocess.run(
        [
            str(ffprobe_path),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ValueError(f"ffprobe returned no stream list for {path}")
    stream_types = [item.get("codec_type") for item in streams if isinstance(item, dict)]
    if "video" not in stream_types or "audio" in stream_types:
        raise ValueError(f"Silent-copy verification failed for {path}: {stream_types}")


def _literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} assignment in {path}")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} in {path} must be a literal") from exc


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    _require_file(path, label="source index")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number}: row must be an object")
            rows.append(payload)
    return rows


def _resolve_source_media(archive_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Required {label} media path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} media path must be a contained relative path: {value}")
    candidate = archive_root / relative
    if candidate.is_symlink():
        raise ValueError(f"{label} media path must not be a symlink: {candidate}")
    path = candidate.resolve()
    _require_within(path, archive_root, label=f"{label} media")
    _require_file(path, label=f"{label} media")
    return path


def _hash_media(path: Path, cache: dict[Path, tuple[str, int]]) -> tuple[str, int]:
    if path not in cache:
        cache[path] = (_sha256(path), path.stat().st_size)
    return cache[path]


def _required_text(row: dict[str, Any], field: str) -> str:
    value = _optional_text(row, field)
    if value is None:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_text(row: dict[str, Any], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    stripped = value.strip()
    return stripped or None


def _require_file(path: Path, *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} must not be empty: {path}")


def _require_executable(path: Path, *, label: str) -> None:
    _require_file(path, label=label)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{label} is not executable: {path}")


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its allowed root: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(_canonical_json(row) + "\n" for row in rows)).encode("utf-8")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _artifact_payload(path: Path, content: bytes) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_immutable_outputs(outputs: dict[Path, bytes]) -> None:
    conflicts = [
        path for path, content in outputs.items() if path.exists() and path.read_bytes() != content
    ]
    if conflicts:
        raise ValueError(
            "Immutable freeze outputs already exist with different content: "
            + ", ".join(str(path) for path in conflicts)
        )
    for path, content in outputs.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
