"""Frozen delivery validation and deterministic manifest derivation."""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from mprisk.data.manifests import FinalManifestRow, read_jsonl, write_jsonl
from mprisk.data.splits import assign_split


DEFAULT_PROVENANCE_PATH = Path(
    "data/processed/manifests/delivery_20260714.provenance.json"
)
SAMPLE_TYPES = ("Conflict", "Aligned", "Ambiguous")
SPLITS = ("train", "val", "test")
PROTOCOL_OUTPUTS = {
    "VT": "protocol_manifests/vt_primary.jsonl",
    "VA": "protocol_manifests/va_aux.jsonl",
}
MANIFEST_ARTIFACTS = {
    "unified": "unified_sample_manifest",
    "conflict": "conflict_manifest",
    "aligned": "aligned_manifest",
    "ambiguous": "ambiguous_manifest",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactProvenance(StrictModel):
    path: str
    bytes: int
    sha256: str

    @field_validator("sha256")
    @classmethod
    def sha256_must_be_hex(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        int(value, 16)
        return value


class DeliveryCounts(StrictModel):
    total: int
    conflict: int
    aligned: int
    ambiguous: int
    protocols: dict[str, int]
    use_in_main: dict[str, int]
    quality_flags: dict[str, int]


class AnnotationPolicy(StrictModel):
    policy_id: str
    label_source: str
    accepted_for_current_experiments: bool
    annotation_requirement_waived: bool
    observed_annotation_count: int
    observed_annotator_agreement: float
    preserve_delivered_use_in_main: bool
    pending_statistics: list[str]


class MediaPolicy(StrictModel):
    media_in_archive: bool
    subtitle_crop_dataset: str
    subtitle_bottom_crop_fraction: float
    cropped_media_prefix: str
    variety_discard_report: str
    variety_discarded_count: int
    variety_suspect_flag: str
    variety_suspect_count: int


class SourceBoundary(StrictModel):
    source_is_generated: bool
    sources: list[str]
    rows: int
    sample_types: dict[str, int]
    use_in_main: dict[str, int]


class DerivationPolicy(StrictModel):
    builder: str
    split_key: Literal["split_group_id"]
    split_algorithm: Literal["sha256_first_8_hex_mod_100"]
    split_thresholds: dict[str, list[int]]
    protocol_outputs: dict[str, str]


class DeliveryProvenance(StrictModel):
    schema_name: Literal["mprisk_delivery_provenance_v1"]
    delivery_id: str
    source_archive: str
    archive_sha256: str
    archive_member_count: int
    artifacts: dict[str, ArtifactProvenance]
    counts: DeliveryCounts
    annotation_policy: AnnotationPolicy
    media_policy: MediaPolicy
    source_boundaries: dict[str, SourceBoundary]
    derivation_policy: DerivationPolicy

    @field_validator("archive_sha256")
    @classmethod
    def archive_sha256_must_be_hex(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("archive_sha256 must contain 64 hexadecimal characters")
        int(value, 16)
        return value


@dataclass(frozen=True)
class DeliveryValidationReport:
    total_rows: int
    sample_type_counts: dict[str, int]
    protocol_counts: dict[str, int]
    use_in_main_counts: dict[str, int]
    source_counts: dict[str, int]
    unique_sample_ids: int
    unique_split_groups: int
    unique_media_paths: int


@dataclass(frozen=True)
class BuildManifestsResult:
    validation: DeliveryValidationReport
    output_paths: dict[str, Path]


def load_delivery_provenance(
    repo_root: str | Path,
    provenance_path: str | Path = DEFAULT_PROVENANCE_PATH,
) -> DeliveryProvenance:
    root = Path(repo_root).resolve()
    path = _resolve_repo_path(root, provenance_path)
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return DeliveryProvenance.model_validate(raw)


def verify_source_archive(provenance: DeliveryProvenance) -> int:
    archive = Path(provenance.source_archive)
    if not archive.is_file():
        raise FileNotFoundError(f"Delivery archive does not exist: {archive}")
    actual_sha256 = _sha256(archive)
    if actual_sha256 != provenance.archive_sha256:
        raise ValueError(
            f"Delivery archive sha256 mismatch: expected {provenance.archive_sha256}, "
            f"got {actual_sha256}"
        )

    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    unsafe: list[str] = []
    for member in members:
        name = member.name.replace("\\", "/")
        if name.startswith("/") or ".." in PurePosixPath(name).parts:
            unsafe.append(member.name)
        if not (member.isdir() or member.isfile()):
            unsafe.append(member.name)
    if unsafe:
        raise ValueError("Unsafe delivery archive members: " + ", ".join(sorted(set(unsafe))))
    if len(members) != provenance.archive_member_count:
        raise ValueError(
            "Delivery archive member count mismatch: "
            f"expected {provenance.archive_member_count}, got {len(members)}"
        )
    return len(members)


def validate_media_paths(rows: list[dict[str, Any]]) -> int:
    paths: set[Path] = set()
    for row in rows:
        media_paths = row.get("media_paths")
        if not isinstance(media_paths, dict) or not media_paths:
            raise ValueError(f"{row.get('sample_id')}: media_paths must be a non-empty object")
        for modality, value in media_paths.items():
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{row.get('sample_id')}: media path for {modality} must be non-empty"
                )
            path = Path(value)
            if not path.is_absolute():
                raise ValueError(f"{row.get('sample_id')}: media path must be absolute: {path}")
            paths.add(path)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} delivery media files are missing: {preview}")
    empty = sorted(str(path) for path in paths if path.stat().st_size == 0)
    if empty:
        preview = ", ".join(empty[:10])
        raise ValueError(f"{len(empty)} delivery media files are empty: {preview}")
    return len(paths)


def validate_delivery(
    repo_root: str | Path,
    *,
    provenance_path: str | Path = DEFAULT_PROVENANCE_PATH,
    check_media: bool,
    verify_archive: bool,
) -> DeliveryValidationReport:
    root = Path(repo_root).resolve()
    provenance = load_delivery_provenance(root, provenance_path)
    _validate_artifact_digests(root, provenance)
    _validate_derivation_policy(provenance.derivation_policy)
    if verify_archive:
        verify_source_archive(provenance)

    rows_by_kind = {
        kind: _load_artifact_rows(root, provenance, artifact_key)
        for kind, artifact_key in MANIFEST_ARTIFACTS.items()
    }
    unified = rows_by_kind["unified"]
    validated = _validate_final_rows(unified)

    sample_ids = [row.sample_id for row in validated]
    duplicates = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
    if duplicates:
        raise ValueError("Duplicate sample_id values: " + ", ".join(duplicates[:10]))

    _validate_class_manifests(rows_by_kind)
    sample_type_counts = {
        sample_type: sum(row.sample_type == sample_type for row in validated)
        for sample_type in SAMPLE_TYPES
    }
    expected_sample_types = {
        "Conflict": provenance.counts.conflict,
        "Aligned": provenance.counts.aligned,
        "Ambiguous": provenance.counts.ambiguous,
    }
    if len(validated) != provenance.counts.total:
        raise ValueError(
            f"Total row count mismatch: expected {provenance.counts.total}, got {len(validated)}"
        )
    if sample_type_counts != expected_sample_types:
        raise ValueError(
            f"Sample type counts mismatch: expected {expected_sample_types}, "
            f"got {sample_type_counts}"
        )

    protocol_counts = dict(Counter(row.protocol for row in validated))
    if protocol_counts != provenance.counts.protocols:
        raise ValueError(
            f"Protocol counts mismatch: expected {provenance.counts.protocols}, "
            f"got {protocol_counts}"
        )
    use_in_main_counts = dict(Counter(row.sample_type for row in validated if row.use_in_main))
    if use_in_main_counts != provenance.counts.use_in_main:
        raise ValueError(
            f"use_in_main counts mismatch: expected {provenance.counts.use_in_main}, "
            f"got {use_in_main_counts}"
        )

    _validate_annotation_policy(unified, provenance.annotation_policy)
    source_counts = _validate_source_boundaries(unified, provenance.source_boundaries)
    _validate_media_policy(root, unified, provenance)
    unique_media_paths = (
        validate_media_paths(unified) if check_media else _count_media_paths(unified)
    )

    return DeliveryValidationReport(
        total_rows=len(validated),
        sample_type_counts=sample_type_counts,
        protocol_counts=protocol_counts,
        use_in_main_counts=use_in_main_counts,
        source_counts=source_counts,
        unique_sample_ids=len(set(sample_ids)),
        unique_split_groups=len({row.split_group_id for row in validated}),
        unique_media_paths=unique_media_paths,
    )


def build_derived_manifests(
    repo_root: str | Path,
    *,
    provenance_path: str | Path = DEFAULT_PROVENANCE_PATH,
    check_media: bool,
    verify_archive: bool,
) -> BuildManifestsResult:
    root = Path(repo_root).resolve()
    validation = validate_delivery(
        root,
        provenance_path=provenance_path,
        check_media=check_media,
        verify_archive=verify_archive,
    )
    provenance = load_delivery_provenance(root, provenance_path)
    unified = _load_artifact_rows(root, provenance, MANIFEST_ARTIFACTS["unified"])
    validated = _validate_final_rows(unified)

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    protocol_rows: dict[str, list[dict[str, Any]]] = {
        protocol: [] for protocol in PROTOCOL_OUTPUTS
    }
    group_splits: dict[str, str] = {}
    for raw, row in zip(unified, validated, strict=True):
        split = assign_split(row.split_group_id)
        existing = group_splits.setdefault(row.split_group_id, split)
        if existing != split:
            raise ValueError(f"split_group_id crosses splits: {row.split_group_id}")
        if row.protocol not in protocol_rows:
            raise ValueError(f"Protocol has no output manifest: {row.protocol}")
        derived = dict(raw)
        derived["split"] = split
        split_rows[split].append(derived)
        protocol_rows[row.protocol].append(derived)

    manifest_root = root / "data/processed/manifests"
    output_paths: dict[str, Path] = {}
    for split, rows in split_rows.items():
        path = manifest_root / "splits" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, rows)
        output_paths[f"split:{split}"] = path
    for protocol, rows in protocol_rows.items():
        path = manifest_root / PROTOCOL_OUTPUTS[protocol]
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, rows)
        output_paths[f"protocol:{protocol}"] = path

    _validate_derived_outputs(
        unified,
        split_rows=split_rows,
        protocol_rows=protocol_rows,
        expected_protocol_counts=provenance.counts.protocols,
    )
    return BuildManifestsResult(validation=validation, output_paths=output_paths)


def _validate_artifact_digests(root: Path, provenance: DeliveryProvenance) -> None:
    for name, artifact in provenance.artifacts.items():
        path = _resolve_repo_path(root, artifact.path)
        if not path.is_file():
            raise FileNotFoundError(f"Delivery artifact does not exist: {name}: {path}")
        size = path.stat().st_size
        if size != artifact.bytes:
            raise ValueError(
                f"Delivery artifact size mismatch for {name}: "
                f"expected {artifact.bytes}, got {size}"
            )
        actual_sha256 = _sha256(path)
        if actual_sha256 != artifact.sha256:
            raise ValueError(
                f"Delivery artifact sha256 mismatch for {name}: "
                f"expected {artifact.sha256}, got {actual_sha256}"
            )


def _validate_derivation_policy(policy: DerivationPolicy) -> None:
    expected_thresholds = {
        "train": [0, 70],
        "val": [70, 85],
        "test": [85, 100],
    }
    expected_outputs = {
        protocol: f"data/processed/manifests/{relative_path}"
        for protocol, relative_path in PROTOCOL_OUTPUTS.items()
    }
    if policy.builder != "scripts/build_manifests.py":
        raise ValueError("Delivery derivation builder does not match the implemented entry point")
    if policy.split_thresholds != expected_thresholds:
        raise ValueError("Delivery split thresholds do not match assign_split")
    if policy.protocol_outputs != expected_outputs:
        raise ValueError("Delivery protocol outputs do not match the implemented outputs")


def _load_artifact_rows(
    root: Path,
    provenance: DeliveryProvenance,
    artifact_key: str,
) -> list[dict[str, Any]]:
    try:
        artifact = provenance.artifacts[artifact_key]
    except KeyError as exc:
        raise ValueError(f"Missing provenance artifact: {artifact_key}") from exc
    return read_jsonl(_resolve_repo_path(root, artifact.path))


def _validate_final_rows(rows: list[dict[str, Any]]) -> list[FinalManifestRow]:
    validated: list[FinalManifestRow] = []
    for line_number, row in enumerate(rows, start=1):
        try:
            validated.append(FinalManifestRow.model_validate(row))
        except ValueError as exc:
            raise ValueError(f"unified manifest line {line_number}: {exc}") from exc
    return validated


def _validate_class_manifests(rows_by_kind: dict[str, list[dict[str, Any]]]) -> None:
    unified_counter = Counter(_canonical(row) for row in rows_by_kind["unified"])
    partition_rows: list[dict[str, Any]] = []
    for kind, sample_type in (
        ("conflict", "Conflict"),
        ("aligned", "Aligned"),
        ("ambiguous", "Ambiguous"),
    ):
        rows = rows_by_kind[kind]
        wrong = [row.get("sample_id") for row in rows if row.get("sample_type") != sample_type]
        if wrong:
            raise ValueError(f"{kind} manifest contains wrong sample_type rows: {wrong[:10]}")
        partition_rows.extend(rows)
    if unified_counter != Counter(_canonical(row) for row in partition_rows):
        raise ValueError("Conflict/Aligned/Ambiguous manifests are not an exact unified partition")


def _validate_annotation_policy(
    rows: list[dict[str, Any]],
    policy: AnnotationPolicy,
) -> None:
    if not (
        policy.accepted_for_current_experiments
        and policy.annotation_requirement_waived
        and policy.preserve_delivered_use_in_main
    ):
        raise ValueError("The delivery annotation waiver must preserve current inclusion labels")
    pending = {"multi_annotator_mean", "multi_annotator_standard_deviation"}
    if set(policy.pending_statistics) != pending:
        raise ValueError(f"Pending annotation statistics must be {sorted(pending)}")
    if any(row.get("annotation_count") != policy.observed_annotation_count for row in rows):
        raise ValueError("Manifest annotation_count does not match the delivery waiver")
    if any(
        row.get("annotator_agreement") != policy.observed_annotator_agreement for row in rows
    ):
        raise ValueError("Manifest annotator_agreement does not match the delivery waiver")


def _validate_source_boundaries(
    rows: list[dict[str, Any]],
    boundaries: dict[str, SourceBoundary],
) -> dict[str, int]:
    membership: defaultdict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = {}
    for name, boundary in boundaries.items():
        selected = [
            row
            for row in rows
            if row.get("source_is_generated") is boundary.source_is_generated
            and row.get("source_dataset") in boundary.sources
        ]
        source_counts[name] = len(selected)
        for row in selected:
            membership[str(row.get("sample_id"))] += 1
        actual_types = dict(Counter(str(row.get("sample_type")) for row in selected))
        actual_main = dict(
            Counter(str(row.get("sample_type")) for row in selected if row.get("use_in_main"))
        )
        if len(selected) != boundary.rows:
            raise ValueError(
                f"Source boundary {name} row mismatch: "
                f"expected {boundary.rows}, got {len(selected)}"
            )
        if actual_types != boundary.sample_types or actual_main != boundary.use_in_main:
            raise ValueError(f"Source boundary {name} sample counts do not match provenance")
    missing = [
        str(row.get("sample_id"))
        for row in rows
        if membership[str(row.get("sample_id"))] != 1
    ]
    if missing:
        raise ValueError("Every delivery row must belong to exactly one source boundary")
    return source_counts


def _validate_media_policy(
    root: Path,
    rows: list[dict[str, Any]],
    provenance: DeliveryProvenance,
) -> None:
    policy = provenance.media_policy
    if policy.media_in_archive:
        raise ValueError("delivery_20260714 must not claim that media is embedded in the archive")
    cropped_prefix = policy.cropped_media_prefix.rstrip("/") + "/"
    uncropped = [
        row.get("sample_id")
        for row in rows
        if row.get("source_dataset") == policy.subtitle_crop_dataset
        and not str((row.get("media_paths") or {}).get("vision", "")).startswith(cropped_prefix)
    ]
    if uncropped:
        raise ValueError(f"Subtitle-cropped media path policy failed: {uncropped[:10]}")

    flag_count = sum(
        policy.variety_suspect_flag in (row.get("quality_flags") or []) for row in rows
    )
    if flag_count != policy.variety_suspect_count:
        raise ValueError(
            f"Variety suspect count mismatch: expected {policy.variety_suspect_count}, "
            f"got {flag_count}"
        )
    if provenance.counts.quality_flags.get(policy.variety_suspect_flag) != flag_count:
        raise ValueError("Provenance quality flag counts do not match the media policy")

    report_path = _resolve_repo_path(root, policy.variety_discard_report)
    with report_path.open(encoding="utf-8") as handle:
        discarded = json.load(handle).get("discarded_sample_ids")
    if not isinstance(discarded, list):
        raise ValueError("variety_discarded.json must contain discarded_sample_ids")
    if len(discarded) != policy.variety_discarded_count or len(set(discarded)) != len(discarded):
        raise ValueError("Variety discard count or uniqueness does not match provenance")
    included_ids = {str(row.get("sample_id")) for row in rows}
    leaked = sorted(included_ids.intersection(discarded))
    if leaked:
        raise ValueError(f"Variety-discarded rows leaked into final manifests: {leaked[:10]}")


def _validate_derived_outputs(
    unified: list[dict[str, Any]],
    *,
    split_rows: dict[str, list[dict[str, Any]]],
    protocol_rows: dict[str, list[dict[str, Any]]],
    expected_protocol_counts: dict[str, int],
) -> None:
    unified_ids = {str(row["sample_id"]) for row in unified}
    split_ids = [str(row["sample_id"]) for rows in split_rows.values() for row in rows]
    protocol_ids = [str(row["sample_id"]) for rows in protocol_rows.values() for row in rows]
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != unified_ids:
        raise ValueError("Split manifests must be an exact, mutually exclusive unified partition")
    if len(protocol_ids) != len(set(protocol_ids)) or set(protocol_ids) != unified_ids:
        raise ValueError(
            "Protocol manifests must be an exact, mutually exclusive unified partition"
        )
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for split, rows in split_rows.items():
        for row in rows:
            if row.get("split") != split:
                raise ValueError(f"Derived split field mismatch for {row.get('sample_id')}")
            group_splits[str(row["split_group_id"])].add(split)
    leaked_groups = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if leaked_groups:
        raise ValueError(f"split_group_id leakage: {leaked_groups[:10]}")
    actual_protocol_counts = {key: len(value) for key, value in protocol_rows.items()}
    if actual_protocol_counts != expected_protocol_counts:
        raise ValueError(
            f"Derived protocol counts mismatch: expected {expected_protocol_counts}, "
            f"got {actual_protocol_counts}"
        )


def _count_media_paths(rows: list[dict[str, Any]]) -> int:
    return len(
        {
            str(path)
            for row in rows
            for path in (row.get("media_paths") or {}).values()
        }
    )


def _resolve_repo_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError(f"Repository artifact path must be relative: {candidate}")
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Repository artifact path escapes the repository: {candidate}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
