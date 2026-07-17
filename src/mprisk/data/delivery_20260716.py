"""Strict read-only ingestion for the archived 2026-07-16 generated delivery."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mprisk.data.manifests import FinalManifestRow
from mprisk.data.representation_splits import (
    build_representation_split_assignment,
    load_representation_split_assignment,
)
from mprisk.data.splits import assign_split

DEFAULT_SOURCE_ROOT = Path("/home/team/lvshuyang/prompt-make/delivery_20260716")
DEFAULT_OUTPUT_ROOT = Path("outputs/datasets/delivery_20260716")
SOURCE_DATASET = "delivery_20260716"
DEFAULT_FFPROBE_PATH = Path("ffprobe")
EXPECTED_FIELDS = frozenset(
    {
        "sample_id",
        "source_id",
        "protocol",
        "sample_type",
        "media_paths",
        "text_content",
        "gt_emotion",
        "surface_emotion",
        "gt_describe",
        "rationale",
        "generation_info",
        "source_is_generated",
    }
)


@dataclass(frozen=True)
class SourceSpec:
    filename: str
    protocol: str
    sample_type: str
    rows: int
    sha256: str


SOURCE_SPECS = (
    SourceSpec(
        "vt_a_manifest.jsonl",
        "VT",
        "Conflict",
        732,
        "6c32b34a569fac50c75c50f4a495c07e92e00e8aabe7af42cb87a89ef5323507",
    ),
    SourceSpec(
        "vt_c_manifest.jsonl",
        "VT",
        "Aligned",
        1144,
        "cde71b9dfa4e40e93d8987fe379684e011bbdd68256089dbd3da99b0d27c1f9b",
    ),
    SourceSpec(
        "va_a_manifest.jsonl",
        "VA",
        "Conflict",
        846,
        "ad356b4570619ce860eb6d2facd9a57d8f21cb3c46cd55b2b951c0e5e85e3be9",
    ),
    SourceSpec(
        "va_c_manifest.jsonl",
        "VA",
        "Aligned",
        1093,
        "de8a7964f1ad3b9a5eeee632bdbebe1105da0b5e92ad025a0f9969b3d48559a8",
    ),
)


@dataclass(frozen=True)
class InvalidVaAssetSpec:
    sample_id: str
    source_id: str
    sha256: str
    reason: str = "missing_audio_stream"


EXPECTED_INVALID_VA_ASSETS = (
    InvalidVaAssetSpec(
        "gen:accept_a_va:S0544",
        "v6_A2_tears_of_joy_SVT_F_13001",
        "fef8fddc51cfd7ad32ad496a06196cf4a811b6ea1b3df1377127a4109c113ae3",
    ),
    InvalidVaAssetSpec(
        "gen:accept_a_va:S0545",
        "v6_A3_suppressed_anger_SVT_M_13002",
        "b3ccc970867d084ca8e4c0157a201d24af91fb43f83a653eb4c0cf1346f84af4",
    ),
    InvalidVaAssetSpec(
        "gen:accept_a_va:S0546",
        "v6_A4_composure_shock_SVT_F_13001",
        "7c7367dadd1237cec76c54e959f18ec8deec77bad06074633401e14c4029d5b5",
    ),
    InvalidVaAssetSpec(
        "gen:accept_a_va:S0547",
        "v6_A5_composure_fear_SVT_M_13001",
        "d91407f588f55a86db777fff315942e9bab9f7881033daa38b6f8ea4088a6b44",
    ),
    InvalidVaAssetSpec(
        "gen:accept_a_va:S0548",
        "v6_A5_composure_fear_SVT_F_13001",
        "a13855a28ed856797fe0146816e0581abbd6a0164c53f14e7f354ad1ed08a8f7",
    ),
)


@dataclass(frozen=True)
class DeliveryIngestionResult:
    output_root: Path
    provenance_path: Path
    representation_split_path: Path
    total_rows: int
    unique_split_groups: int
    state_valid_rows: int
    invalid_asset_rows: int


def ingest_delivery_20260716(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    ffprobe_path: str | Path = DEFAULT_FFPROBE_PATH,
    ffprobe_workers: int = 16,
) -> DeliveryIngestionResult:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Delivery source directory does not exist: {source}")
    if output == source or source in output.parents:
        raise ValueError("Derived output root must not be inside the read-only delivery source")

    raw_rows: list[tuple[SourceSpec, dict[str, Any]]] = []
    source_artifacts: dict[str, dict[str, Any]] = {}
    source_hashes_before: dict[str, str] = {}
    for spec in SOURCE_SPECS:
        path = source / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing delivery artifact: {path}")
        digest = _sha256(path)
        if digest != spec.sha256:
            raise ValueError(
                f"Delivery artifact SHA-256 mismatch for {spec.filename}: "
                f"expected {spec.sha256}, got {digest}"
            )
        rows = _read_strict_jsonl(path)
        if len(rows) != spec.rows:
            raise ValueError(
                f"Delivery row count mismatch for {spec.filename}: "
                f"expected {spec.rows}, got {len(rows)}"
            )
        for line_number, row in enumerate(rows, start=1):
            _validate_raw_row(row, spec=spec, path=path, line_number=line_number)
            raw_rows.append((spec, row))
        source_hashes_before[spec.filename] = digest
        source_artifacts[spec.filename] = {
            "bytes": path.stat().st_size,
            "rows": len(rows),
            "sha256": digest,
            "protocol": spec.protocol,
            "sample_type": spec.sample_type,
        }

    sample_ids = [str(row["sample_id"]) for _spec, row in raw_rows]
    duplicates = [key for key, count in Counter(sample_ids).items() if count != 1]
    if duplicates:
        raise ValueError(f"Delivery sample IDs must be globally unique: {sorted(duplicates)[:10]}")

    source_id_classes: defaultdict[str, set[str]] = defaultdict(set)
    for spec, row in raw_rows:
        source_id_classes[str(row["source_id"])].add(spec.sample_type)
    conflicting_groups = sorted(
        source_id for source_id, labels in source_id_classes.items() if len(labels) != 1
    )
    if conflicting_groups:
        raise ValueError(
            "A shared source_id cannot cross Conflict/Aligned labels: "
            f"{conflicting_groups[:10]}"
        )

    normalized = [_normalize_row(row, spec) for spec, row in raw_rows]
    validated = [FinalManifestRow.model_validate(row) for row in normalized]
    if len(validated) != sum(spec.rows for spec in SOURCE_SPECS):
        raise AssertionError("Normalized row count changed unexpectedly")
    _validate_group_splits(normalized)

    va_state_valid, invalid_assets = _audit_va_assets(
        [row for row in normalized if row["protocol"] == "VA"],
        ffprobe_path=Path(ffprobe_path),
        workers=ffprobe_workers,
    )
    vt_state_valid = [row for row in normalized if row["protocol"] == "VT"]
    state_valid = vt_state_valid + va_state_valid

    manifests = output / "manifests"
    manifest_rows = {
        "unified_sample_manifest.jsonl": normalized,
        "conflict_manifest.jsonl": [r for r in normalized if r["sample_type"] == "Conflict"],
        "aligned_manifest.jsonl": [r for r in normalized if r["sample_type"] == "Aligned"],
        "vt_primary.jsonl": [r for r in normalized if r["protocol"] == "VT"],
        "va_aux.jsonl": [r for r in normalized if r["protocol"] == "VA"],
        "va_state_valid.jsonl": va_state_valid,
        "invalid_assets.jsonl": invalid_assets,
    }
    for filename, rows in manifest_rows.items():
        _atomic_jsonl(manifests / filename, rows)
    for split in ("train", "val", "test"):
        _atomic_jsonl(
            manifests / "master_splits" / f"{split}.jsonl",
            [row for row in normalized if row["split"] == split],
        )

    split_root = output / "splits"
    split_config = split_root / "representation_split_config_v1.yaml"
    _atomic_text(split_config, yaml.safe_dump(_split_config(), sort_keys=False))
    split_result = build_representation_split_assignment(
        config_path=split_config,
        output_dir=split_root,
    )
    assignments = load_representation_split_assignment(split_result.manifest_path)
    expected_groups = {str(row["split_group_id"]) for row in state_valid}
    if set(assignments) != expected_groups:
        raise ValueError("Representation split groups do not match normalized delivery groups")
    assigned_samples = {
        str(sample_id)
        for assignment in assignments.values()
        for sample_id in assignment["sample_ids"]
    }
    state_valid_sample_ids = {str(row["sample_id"]) for row in state_valid}
    if assigned_samples != state_valid_sample_ids:
        raise ValueError("Representation split sample IDs do not match state-valid delivery rows")

    cache_plan_path = output / "state_cache_plan_v1.yaml"
    _atomic_text(
        cache_plan_path,
        yaml.safe_dump(
            _state_cache_plan(
                invalid_assets_sha256=_sha256(manifests / "invalid_assets.jsonl"),
                vt_rows=len(vt_state_valid),
                va_rows=len(va_state_valid),
            ),
            sort_keys=False,
        ),
    )

    source_hashes_after = {
        spec.filename: _sha256(source / spec.filename) for spec in SOURCE_SPECS
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("Read-only delivery source changed during ingestion")

    derived_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "provenance.json"
    )
    provenance_path = output / "provenance.json"
    _atomic_json(
        provenance_path,
        {
            "schema": "mprisk_delivery_20260716_ingestion_v2",
            "source_root": str(source),
            "source_read_only": True,
            "source_artifacts": source_artifacts,
            "normalization": {
                "source_dataset": SOURCE_DATASET,
                "class_mapping": {
                    "*_a_manifest.jsonl": "Conflict",
                    "*_c_manifest.jsonl": "Aligned",
                },
                "protocol_mapping": {"vt_*.jsonl": "VT", "va_*.jsonl": "VA"},
                "split_group_rule": "delivery_20260716:{source_id}",
                "master_split_rule": "sha256_first_8_hex_mod_100",
                "master_split_thresholds": {"train": [0, 70], "val": [70, 85], "test": [85, 100]},
            },
            "counts": {
                "total": len(normalized),
                "sample_types": dict(sorted(Counter(r["sample_type"] for r in normalized).items())),
                "protocols": dict(sorted(Counter(r["protocol"] for r in normalized).items())),
                "master_splits": dict(sorted(Counter(r["split"] for r in normalized).items())),
                "unique_sample_ids": len(set(sample_ids)),
                "unique_split_groups": len(expected_groups),
                "unique_media_paths": len(
                    {path for row in normalized for path in row["media_paths"].values()}
                ),
                "state_valid": len(state_valid),
                "va_state_valid": len(va_state_valid),
                "invalid_assets": len(invalid_assets),
            },
            "asset_qa": {
                "ffprobe_path": str(ffprobe_path),
                "policy": "VA requires at least one video and one audio stream",
                "fail_closed_expected_invalid_set": True,
                "invalid_asset_sample_ids": [row["sample_id"] for row in invalid_assets],
                "invalid_assets_manifest": "manifests/invalid_assets.jsonl",
                "invalid_assets_sha256": _sha256(manifests / "invalid_assets.jsonl"),
                "state_valid_manifest": "manifests/va_state_valid.jsonl",
                "state_valid_manifest_sha256": _sha256(manifests / "va_state_valid.jsonl"),
            },
            "expected_prefill_tasks_p8_m1_m2_m12": {"VT_per_model": 45024, "VA_per_model": 46416},
            "derived_artifacts": {
                str(path.relative_to(output)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in derived_paths
            },
        },
    )
    return DeliveryIngestionResult(
        output_root=output,
        provenance_path=provenance_path,
        representation_split_path=split_result.manifest_path,
        total_rows=len(normalized),
        unique_split_groups=len(expected_groups),
        state_valid_rows=len(state_valid),
        invalid_asset_rows=len(invalid_assets),
    )


def _validate_raw_row(
    row: dict[str, Any], *, spec: SourceSpec, path: Path, line_number: int
) -> None:
    if set(row) != EXPECTED_FIELDS:
        missing = sorted(EXPECTED_FIELDS - set(row))
        extra = sorted(set(row) - EXPECTED_FIELDS)
        raise ValueError(f"{path}:{line_number}: field mismatch; missing={missing}, extra={extra}")
    for field in ("sample_id", "source_id", "text_content", "gt_emotion", "gt_describe"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"{path}:{line_number}: {field} must be non-empty text")
    if row["protocol"] != spec.protocol or row["sample_type"] != spec.sample_type:
        raise ValueError(
            f"{path}:{line_number}: file role requires {spec.protocol}/{spec.sample_type}"
        )
    if row["source_is_generated"] is not True:
        raise ValueError(f"{path}:{line_number}: source_is_generated must be true")
    if not isinstance(row["generation_info"], dict):
        raise ValueError(f"{path}:{line_number}: generation_info must be an object")
    surface = row["surface_emotion"]
    if spec.sample_type == "Conflict" and (not isinstance(surface, str) or not surface.strip()):
        raise ValueError(f"{path}:{line_number}: Conflict surface_emotion must be non-empty")
    if spec.sample_type == "Aligned" and surface is not None:
        raise ValueError(f"{path}:{line_number}: Aligned surface_emotion must be null")

    media = row["media_paths"]
    expected_keys = {"vision"} if spec.protocol == "VT" else {"vision", "audio"}
    if not isinstance(media, dict) or set(media) != expected_keys:
        raise ValueError(f"{path}:{line_number}: media keys must be {sorted(expected_keys)}")
    if spec.protocol == "VA" and media["vision"] != media["audio"]:
        raise ValueError(f"{path}:{line_number}: VA vision/audio must reference the same MP4")
    for media_path in media.values():
        candidate = Path(str(media_path))
        if not candidate.is_absolute() or not candidate.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: missing absolute media {candidate}")
        if candidate.stat().st_size == 0:
            raise ValueError(f"{path}:{line_number}: media is empty: {candidate}")


def _normalize_row(row: dict[str, Any], spec: SourceSpec) -> dict[str, Any]:
    gt_emotion = str(row["gt_emotion"])
    visual_emotion = (
        str(row["surface_emotion"]) if spec.sample_type == "Conflict" else gt_emotion
    )
    second_modality = "text" if spec.protocol == "VT" else "audio"
    split_group_id = f"{SOURCE_DATASET}:{row['source_id']}"
    normalized = dict(row)
    normalized.update(
        {
            "source_dataset": SOURCE_DATASET,
            "split_group_id": split_group_id,
            "split": assign_split(split_group_id),
            "views": {
                "M1": {
                    "modality": "vision",
                    "label": visual_emotion,
                    "specific_affect": visual_emotion,
                    "is_clear": True,
                },
                "M2": {
                    "modality": second_modality,
                    "label": gt_emotion,
                    "specific_affect": gt_emotion,
                    "is_clear": True,
                },
                "M12": {
                    "modality": f"vision+{second_modality}",
                    "label": gt_emotion,
                    "specific_affect": gt_emotion,
                    "is_clear": True,
                },
            },
            "use_in_main": True,
            "annotation_count": 1,
            "quality_flags": [],
            "label_basis": "delivery_20260716_archived_manual_review",
            "source_delivery_file": spec.filename,
            "source_delivery_sha256": spec.sha256,
        }
    )
    return normalized


def _validate_group_splits(rows: list[dict[str, Any]]) -> None:
    by_group: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        expected = assign_split(str(row["split_group_id"]))
        if row["split"] != expected:
            raise ValueError(f"Master split mismatch for {row['sample_id']}")
        by_group[str(row["split_group_id"])].add(str(row["split"]))
    leaked = sorted(group for group, splits in by_group.items() if len(splits) != 1)
    if leaked:
        raise ValueError(f"split_group_id leakage across master splits: {leaked[:10]}")


def _audit_va_assets(
    rows: list[dict[str, Any]], *, ffprobe_path: Path, workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if workers <= 0:
        raise ValueError("ffprobe_workers must be positive")
    media_paths = sorted({str(row["media_paths"]["audio"]) for row in rows})

    def probe(media_path: str) -> tuple[str, list[dict[str, Any]]]:
        completed = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type,codec_name,channels,sample_rate",
                "-of",
                "json",
                media_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        streams = payload.get("streams")
        if not isinstance(streams, list) or not all(isinstance(item, dict) for item in streams):
            raise ValueError(f"ffprobe returned an invalid stream list for {media_path}")
        return media_path, streams

    with ThreadPoolExecutor(max_workers=workers) as pool:
        probes = dict(pool.map(probe, media_paths))

    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for row in rows:
        media_path = str(row["media_paths"]["audio"])
        streams = probes[media_path]
        stream_types = [str(item.get("codec_type")) for item in streams]
        if "video" not in stream_types:
            raise ValueError(f"VA asset has no video stream: {row['sample_id']} {media_path}")
        if "audio" in stream_types:
            valid_rows.append(row)
            continue
        asset = Path(media_path)
        invalid_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "source_id": str(row["source_id"]),
                "protocol": "VA",
                "sample_type": str(row["sample_type"]),
                "media_path": media_path,
                "bytes": asset.stat().st_size,
                "sha256": _sha256(asset),
                "reason": "missing_audio_stream",
                "stream_types": stream_types,
                "state_eligible": False,
            }
        )

    observed = {
        (row["sample_id"], row["source_id"], row["sha256"], row["reason"])
        for row in invalid_rows
    }
    expected = {
        (spec.sample_id, spec.source_id, spec.sha256, spec.reason)
        for spec in EXPECTED_INVALID_VA_ASSETS
    }
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "VA invalid-asset set changed; explicit review is required before state/cache use: "
            f"missing={missing}, unexpected={unexpected}"
        )
    invalid_rows.sort(key=lambda row: row["sample_id"])
    return valid_rows, invalid_rows


def _state_cache_plan(
    *, invalid_assets_sha256: str, vt_rows: int, va_rows: int
) -> dict[str, Any]:
    return {
        "schema": "mprisk_delivery_state_cache_plan_v1",
        "source_dataset": SOURCE_DATASET,
        "prompt_count": 8,
        "conditions": ["M1", "M2", "M12"],
        "manifests": {
            "VT": "manifests/vt_primary.jsonl",
            "VA": "manifests/va_state_valid.jsonl",
        },
        "rows": {"VT": vt_rows, "VA": va_rows},
        "expected_tasks": {"VT_per_model": vt_rows * 8 * 3, "VA_per_model": va_rows * 8 * 3},
        "asset_qa": {
            "invalid_assets_manifest": "manifests/invalid_assets.jsonl",
            "invalid_assets_sha256": invalid_assets_sha256,
            "fail_closed_on_invalid_set_change": True,
        },
    }


def _split_config() -> dict[str, Any]:
    return {
        "schema": "mprisk_representation_split_config_v1",
        "key": "delivery_20260716_representation_split_v1",
        "scope": "all_valid_conflict_aligned",
        "source_manifests": [
            "../manifests/vt_primary.jsonl",
            "../manifests/va_state_valid.jsonl",
        ],
        "seed": 20260716,
        "calibration_fraction": 0.5,
        "calibration_rounding": "floor",
        "ranking_rule": "sha256(seed:split_group_id)",
        "master_split_field": "split",
        "split_group_field": "split_group_id",
        "use_in_main_only": False,
        "calibration_master_split": "val",
        "calibration_eligible_sample_type": "Aligned",
        "minimum_eligible_groups": 2,
        "assignments": {
            "train": "relation_train",
            "validation": "relation_val",
            "calibration": "aligned_calibration",
            "test": "official_test",
        },
    }


def _read_strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(value)
    return rows


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
