"""Versioned group-level representation split assignments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mprisk.utils.io import write_json

CONFIG_SCHEMA = "mprisk_representation_split_config_v1"
ASSIGNMENT_SCHEMA = "mprisk_representation_split_assignment_v1"
MASTER_SPLITS = frozenset({"train", "val", "test"})
REPRESENTATION_SPLITS = frozenset(
    {"relation_train", "relation_val", "aligned_calibration", "official_test"}
)


@dataclass(frozen=True)
class RepresentationSplitBuildResult:
    manifest_path: Path
    summary_path: Path
    group_count: int
    sample_count: int


def build_representation_split_assignment(
    *, config_path: str | Path, output_dir: str | Path
) -> RepresentationSplitBuildResult:
    config_file = Path(config_path)
    config = _load_config(config_file)
    rows, sources = _load_source_rows(config_file, config)
    groups = _group_rows(rows, config)
    assignment = _assign_groups(groups, config)
    manifest_rows = _manifest_rows(groups, assignment, config)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "representation_split_assignment_v1.jsonl"
    _atomic_jsonl(manifest_path, manifest_rows)
    manifest_sha256 = _sha256(manifest_path)
    assignment_checksum = hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    group_counts = Counter(assignment.values())
    sample_counts = Counter(
        split for group, split in assignment.items() for _row in groups[group]
    )
    label_counts = {
        split: dict(
            sorted(
                Counter(
                    str(row["sample_type"])
                    for group, group_split in assignment.items()
                    if group_split == split
                    for row in groups[group]
                ).items()
            )
        )
        for split in sorted(REPRESENTATION_SPLITS)
    }
    summary_path = write_json(
        output_root / "representation_split_summary_v1.json",
        {
            "schema": "mprisk_representation_split_summary_v1",
            "config_key": config["key"],
            "config_path": str(config_file),
            "config_sha256": _sha256(config_file),
            "sources": sources,
            "seed": config["seed"],
            "ranking_rule": config["ranking_rule"],
            "scope": config["scope"],
            "calibration_fraction": config["calibration_fraction"],
            "calibration_rounding": config["calibration_rounding"],
            "use_in_main_only": config["use_in_main_only"],
            "legacy_use_in_main_counts": dict(
                sorted(Counter(str(bool(row.get("use_in_main"))).lower() for row in rows).items())
            ),
            "selection_policy": (
                "official val groups containing only Aligned samples; fixed seeded hash rank"
            ),
            "group_counts": dict(sorted(group_counts.items())),
            "sample_counts": dict(sorted(sample_counts.items())),
            "label_counts": label_counts,
            "group_count": len(groups),
            "sample_count": len(rows),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "assignment_checksum": assignment_checksum,
        },
    )
    return RepresentationSplitBuildResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        group_count=len(groups),
        sample_count=len(rows),
    )


def load_representation_split_assignment(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(Path(path))
    assignments: dict[str, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    for row in rows:
        if row.get("schema") != ASSIGNMENT_SCHEMA:
            raise ValueError("representation split assignment schema mismatch")
        group = _required_text(row, "split_group_id")
        if group in assignments:
            raise ValueError(f"duplicate split_group_id in assignment: {group}")
        if row.get("master_split") not in MASTER_SPLITS:
            raise ValueError(f"invalid master_split for assignment group {group}")
        if row.get("representation_split") not in REPRESENTATION_SPLITS:
            raise ValueError(f"invalid representation_split for assignment group {group}")
        listed = row.get("sample_ids")
        if not isinstance(listed, list) or not listed or any(not str(item) for item in listed):
            raise ValueError(f"assignment group {group} requires sample_ids")
        duplicates = sample_ids.intersection(map(str, listed))
        if duplicates:
            raise ValueError(f"sample IDs cross assignment groups: {sorted(duplicates)[:3]}")
        sample_ids.update(map(str, listed))
        assignments[group] = row
    if not assignments:
        raise ValueError("representation split assignment is empty")
    return assignments


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"split config schema must be {CONFIG_SCHEMA}")
    required = {
        "key",
        "source_manifests",
        "seed",
        "calibration_fraction",
        "calibration_rounding",
        "ranking_rule",
        "master_split_field",
        "split_group_field",
        "scope",
        "use_in_main_only",
        "calibration_master_split",
        "calibration_eligible_sample_type",
        "minimum_eligible_groups",
        "assignments",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"split config missing fields: {', '.join(sorted(missing))}")
    if payload["calibration_fraction"] != 0.5:
        raise ValueError("calibration_fraction must be explicitly pre-registered as 0.5")
    if payload["calibration_rounding"] != "floor":
        raise ValueError("calibration_rounding must be floor")
    if payload["ranking_rule"] != "sha256(seed:split_group_id)":
        raise ValueError("unsupported split ranking_rule")
    if payload["calibration_master_split"] != "val":
        raise ValueError("calibration carve must use official validation only")
    if payload["calibration_eligible_sample_type"] != "Aligned":
        raise ValueError("calibration carve eligibility must be Aligned")
    if payload["scope"] != "all_valid_conflict_aligned":
        raise ValueError("split scope must be all_valid_conflict_aligned")
    if payload["use_in_main_only"] is not False:
        raise ValueError("use_in_main_only must be explicitly false for the all-A/C scope")
    expected_assignments = {
        "train": "relation_train",
        "validation": "relation_val",
        "calibration": "aligned_calibration",
        "test": "official_test",
    }
    if payload["assignments"] != expected_assignments:
        raise ValueError("split assignment names do not match the registered policy")
    return payload


def _load_source_rows(
    config_path: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    source_paths = config["source_manifests"]
    if not isinstance(source_paths, list) or not source_paths:
        raise ValueError("source_manifests must be a non-empty list")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    seen_samples: set[str] = set()
    for raw_path in source_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        source_rows = _read_jsonl(path)
        sources.append({"path": str(raw_path), "sha256": _sha256(path)})
        for row in source_rows:
            sample_id = _required_text(row, "sample_id")
            if _required_text(row, "sample_type") not in {"Aligned", "Conflict"}:
                raise ValueError("split scope accepts only valid Conflict/Aligned samples")
            if sample_id in seen_samples:
                raise ValueError(f"sample_id appears in multiple source rows: {sample_id}")
            seen_samples.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError("split sources contain no valid Conflict/Aligned samples")
    return rows, sources


def _group_rows(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_field = str(config["master_split_field"])
    group_field = str(config["split_group_field"])
    for row in rows:
        group = _required_text(row, group_field)
        master_split = _required_text(row, split_field)
        if master_split not in MASTER_SPLITS:
            raise ValueError(f"invalid official master split: {master_split}")
        _required_text(row, "sample_type")
        groups[group].append(row)
    for group, group_rows in groups.items():
        splits = {str(row[split_field]) for row in group_rows}
        if len(splits) != 1:
            raise ValueError(f"split_group_id {group} crosses official master splits")
    return dict(groups)


def _assign_groups(
    groups: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, str]:
    split_field = str(config["master_split_field"])
    eligible = sorted(
        group
        for group, rows in groups.items()
        if {str(row[split_field]) for row in rows} == {"val"}
        and {str(row["sample_type"]) for row in rows} == {"Aligned"}
    )
    minimum = int(config["minimum_eligible_groups"])
    if len(eligible) < minimum:
        raise ValueError(
            f"calibration split requires at least {minimum} eligible Aligned validation groups"
        )
    ranked = sorted(
        eligible,
        key=lambda group: hashlib.sha256(f"{config['seed']}:{group}".encode()).hexdigest(),
    )
    calibration_count = math.floor(len(ranked) * float(config["calibration_fraction"]))
    if calibration_count <= 0 or calibration_count >= len(ranked):
        raise ValueError("calibration carve must leave non-empty calibration and validation groups")
    calibration_groups = set(ranked[:calibration_count])
    assignment: dict[str, str] = {}
    for group, rows in groups.items():
        master_split = str(rows[0][split_field])
        if master_split == "train":
            assignment[group] = "relation_train"
        elif master_split == "test":
            assignment[group] = "official_test"
        elif group in calibration_groups:
            assignment[group] = "aligned_calibration"
        else:
            assignment[group] = "relation_val"
    _require_train_val_labels(groups, assignment)
    return assignment


def _require_train_val_labels(
    groups: dict[str, list[dict[str, Any]]], assignment: dict[str, str]
) -> None:
    for split in ("relation_train", "relation_val"):
        labels = {
            str(row["sample_type"])
            for group, group_split in assignment.items()
            if group_split == split
            for row in groups[group]
        }
        if not {"Aligned", "Conflict"} <= labels:
            raise ValueError(f"{split} must contain both Aligned and Conflict samples")


def _manifest_rows(
    groups: dict[str, list[dict[str, Any]]],
    assignment: dict[str, str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    split_field = str(config["master_split_field"])
    return [
        {
            "schema": ASSIGNMENT_SCHEMA,
            "config_key": config["key"],
            "split_group_id": group,
            "master_split": str(groups[group][0][split_field]),
            "representation_split": assignment[group],
            "sample_ids": sorted(str(row["sample_id"]) for row in groups[group]),
            "sample_count": len(groups[group]),
            "protocols": sorted({str(row.get("protocol", "")) for row in groups[group]}),
            "source_datasets": sorted(
                {str(row.get("source_dataset", "")) for row in groups[group]}
            ),
        }
        for group in sorted(groups)
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def _required_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"split source field {field} must be non-empty text")
    return value


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
