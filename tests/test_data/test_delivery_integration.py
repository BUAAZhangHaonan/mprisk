from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from mprisk.data.delivery import (
    DEFAULT_PROVENANCE_PATH,
    load_delivery_provenance,
    validate_delivery,
    validate_media_paths,
)
from mprisk.data.manifests import read_jsonl
from mprisk.data.splits import assign_split


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = ROOT / "data/processed/manifests"


def _rows(path: Path) -> list[dict[str, object]]:
    return read_jsonl(path)


def test_frozen_delivery_matches_provenance() -> None:
    report = validate_delivery(ROOT, check_media=False, verify_archive=False)

    assert report.total_rows == 4754
    assert report.sample_type_counts == {
        "Conflict": 585,
        "Aligned": 4169,
        "Ambiguous": 0,
    }
    assert report.protocol_counts == {"VT": 2512, "VA": 2242}
    assert report.use_in_main_counts == {"Aligned": 4089, "Conflict": 460}
    assert report.source_counts == {"real": 4225, "generated": 529}
    assert report.unique_sample_ids == 4754
    assert report.unique_split_groups == 2974
    assert report.unique_media_paths == 2974


def test_final_class_manifests_are_exact_partition() -> None:
    unified = _rows(MANIFEST_ROOT / "unified_sample_manifest.jsonl")
    conflict = _rows(MANIFEST_ROOT / "conflict_manifest.jsonl")
    aligned = _rows(MANIFEST_ROOT / "aligned_manifest.jsonl")
    ambiguous = _rows(MANIFEST_ROOT / "ambiguous_manifest.jsonl")

    def canonical(row: dict[str, object]) -> str:
        return json.dumps(row, ensure_ascii=False, sort_keys=True)

    assert len({str(row["sample_id"]) for row in unified}) == len(unified)
    assert Counter(map(canonical, unified)) == Counter(
        map(canonical, conflict + aligned + ambiguous)
    )


def test_annotation_waiver_preserves_current_inclusion_labels() -> None:
    provenance = load_delivery_provenance(ROOT, DEFAULT_PROVENANCE_PATH)
    policy = provenance.annotation_policy

    assert policy.accepted_for_current_experiments is True
    assert policy.annotation_requirement_waived is True
    assert policy.preserve_delivered_use_in_main is True
    assert policy.observed_annotation_count == 0
    assert policy.observed_annotator_agreement == 0.0
    assert set(policy.pending_statistics) == {
        "multi_annotator_mean",
        "multi_annotator_standard_deviation",
    }


def test_split_manifests_are_deterministic_partition_without_group_leakage() -> None:
    split_rows = {
        split: _rows(MANIFEST_ROOT / "splits" / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    assert {split: len(rows) for split, rows in split_rows.items()} == {
        "train": 3354,
        "val": 666,
        "test": 734,
    }

    ids = [str(row["sample_id"]) for rows in split_rows.values() for row in rows]
    unified_ids = {
        str(row["sample_id"]) for row in _rows(MANIFEST_ROOT / "unified_sample_manifest.jsonl")
    }
    assert len(ids) == len(set(ids))
    assert set(ids) == unified_ids

    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for split, rows in split_rows.items():
        for row in rows:
            group = str(row["split_group_id"])
            assert row["split"] == split
            assert split == assign_split(group)
            group_splits[group].add(split)
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_protocol_manifests_are_exact_mutually_exclusive_partition() -> None:
    protocol_rows = {
        "VT": _rows(MANIFEST_ROOT / "protocol_manifests/vt_primary.jsonl"),
        "VA": _rows(MANIFEST_ROOT / "protocol_manifests/va_aux.jsonl"),
    }
    assert {protocol: len(rows) for protocol, rows in protocol_rows.items()} == {
        "VT": 2512,
        "VA": 2242,
    }
    ids = [str(row["sample_id"]) for rows in protocol_rows.values() for row in rows]
    unified_ids = {
        str(row["sample_id"]) for row in _rows(MANIFEST_ROOT / "unified_sample_manifest.jsonl")
    }
    assert len(ids) == len(set(ids))
    assert set(ids) == unified_ids
    for protocol, rows in protocol_rows.items():
        assert {row["protocol"] for row in rows} == {protocol}


def test_media_path_validation_checks_presence_and_nonzero_size(tmp_path: Path) -> None:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    row = {"sample_id": "sample", "media_paths": {"vision": str(media), "audio": str(media)}}

    assert validate_media_paths([row]) == 1
    media.unlink()
    with pytest.raises(FileNotFoundError, match="1 delivery media files are missing"):
        validate_media_paths([row])
