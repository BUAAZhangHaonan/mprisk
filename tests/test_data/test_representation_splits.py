from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mprisk.data.representation_splits import build_representation_split_assignment


def _row(
    sample_id: str,
    group: str,
    master_split: str,
    sample_type: str,
    protocol: str = "VT",
    use_in_main: bool = True,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split_group_id": group,
        "split": master_split,
        "sample_type": sample_type,
        "source_dataset": "fixture",
        "protocol": protocol,
        "use_in_main": use_in_main,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _config(tmp_path: Path, sources: list[Path]) -> Path:
    path = tmp_path / "split.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "mprisk_representation_split_config_v1",
                "key": "fixture_split_v1",
                "source_manifests": [str(source) for source in sources],
                "seed": 20260716,
                "calibration_fraction": 0.5,
                "calibration_rounding": "floor",
                "ranking_rule": "sha256(seed:split_group_id)",
                "master_split_field": "split",
                "split_group_field": "split_group_id",
                "scope": "all_valid_conflict_aligned",
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _fixture_rows() -> list[dict[str, object]]:
    return [
        _row("train-a", "g-train-a", "train", "Aligned"),
        _row("train-c", "g-train-c", "train", "Conflict"),
        _row("val-a1-vt", "g-val-a1", "val", "Aligned"),
        _row("val-a1-va", "g-val-a1", "val", "Aligned", "VA"),
        _row("val-a2", "g-val-a2", "val", "Aligned"),
        _row("val-a3", "g-val-a3", "val", "Aligned"),
        _row("val-a4", "g-val-a4", "val", "Aligned"),
        _row("val-c", "g-val-c", "val", "Conflict"),
        _row("test-a", "g-test-a", "test", "Aligned"),
        _row("test-c", "g-test-c", "test", "Conflict"),
    ]


def test_split_assignment_is_group_level_deterministic_and_checksum_backed(tmp_path) -> None:
    first_source = _write_jsonl(tmp_path / "vt.jsonl", _fixture_rows()[::2])
    second_source = _write_jsonl(tmp_path / "va.jsonl", _fixture_rows()[1::2])
    result = build_representation_split_assignment(
        config_path=_config(tmp_path, [first_source, second_source]),
        output_dir=tmp_path / "out",
    )

    rows = [json.loads(line) for line in result.manifest_path.read_text().splitlines()]
    summary = json.loads(result.summary_path.read_text())
    by_group = {row["split_group_id"]: row for row in rows}
    ranked = sorted(
        ["g-val-a1", "g-val-a2", "g-val-a3", "g-val-a4"],
        key=lambda group: hashlib.sha256(f"20260716:{group}".encode()).hexdigest(),
    )

    assert {
        group
        for group, row in by_group.items()
        if row["representation_split"] == "aligned_calibration"
    } == set(ranked[:2])
    assert by_group["g-val-c"]["representation_split"] == "relation_val"
    assert by_group["g-test-a"]["representation_split"] == "official_test"
    assert by_group["g-test-c"]["representation_split"] == "official_test"
    assert by_group["g-val-a1"]["sample_ids"] == ["val-a1-va", "val-a1-vt"]
    assert summary["seed"] == 20260716
    assert summary["ranking_rule"] == "sha256(seed:split_group_id)"
    assert summary["calibration_fraction"] == 0.5
    assert summary["calibration_rounding"] == "floor"
    assert summary["group_counts"]["aligned_calibration"] == 2
    assert summary["manifest_sha256"] == hashlib.sha256(
        result.manifest_path.read_bytes()
    ).hexdigest()
    assert len(summary["assignment_checksum"]) == 64

    assignment_groups = {
        split: {
            row["split_group_id"]
            for row in rows
            if row["representation_split"] == split
        }
        for split in ("relation_train", "relation_val", "aligned_calibration", "official_test")
    }
    for left, left_groups in assignment_groups.items():
        for right, right_groups in assignment_groups.items():
            if left < right:
                assert left_groups.isdisjoint(right_groups)


def test_split_assignment_rejects_too_few_aligned_validation_groups(tmp_path) -> None:
    source = _write_jsonl(
        tmp_path / "rows.jsonl",
        [
            _row("train-a", "g-train-a", "train", "Aligned"),
            _row("train-c", "g-train-c", "train", "Conflict"),
            _row("val-a", "g-val-a", "val", "Aligned"),
            _row("val-c", "g-val-c", "val", "Conflict"),
            _row("test-a", "g-test-a", "test", "Aligned"),
        ],
    )
    with pytest.raises(ValueError, match="at least 2 eligible Aligned validation groups"):
        build_representation_split_assignment(
            config_path=_config(tmp_path, [source]),
            output_dir=tmp_path / "out",
        )


def test_split_assignment_rejects_group_crossing_official_splits(tmp_path) -> None:
    rows = _fixture_rows()
    rows.append(_row("leak", "g-train-a", "test", "Aligned"))
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)
    with pytest.raises(ValueError, match="crosses official master splits"):
        build_representation_split_assignment(
            config_path=_config(tmp_path, [source]),
            output_dir=tmp_path / "out",
        )


def test_split_assignment_includes_valid_ac_rows_regardless_of_legacy_main_flag(
    tmp_path,
) -> None:
    rows = _fixture_rows()
    rows.append(
        _row(
            "legacy-supplemental",
            "g-legacy-supplemental",
            "train",
            "Conflict",
            use_in_main=False,
        )
    )
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    result = build_representation_split_assignment(
        config_path=_config(tmp_path, [source]),
        output_dir=tmp_path / "out",
    )
    assignments = [json.loads(line) for line in result.manifest_path.read_text().splitlines()]
    summary = json.loads(result.summary_path.read_text())

    assert any(
        "legacy-supplemental" in row["sample_ids"] for row in assignments
    )
    assert summary["scope"] == "all_valid_conflict_aligned"
    assert summary["use_in_main_only"] is False


def test_registered_split_artifact_matches_sources_and_exact_counts(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = root / "configs/splits/representation_split_v1.yaml"
    committed_root = root / "data/processed/manifests/splits/representation_v1"
    rebuilt = build_representation_split_assignment(
        config_path=config,
        output_dir=tmp_path / "rebuilt",
    )
    committed_manifest = committed_root / "representation_split_assignment_v1.jsonl"
    committed_summary = json.loads(
        (committed_root / "representation_split_summary_v1.json").read_text()
    )

    assert rebuilt.manifest_path.read_bytes() == committed_manifest.read_bytes()
    assert committed_summary["scope"] == "all_valid_conflict_aligned"
    assert committed_summary["use_in_main_only"] is False
    assert committed_summary["legacy_use_in_main_counts"] == {
        "false": 205,
        "true": 4549,
    }
    assert committed_summary["sample_count"] == 4754
    assert committed_summary["sample_counts"]["relation_train"] == 3354
    assert (
        committed_summary["sample_counts"]["relation_val"]
        + committed_summary["sample_counts"]["aligned_calibration"]
        == 666
    )
    assert committed_summary["sample_counts"]["official_test"] == 734
    source_ids = {
        json.loads(line)["sample_id"]
        for path in (
            root / "data/processed/manifests/protocol_manifests/vt_primary.jsonl",
            root / "data/processed/manifests/protocol_manifests/va_aux.jsonl",
        )
        for line in path.read_text().splitlines()
        if line
    }
    assigned_ids = {
        sample_id
        for row in (
            json.loads(line) for line in committed_manifest.read_text().splitlines() if line
        )
        for sample_id in row["sample_ids"]
    }
    assert assigned_ids == source_ids
    assert len(assigned_ids) == 4754
    assert committed_summary["manifest_sha256"] == hashlib.sha256(
        committed_manifest.read_bytes()
    ).hexdigest()
