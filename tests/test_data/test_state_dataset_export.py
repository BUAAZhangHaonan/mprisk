from __future__ import annotations

import json
from pathlib import Path

import pytest

from mprisk.data.manifests import write_jsonl
from mprisk.data.state_dataset import build_state_dataset


def _manifest_row(
    sample_id: str, sample_type: str = "Conflict", split: str = "train"
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_dataset": "ch_sims_v2",
        "source_id": f"{sample_id}-source",
        "protocol": "VT",
        "sample_type": sample_type,
        "split_group_id": sample_id,
        "split": split,
        "media_paths": {"vision": "video.mp4", "text": "text.txt"},
        "views": {
            "M1": {
                "modality": "vision",
                "label": "positive",
                "specific_affect": "joy",
                "is_clear": True,
            },
            "M2": {
                "modality": "text",
                "label": "negative",
                "specific_affect": "anger",
                "is_clear": True,
            },
            "M12": {
                "modality": "vision+text",
                "label": "negative",
                "specific_affect": "frustration",
                "is_clear": True,
            },
        },
        "dominant_modality": "M2",
        "use_in_main": True,
    }


def _cache_entry(root, sample_id: str, condition: str, *, hidden_dim: int = 3) -> dict[str, object]:
    shard_path = f"outputs/full_cache/shards/{sample_id}-{condition}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"placeholder")
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": condition,
        "dataset_key": "ch_sims_v2",
        "split": "test",
        "shard_path": shard_path,
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": hidden_dim,
        "token_count": 4,
        "metadata": {"t0_token_index": -1},
    }


def _write_cache_manifest(root, entries) -> None:
    manifest = root / "outputs/full_cache/manifests/unified_full_cache_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    (manifest.parent / "extraction_ledger.csv").write_text("", encoding="utf-8")


def _write_split_assignment(root: Path, assignments: dict[str, tuple[str, str]]) -> Path:
    path = root / "representation_split_assignment_v1.jsonl"
    rows = [
        {
            "schema": "mprisk_representation_split_assignment_v1",
            "config_key": "fixture_v1",
            "split_group_id": sample_id,
            "master_split": master_split,
            "representation_split": representation_split,
            "sample_ids": [sample_id],
            "sample_count": 1,
            "protocols": ["VT"],
            "source_datasets": ["ch_sims_v2"],
        }
        for sample_id, (master_split, representation_split) in sorted(assignments.items())
    ]
    write_jsonl(path, rows)
    return path


def test_build_state_dataset_exports_resolved_rows_and_missing_cache_report(tmp_path) -> None:
    labels = tmp_path / "data/processed/manifests/conflict_manifest.jsonl"
    labels.parent.mkdir(parents=True)
    write_jsonl(labels, [_manifest_row("sample-ok"), _manifest_row("sample-missing")])
    _write_cache_manifest(
        tmp_path,
        [
            _cache_entry(tmp_path, "sample-ok", "M1"),
            _cache_entry(tmp_path, "sample-ok", "M2"),
            _cache_entry(tmp_path, "sample-ok", "M12"),
            _cache_entry(tmp_path, "sample-missing", "M1"),
        ],
    )

    result = build_state_dataset(
        manifest_paths=[labels],
        cache_root=tmp_path,
        model_key="qwen3_vl_8b",
        protocol="VT",
        split_assignment_path=_write_split_assignment(
            tmp_path,
            {
                "sample-ok": ("train", "relation_train"),
                "sample-missing": ("train", "relation_train"),
            },
        ),
        output_dir=tmp_path / "outputs/state_data/qwen3_vl_8b/VT",
    )

    manifest_rows = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    missing_rows = [
        json.loads(line)
        for line in result.missing_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.resolved_count == 1
    assert result.missing_count == 1
    assert manifest_rows[0]["sample_id"] == "sample-ok"
    assert manifest_rows[0]["model_key"] == "qwen3_vl_8b"
    assert manifest_rows[0]["split_group_id"] == "sample-ok"
    assert manifest_rows[0]["master_split"] == "train"
    assert manifest_rows[0]["representation_split"] == "relation_train"
    assert manifest_rows[0]["calibration_split"] == ""
    assert len(manifest_rows[0]["split_assignment_sha256"]) == 64
    assert manifest_rows[0]["target_label"] == "negative"
    assert manifest_rows[0]["view_labels"] == {
        "M1": {"label": "positive", "specific_affect": "joy", "is_clear": True},
        "M2": {"label": "negative", "specific_affect": "anger", "is_clear": True},
        "M12": {"label": "negative", "specific_affect": "frustration", "is_clear": True},
    }
    assert manifest_rows[0]["m1_entry"]["condition"] == "M1"
    assert manifest_rows[0]["trajectory_meta"] == {
        "layer_count": 2,
        "hidden_dim": 3,
        "t0_token_index": -1,
    }
    assert missing_rows[0]["sample_id"] == "sample-missing"
    assert missing_rows[0]["missing_conditions"] == ["M2", "M12"]
    assert missing_rows[0]["master_split"] == "train"
    assert missing_rows[0]["representation_split"] == "relation_train"
    assert summary["resolved_rows"] == 1
    assert summary["missing_cache_rows"] == 1


def test_build_state_dataset_filters_protocol_but_preserves_legacy_main_flag(tmp_path) -> None:
    labels = tmp_path / "data/processed/manifests/unified_sample_manifest.jsonl"
    labels.parent.mkdir(parents=True)
    skipped = _manifest_row("sample-skip")
    skipped["use_in_main"] = False
    for view in skipped["views"].values():
        view["label"] = ""
        view["specific_affect"] = ""
        view["is_clear"] = False
    other_protocol = _manifest_row("sample-va")
    other_protocol["protocol"] = "VA"
    write_jsonl(labels, [_manifest_row("sample-ok"), skipped, other_protocol])
    _write_cache_manifest(
        tmp_path,
        [
            _cache_entry(tmp_path, "sample-ok", "M1"),
            _cache_entry(tmp_path, "sample-ok", "M2"),
            _cache_entry(tmp_path, "sample-ok", "M12"),
            _cache_entry(tmp_path, "sample-skip", "M1"),
            _cache_entry(tmp_path, "sample-skip", "M2"),
            _cache_entry(tmp_path, "sample-skip", "M12"),
        ],
    )

    result = build_state_dataset(
        manifest_paths=[labels],
        cache_root=tmp_path,
        model_key="qwen3_vl_8b",
        protocol="vt",
        split_assignment_path=_write_split_assignment(
            tmp_path,
            {
                "sample-ok": ("train", "relation_train"),
                "sample-skip": ("train", "relation_train"),
            },
        ),
        output_dir=tmp_path / "outputs/state_data/qwen3_vl_8b/VT",
    )

    rows = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["sample_id"] for row in rows] == ["sample-ok", "sample-skip"]
    assert [row["use_in_main"] for row in rows] == [True, False]
    assert rows[1]["target_label"] is None


def test_build_state_dataset_rejects_missing_master_split_before_cache_resolution(
    tmp_path,
) -> None:
    labels = tmp_path / "labels.jsonl"
    row = _manifest_row("sample-missing-split")
    del row["split"]
    write_jsonl(labels, [row])
    _write_cache_manifest(tmp_path, [])
    assignment = _write_split_assignment(
        tmp_path, {"sample-missing-split": ("train", "relation_train")}
    )

    with pytest.raises(ValueError, match="missing a valid master_split"):
        build_state_dataset(
            manifest_paths=[labels],
            cache_root=tmp_path,
            model_key="qwen3_vl_8b",
            protocol="VT",
            split_assignment_path=assignment,
        )
