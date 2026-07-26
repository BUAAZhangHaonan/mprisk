from __future__ import annotations

import json

from mprisk.cache.cache_manifest import load_full_cache_manifest
from mprisk.cache.validate import validate_full_cache_manifest


def _write_manifest(root, entries) -> None:
    manifest_path = root / "outputs/full_cache/manifests/unified_full_cache_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"schema": "mprisk_full_cache_manifest_v1", "entries": entries}),
        encoding="utf-8",
    )
    (manifest_path.parent / "extraction_ledger.csv").write_text(
        "run_id,model_key,protocol,dataset_key,split,condition,status,artifact_uri,notes\n",
        encoding="utf-8",
    )


def _touch_shard(root, rel_path: str) -> None:
    shard_path = root / rel_path
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(b"")


def _entry(sample_id: str, condition: str, shard_path: str, **overrides) -> dict[str, object]:
    entry = {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": condition,
        "dataset_key": "ch_sims_v2",
        "split": "test",
        "artifact_uri": shard_path,
        "index_in_shard": 0,
        "layer_count": 33,
        "hidden_dim": 4096,
        "token_count": 11,
    }
    entry.update(overrides)
    return entry


def test_validation_passes_when_three_conditions_are_present_and_consistent(tmp_path) -> None:
    entries = []
    for condition in ("M1", "M2", "M12"):
        shard = f"outputs/full_cache/shards/sample-1-{condition}.safetensors"
        _touch_shard(tmp_path, shard)
        entries.append(_entry("sample-1", condition, shard))
    _write_manifest(tmp_path, entries)

    report = validate_full_cache_manifest(load_full_cache_manifest(tmp_path))

    assert report.ok is True
    assert report.errors == []


def test_validation_reports_missing_condition_inconsistent_shape_missing_shard_and_duplicates(
    tmp_path,
) -> None:
    present_shard = "outputs/full_cache/shards/sample-1-M1.safetensors"
    missing_shard = "outputs/full_cache/shards/sample-1-M12.safetensors"
    _touch_shard(tmp_path, present_shard)
    _write_manifest(
        tmp_path,
        [
            _entry("sample-1", "M1", present_shard),
            _entry("sample-1", "m1", present_shard),
            _entry("sample-1", "M12", missing_shard, layer_count=34, hidden_dim=8192),
        ],
    )

    report = validate_full_cache_manifest(load_full_cache_manifest(tmp_path))
    codes = {error.code for error in report.errors}

    assert report.ok is False
    assert {
        "duplicate_key",
        "missing_condition",
        "inconsistent_layer_count",
        "inconsistent_hidden_dim",
        "missing_shard_file",
    }.issubset(codes)
