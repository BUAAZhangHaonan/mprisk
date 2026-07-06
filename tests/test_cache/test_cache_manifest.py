from __future__ import annotations

import csv
import json

import pytest

from mprisk.cache.cache_manifest import (
    CacheResolutionError,
    load_full_cache_manifest,
    write_cache_resolution_summary,
)


def _write_manifest(root, entries) -> None:
    manifest_path = root / "outputs/full_cache/manifests/unified_full_cache_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"schema": "mprisk_full_cache_manifest_v1", "entries": entries}),
        encoding="utf-8",
    )


def _write_ledger(root, rows) -> None:
    ledger_path = root / "outputs/full_cache/manifests/extraction_ledger.csv"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "sample_id",
        "model_key",
        "protocol",
        "dataset_key",
        "split",
        "condition",
        "status",
        "artifact_uri",
        "index_in_shard",
        "layer_count",
        "hidden_dim",
        "token_count",
        "notes",
    ]
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _touch_shard(root, rel_path: str) -> None:
    shard_path = root / rel_path
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(b"")


def test_loads_entries_from_manifest_and_ledger_with_empty_manifest_entries(tmp_path) -> None:
    _touch_shard(tmp_path, "outputs/full_cache/shards/m1.safetensors")
    _write_manifest(tmp_path, [])
    _write_ledger(
        tmp_path,
        [
            {
                "run_id": "run-1",
                "sample_id": "sample-1",
                "model_key": "qwen3_vl_8b",
                "protocol": "VT",
                "dataset_key": "ch_sims_v2",
                "split": "test",
                "condition": "m1",
                "status": "ok",
                "artifact_uri": "outputs/full_cache/shards/m1.safetensors",
                "index_in_shard": "7",
                "layer_count": "33",
                "hidden_dim": "4096",
                "token_count": "11",
                "notes": "ledger row",
            }
        ],
    )

    manifest = load_full_cache_manifest(tmp_path)
    entry = manifest.query(
        sample_id="sample-1",
        model_key="qwen3_vl_8b",
        protocol="vt",
        condition="M1",
    )

    assert entry is not None
    assert entry.condition == "M1"
    assert entry.shard_path == "outputs/full_cache/shards/m1.safetensors"
    assert entry.index_in_shard == 7
    assert entry.metadata["run_id"] == "run-1"


def test_manifest_query_accepts_shard_path_and_protocol_case(tmp_path) -> None:
    _touch_shard(tmp_path, "outputs/full_cache/shards/m2.safetensors")
    _write_manifest(
        tmp_path,
        [
            {
                "sample_id": "sample-2",
                "model_key": "qwen3_vl_8b",
                "protocol": "VT",
                "condition": "m2",
                "dataset_key": "ch_sims_v2",
                "split": "test",
                "shard_path": "outputs/full_cache/shards/m2.safetensors",
                "index_in_shard": 2,
                "layer_count": 33,
                "hidden_dim": 4096,
                "token_count": 9,
            }
        ],
    )
    _write_ledger(tmp_path, [])

    manifest = load_full_cache_manifest(tmp_path)

    assert manifest.query("sample-2", "qwen3_vl_8b", "vt", "M2").shard_file.exists()


def test_resolves_m1_m2_m12_for_sample_batch_and_reports_missing_conditions(tmp_path) -> None:
    entries = []
    for sample_id in ("sample-ok", "sample-missing"):
        conditions = ("M1", "M2", "M12") if sample_id == "sample-ok" else ("M1", "M12")
        for condition in conditions:
            shard = f"outputs/full_cache/shards/{sample_id}-{condition}.safetensors"
            _touch_shard(tmp_path, shard)
            entries.append(
                {
                    "sample_id": sample_id,
                    "model_key": "qwen3_vl_8b",
                    "protocol": "vt",
                    "condition": condition,
                    "dataset_key": "ch_sims_v2",
                    "split": "test",
                    "artifact_uri": shard,
                    "index_in_shard": 0,
                    "layer_count": 33,
                    "hidden_dim": 4096,
                    "token_count": 11,
                }
            )
    _write_manifest(tmp_path, entries)
    _write_ledger(tmp_path, [])

    manifest = load_full_cache_manifest(tmp_path)
    result = manifest.resolve_m_conditions(
        ["sample-ok", "sample-missing"],
        model_key="qwen3_vl_8b",
        protocol="VT",
    )

    assert result["sample-ok"].ok is True
    assert set(result["sample-ok"].entries) == {"M1", "M2", "M12"}
    assert result["sample-missing"].ok is False
    assert result["sample-missing"].missing_conditions == ["M2"]
    with pytest.raises(CacheResolutionError, match="sample-missing.*M2"):
        manifest.require_m_conditions(["sample-ok", "sample-missing"], "qwen3_vl_8b", "vt")


def test_write_cache_resolution_summary_json_and_markdown(tmp_path) -> None:
    _write_manifest(tmp_path, [])
    _write_ledger(tmp_path, [])
    manifest = load_full_cache_manifest(tmp_path)
    resolutions = manifest.resolve_m_conditions(["sample-1"], "qwen3_vl_8b", "vt")

    paths = write_cache_resolution_summary(
        resolutions,
        reports_dir=tmp_path / "outputs/state_data/reports",
    )

    summary = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert paths["json"].name == "cache_resolution_summary.json"
    assert paths["markdown"].name == "cache_resolution_summary.md"
    assert summary["total_samples"] == 1
    assert summary["missing_samples"] == 1
    assert summary["samples"][0]["missing_conditions"] == ["M1", "M2", "M12"]
    assert "sample-1" in paths["markdown"].read_text(encoding="utf-8")
