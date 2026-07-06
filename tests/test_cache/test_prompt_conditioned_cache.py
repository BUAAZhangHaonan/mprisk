from __future__ import annotations

import json

from mprisk.cache.hidden_state_cache import HiddenStateEntry
from mprisk.cache.prompt_conditioned_cache import (
    PromptConditionedManifest,
    PromptConditionedStateEntry,
    load_prompt_conditioned_manifest,
    read_prompt_conditioned_entries,
    write_prompt_conditioned_manifest,
)
from scripts.build_prompt_conditioned_cache import build_prompt_conditioned_cache


def _entry(tmp_path, **overrides) -> PromptConditionedStateEntry:
    values = {
        "sample_id": "sample-1",
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": "m1",
        "prompt_set_key": "vt_primary_v1",
        "prompt_id": "vt_primary_v1_t01",
        "shard_path": "shards/prompted.safetensors",
        "index_in_shard": 2,
        "layer_count": 3,
        "hidden_dim": 4,
        "token_count": 5,
        "t0_token_index": 1,
        "cache_root": tmp_path,
        "checksum": "sha256:abc",
        "metadata": {"tensor_key": "hidden_states", "dataset_key": "ch_sims_v2", "split": "test"},
    }
    values.update(overrides)
    return PromptConditionedStateEntry(**values)


def test_entry_normalizes_protocol_and_condition(tmp_path) -> None:
    entry = _entry(tmp_path, protocol="VT", condition="m12")

    assert entry.protocol == "vt"
    assert entry.condition == "M12"
    assert entry.key == (
        "sample-1",
        "qwen3_vl_8b",
        "vt",
        "M12",
        "vt_primary_v1",
        "vt_primary_v1_t01",
    )


def test_manifest_round_trips_jsonl(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    entries = [_entry(tmp_path), _entry(tmp_path, sample_id="sample-2", prompt_id="vt_primary_v1_t02")]

    written = write_prompt_conditioned_manifest(manifest_path, entries)
    loaded_entries = read_prompt_conditioned_entries(written)

    assert [entry.sample_id for entry in loaded_entries] == ["sample-1", "sample-2"]
    assert loaded_entries[0].protocol == "vt"
    assert loaded_entries[0].condition == "M1"
    assert loaded_entries[0].cache_root == tmp_path
    assert loaded_entries[0].metadata["tensor_key"] == "hidden_states"


def test_manifest_lookup_uses_full_prompt_conditioned_key(tmp_path) -> None:
    manifest = PromptConditionedManifest(
        [
            _entry(tmp_path, condition="M1", prompt_id="vt_primary_v1_t01"),
            _entry(tmp_path, condition="M2", prompt_id="vt_primary_v1_t01"),
            _entry(tmp_path, condition="M1", prompt_id="vt_primary_v1_t02"),
        ]
    )

    found = manifest.lookup(
        sample_id="sample-1",
        model_key="qwen3_vl_8b",
        protocol="VT",
        condition="m2",
        prompt_set_key="vt_primary_v1",
        prompt_id="vt_primary_v1_t01",
    )

    assert found is not None
    assert found.condition == "M2"
    assert found.prompt_id == "vt_primary_v1_t01"
    assert manifest.lookup(
        sample_id="sample-1",
        model_key="qwen3_vl_8b",
        protocol="VT",
        condition="m12",
        prompt_set_key="vt_primary_v1",
        prompt_id="vt_primary_v1_t01",
    ) is None


def test_entry_adapts_to_hidden_state_entry_with_t0_metadata(tmp_path) -> None:
    entry = _entry(tmp_path, metadata={"tensor_key": "hidden_states"})

    hidden_entry = entry.to_hidden_state_entry()

    assert isinstance(hidden_entry, HiddenStateEntry)
    assert hidden_entry.sample_id == entry.sample_id
    assert hidden_entry.model_key == entry.model_key
    assert hidden_entry.protocol == "vt"
    assert hidden_entry.condition == "M1"
    assert hidden_entry.shard_path == entry.shard_path
    assert hidden_entry.index_in_shard == entry.index_in_shard
    assert hidden_entry.layer_count == entry.layer_count
    assert hidden_entry.hidden_dim == entry.hidden_dim
    assert hidden_entry.token_count == entry.token_count
    assert hidden_entry.cache_root == tmp_path
    assert hidden_entry.metadata["tensor_key"] == "hidden_states"
    assert hidden_entry.metadata["t0_token_index"] == 1


def test_load_prompt_conditioned_manifest_lookup(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    write_prompt_conditioned_manifest(manifest_path, [_entry(tmp_path)])

    loaded = load_prompt_conditioned_manifest(manifest_path)

    assert loaded.lookup(
        sample_id="sample-1",
        model_key="qwen3_vl_8b",
        protocol="vt",
        condition="M1",
        prompt_set_key="vt_primary_v1",
        prompt_id="vt_primary_v1_t01",
    ) is not None


def test_build_prompt_conditioned_cache_mode_a_exports_manifest_summary_and_missing_rows(
    tmp_path,
) -> None:
    source_manifest = tmp_path / "source.jsonl"
    source_rows = [
        {
            "sample_id": "sample-1",
            "model_key": "qwen3_vl_8b",
            "protocol": "VT",
            "condition": "m1",
            "prompt_set_key": "vt_primary_v1",
            "prompt_id": "vt_primary_v1_t01",
            "artifact_uri": "existing/prompted-m1.safetensors",
            "index_in_shard": "0",
            "layer_count": "2",
            "hidden_dim": "3",
            "token_count": "4",
            "t0_token_index": "1",
            "checksum": "sha256:abc",
            "cache_root": str(tmp_path),
            "extra_field": "kept-in-metadata",
        },
        {
            "sample_id": "sample-2",
            "model_key": "qwen3_vl_8b",
            "protocol": "VT",
            "condition": "m2",
            "prompt_set_key": "vt_primary_v1",
            "prompt_id": "vt_primary_v1_t01",
            "index_in_shard": "0",
            "layer_count": "2",
            "hidden_dim": "3",
            "token_count": "4",
            "t0_token_index": "1",
        },
    ]
    source_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    result = build_prompt_conditioned_cache(
        mode="A",
        source_manifest=source_manifest,
        output_root=tmp_path / "outputs/prompt_conditioned_cache",
        model_key="qwen3_vl_8b",
        protocol="VT",
        prompt_set_key="vt_primary_v1",
    )

    assert result.manifest_path == (
        tmp_path
        / "outputs/prompt_conditioned_cache/qwen3_vl_8b/vt/vt_primary_v1/manifest.jsonl"
    )
    rows = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    missing = [
        json.loads(line)
        for line in result.missing_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert rows[0]["condition"] == "M1"
    assert rows[0]["protocol"] == "vt"
    assert rows[0]["shard_path"] == "existing/prompted-m1.safetensors"
    assert rows[0]["metadata"]["extra_field"] == "kept-in-metadata"
    assert summary["total_source_rows"] == 2
    assert summary["exported_rows"] == 1
    assert summary["missing_rows"] == 1
    assert summary["manifest_path"] == str(result.manifest_path)
    assert missing[0]["sample_id"] == "sample-2"
    assert "shard_path" in missing[0]["reason"]
