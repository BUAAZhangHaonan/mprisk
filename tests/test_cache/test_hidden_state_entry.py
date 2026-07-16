from __future__ import annotations

from mprisk.cache.hidden_state_cache import HiddenStateEntry


def test_hidden_state_entry_keeps_full_manifest_location_contract(tmp_path) -> None:
    entry = HiddenStateEntry(
        sample_id="sample-1",
        model_key="qwen3_vl_8b",
        protocol="VT",
        condition="m12",
        dataset_key="ch_sims_v2",
        split="test",
        shard_path="shards/m12.safetensors",
        index_in_shard=3,
        layer_count=33,
        hidden_dim=4096,
        token_count=17,
        cache_root=tmp_path,
        checksum="sha256:abc",
        metadata={"run_id": "run-1"},
    )

    assert entry.condition == "M12"
    assert entry.protocol == "vt"
    assert entry.shard_file == tmp_path / "shards/m12.safetensors"
    assert entry.checksum == "sha256:abc"
    assert entry.metadata == {"run_id": "run-1"}
