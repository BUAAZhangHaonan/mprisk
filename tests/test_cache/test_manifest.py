from __future__ import annotations

from mprisk.cache.cache_manifest import validate_cache_manifest_entry


def test_cache_manifest_entry_contract() -> None:
    assert validate_cache_manifest_entry(
        {
            "model_key": "qwen3_vl_8b",
            "protocol": "vt",
            "dataset_key": "ch_sims_v2",
            "split": "test",
            "condition": "m12",
            "artifact_uri": "outputs/full_cache/example.safetensors",
        }
    )
