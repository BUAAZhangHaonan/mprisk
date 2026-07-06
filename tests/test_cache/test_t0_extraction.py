from __future__ import annotations

import numpy as np
from safetensors.numpy import save_file

from mprisk.cache.hidden_state_cache import HiddenStateEntry
from mprisk.cache.prefill_extract import extract_t0_trajectory, t0_token_index


def _entry(tmp_path, *, metadata=None) -> HiddenStateEntry:
    return HiddenStateEntry(
        sample_id="sample-1",
        model_key="qwen3_vl_8b",
        protocol="VT",
        condition="M1",
        dataset_key="ch_sims_v2",
        split="test",
        shard_path="shards/m1.safetensors",
        index_in_shard=0,
        layer_count=2,
        hidden_dim=3,
        token_count=4,
        cache_root=tmp_path,
        metadata=metadata or {},
    )


def test_t0_token_index_defaults_to_last_conditioning_token(tmp_path) -> None:
    entry = _entry(tmp_path)

    assert t0_token_index(entry) == -1


def test_t0_token_index_uses_metadata_override(tmp_path) -> None:
    entry = _entry(tmp_path, metadata={"t0_token_index": 1})

    assert t0_token_index(entry) == 1


def test_extract_t0_trajectory_reads_sample_from_safetensors_shard(tmp_path) -> None:
    shard = tmp_path / "shards/m1.safetensors"
    shard.parent.mkdir(parents=True)
    hidden_states = np.arange(2 * 2 * 4 * 3, dtype=np.float32).reshape(2, 2, 4, 3)
    save_file({"hidden_states": hidden_states}, shard)
    entry = _entry(tmp_path, metadata={"tensor_key": "hidden_states"})

    trajectory = extract_t0_trajectory(entry)

    assert trajectory == hidden_states[0, :, -1, :].tolist()


def test_extract_t0_trajectory_supports_metadata_token_override(tmp_path) -> None:
    shard = tmp_path / "shards/m1.safetensors"
    shard.parent.mkdir(parents=True)
    hidden_states = np.arange(1 * 2 * 4 * 3, dtype=np.float32).reshape(1, 2, 4, 3)
    save_file({"hidden_states": hidden_states}, shard)
    entry = _entry(tmp_path, metadata={"tensor_key": "hidden_states", "t0_token_index": 2})

    trajectory = extract_t0_trajectory(entry)

    assert trajectory == hidden_states[0, :, 2, :].tolist()
