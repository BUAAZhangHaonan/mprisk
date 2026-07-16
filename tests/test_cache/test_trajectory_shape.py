from __future__ import annotations

import numpy as np
import pytest
from safetensors.numpy import save_file

from mprisk.cache.hidden_state_cache import HiddenStateEntry
from mprisk.cache.prefill_extract import bundle_three_views, extract_t0_trajectory


def _entry(
    tmp_path,
    condition: str,
    *,
    shard_name: str | None = None,
    layer_count: int = 2,
    hidden_dim: int = 3,
    token_count: int = 4,
    metadata=None,
) -> HiddenStateEntry:
    return HiddenStateEntry(
        sample_id="sample-1",
        model_key="qwen3_vl_8b",
        protocol="VT",
        condition=condition,
        dataset_key="ch_sims_v2",
        split="test",
        shard_path=f"shards/{shard_name or condition}.safetensors",
        index_in_shard=0,
        layer_count=layer_count,
        hidden_dim=hidden_dim,
        token_count=token_count,
        cache_root=tmp_path,
        metadata=metadata or {"tensor_key": "hidden_states"},
    )


def _write_hidden_states(tmp_path, name: str, array: np.ndarray) -> None:
    shard = tmp_path / f"shards/{name}.safetensors"
    shard.parent.mkdir(parents=True, exist_ok=True)
    save_file({"hidden_states": array.astype(np.float32)}, shard)


def test_extract_t0_trajectory_validates_expected_shape(tmp_path) -> None:
    _write_hidden_states(tmp_path, "M1", np.zeros((1, 3, 4, 3), dtype=np.float32))
    entry = _entry(tmp_path, "M1", layer_count=2)

    with pytest.raises(ValueError, match="layer_count"):
        extract_t0_trajectory(entry)


def test_extract_t0_trajectory_rejects_nan_values(tmp_path) -> None:
    hidden_states = np.zeros((1, 2, 4, 3), dtype=np.float32)
    hidden_states[0, 0, 3, 2] = np.nan
    _write_hidden_states(tmp_path, "M1", hidden_states)
    entry = _entry(tmp_path, "M1")

    with pytest.raises(ValueError, match="finite"):
        extract_t0_trajectory(entry)


def test_bundle_three_views_returns_trajectories_and_shared_meta(tmp_path) -> None:
    for condition in ("M1", "M2", "M12"):
        _write_hidden_states(tmp_path, condition, np.ones((1, 2, 4, 3), dtype=np.float32))

    bundle = bundle_three_views(
        _entry(tmp_path, "M1"),
        _entry(tmp_path, "M2"),
        _entry(tmp_path, "M12"),
    )

    assert bundle.sample_id == "sample-1"
    assert bundle.trajectory_meta == {
        "layer_count": 2,
        "hidden_dim": 3,
        "t0_token_index": -1,
    }
    assert len(bundle.m1_trajectory) == 2
    assert len(bundle.m2_trajectory) == 2
    assert len(bundle.m12_trajectory) == 2


def test_bundle_three_views_rejects_inconsistent_shape_metadata(tmp_path) -> None:
    for condition in ("M1", "M2", "M12"):
        _write_hidden_states(tmp_path, condition, np.ones((1, 2, 4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="same layer_count and hidden_dim"):
        bundle_three_views(
            _entry(tmp_path, "M1"),
            _entry(tmp_path, "M2", hidden_dim=4),
            _entry(tmp_path, "M12"),
        )
