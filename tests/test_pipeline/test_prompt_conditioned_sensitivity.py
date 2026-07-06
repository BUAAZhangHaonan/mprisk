from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from mprisk.data.manifests import write_jsonl
from mprisk.state.s_measure import compute_s_for_bundle
from scripts.run_state_measurement_smoke import build_embedding_manifest


def _state_cache(root, sample_id: str, condition: str) -> dict[str, object]:
    shard_path = f"state/{sample_id}-{condition}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.zeros((1, 2, 4, 3), dtype=np.float32)
    hidden_states[0, :, -1, :] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    save_file({"hidden_states": hidden_states}, shard)
    return _entry_row(root, sample_id, condition, shard_path)


def _prompted_state(
    root,
    sample_id: str,
    condition: str,
    prompt_id: str,
    vector: list[float],
) -> dict[str, object]:
    shard_path = f"prompted/{sample_id}-{condition}-{prompt_id}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.zeros((1, 2, 4, 3), dtype=np.float32)
    hidden_states[0, :, -1, :] = np.asarray(vector, dtype=np.float32)
    save_file({"hidden_states": hidden_states}, shard)
    return {
        **_entry_row(root, sample_id, condition, shard_path),
        "prompt_set_key": "vt_primary_v1",
        "prompt_id": prompt_id,
        "t0_token_index": -1,
    }


def _entry_row(root, sample_id: str, condition: str, shard_path: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "vt",
        "condition": condition,
        "dataset_key": "ch_sims_v2",
        "split": "test",
        "shard_path": shard_path,
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": 3,
        "token_count": 4,
        "cache_root": str(root),
        "metadata": {"tensor_key": "hidden_states", "t0_token_index": -1},
    }


def _bundle(root, *, variant: bool) -> dict[str, object]:
    prompts = [
        {"prompt_id": "p01", "prompt_cache": {"cache_key": "p01"}},
        {"prompt_id": "p02", "prompt_cache": {"cache_key": "p02"}},
    ]
    views: dict[str, object] = {}
    for condition in ("M1", "M2", "M12"):
        p01 = _prompted_state(root, "sample-1", condition, "p01", [1.0, 0.0, 0.0])
        p02_vector = [0.0, 1.0, 0.0] if variant else [1.0, 0.0, 0.0]
        p02 = _prompted_state(root, "sample-1", condition, "p02", p02_vector)
        views[condition] = {
            "state_cache": _state_cache(root, "sample-1", condition),
            "prompts": {
                "p01": {"prompt_conditioned_state": p01},
                "p02": {"prompt_conditioned_state": p02},
            },
        }
    return {
        "sample_id": "sample-1",
        "sample_type": "Conflict",
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "prompt_set_key": "vt_primary_v1",
        "prompts": prompts,
        "views": views,
    }


def _embedding_row(tmp_path, *, variant: bool) -> dict[str, object]:
    bundle_manifest = tmp_path / f"bundle-{variant}.jsonl"
    write_jsonl(bundle_manifest, [_bundle(tmp_path, variant=variant)])
    result = build_embedding_manifest(
        bundle_manifest_path=bundle_manifest,
        repr_key="raw_layernorm_mean",
        output_dir=tmp_path / f"embeddings-{variant}",
    )
    return json.loads(result.manifest_path.read_text(encoding="utf-8").splitlines()[0])


def test_s_is_nonzero_when_prompt_conditioned_states_differ(tmp_path) -> None:
    scores = compute_s_for_bundle(_embedding_row(tmp_path, variant=True))

    assert scores["S_M1"] > 0
    assert scores["S_M2"] > 0
    assert scores["S_M12"] > 0


def test_s_is_zero_when_prompt_conditioned_states_are_identical(tmp_path) -> None:
    scores = compute_s_for_bundle(_embedding_row(tmp_path, variant=False))

    assert scores["S_M1"] == pytest.approx(0.0)
