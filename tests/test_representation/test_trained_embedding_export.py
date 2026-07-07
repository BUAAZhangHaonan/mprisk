from __future__ import annotations

import json

import numpy as np
import torch
from safetensors.numpy import save_file

from mprisk.data.manifests import write_jsonl
from mprisk.representation.export import export_trained_embeddings
from mprisk.representation.trajectory_model import MLPProjection
from scripts.compute_sdr_scores import compute_sdr_scores


def _prompted_state(root, sample_id: str, condition: str, prompt_id: str, value: float) -> dict:
    shard_path = f"prompted/{sample_id}-{condition}-{prompt_id}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.zeros((1, 2, 3, 3), dtype=np.float32)
    hidden_states[0, :, -1, :] = np.asarray([value, value + 0.1, value + 0.2], dtype=np.float32)
    save_file({"hidden_states": hidden_states}, shard)
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": condition,
        "prompt_set_key": "vt_primary_v1",
        "prompt_id": prompt_id,
        "shard_path": shard_path,
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": 3,
        "token_count": 3,
        "t0_token_index": -1,
        "cache_root": str(root),
        "metadata": {"tensor_key": "hidden_states"},
    }


def _bundle(root) -> dict:
    prompt_ids = ["p01", "p02"]
    views = {}
    for condition_index, condition in enumerate(("M1", "M2", "M12"), start=1):
        views[condition] = {
            "prompts": {
                prompt_id: {
                    "prompt_conditioned_state": _prompted_state(
                        root,
                        "sample-1",
                        condition,
                        prompt_id,
                        float(condition_index + prompt_index),
                    )
                }
                for prompt_index, prompt_id in enumerate(prompt_ids)
            }
        }
    return {
        "sample_id": "sample-1",
        "sample_type": "Conflict",
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "prompt_set_key": "vt_primary_v1",
        "prompts": [{"prompt_id": prompt_id} for prompt_id in prompt_ids],
        "views": views,
    }


def _checkpoint(path) -> None:
    model = MLPProjection(input_dim=3, embed_dim=4, hidden_dim=8, dropout=0.0)
    torch.save(
        {
            "repr_key": "tme_supcon_v1",
            "model_config": {
                "input_dim": 3,
                "embed_dim": 4,
                "hidden_dim": 8,
                "dropout": 0.0,
                "pooling": "mean",
                "normalize_output": True,
            },
            "model_state_dict": model.state_dict(),
            "label_to_id": {"negative": 0, "positive": 1},
        },
        path,
    )


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_export_trained_embeddings_matches_raw_manifest_shape_and_feeds_sdr(tmp_path) -> None:
    bundle_manifest = tmp_path / "bundle_manifest.jsonl"
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_dir = tmp_path / "representation/qwen3_vl_8b/VT/vt_primary_v1/tme_supcon_v1"
    write_jsonl(bundle_manifest, [_bundle(tmp_path)])
    _checkpoint(checkpoint_path)

    result = export_trained_embeddings(
        bundle_manifest_path=bundle_manifest,
        checkpoint_path=checkpoint_path,
        repr_key="tme_supcon_v1",
        output_dir=output_dir,
    )

    rows = _read_jsonl(result.manifest_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.count == 1
    assert rows[0]["repr_key"] == "tme_supcon_v1"
    assert set(rows[0]["embeddings"]) == {"M1", "M2", "M12"}
    for view_embeddings in rows[0]["embeddings"].values():
        assert set(view_embeddings) == {"p01", "p02"}
        assert {len(embedding) for embedding in view_embeddings.values()} == {4}
        for embedding in view_embeddings.values():
            assert np.isfinite(np.asarray(embedding)).all()
    assert summary["embedding_dim"] == 4

    sdr = compute_sdr_scores(embedding_manifest_path=result.manifest_path, output_dir=tmp_path / "sdr")
    assert _read_jsonl(sdr.scores_path)[0]["sample_id"] == "sample-1"
