from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

from mprisk.utils.io import write_jsonl
from scripts.run_representation_training_smoke import run_representation_training_smoke


def _prompted_state(
    root,
    sample_id: str,
    view_key: str,
    prompt_id: str,
    vector: list[float],
) -> dict[str, object]:
    shard_path = f"prompt_conditioned/{sample_id}-{view_key}-{prompt_id}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.zeros((1, 2, 3, len(vector)), dtype=np.float32)
    hidden_states[0, :, -1, :] = np.asarray(vector, dtype=np.float32)
    save_file({"hidden_states": hidden_states}, shard)
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": view_key,
        "prompt_set_key": "vt_primary_v1",
        "prompt_id": prompt_id,
        "shard_path": shard_path,
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": len(vector),
        "token_count": 3,
        "t0_token_index": -1,
        "cache_root": str(root),
        "metadata": {"tensor_key": "hidden_states"},
    }


def _bundle(
    root, sample_id: str, sample_type: str, offset: float, master_split: str
) -> dict[str, object]:
    prompt_ids = ["vt_primary_v1_t01", "vt_primary_v1_t02"]
    base_vectors = {
        "M1": [1.0 + offset, 0.0, 0.1],
        "M2": [0.0, 1.0 + offset, 0.1],
        "M12": [0.2, 0.2 + offset, 1.0],
    }
    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "prompt_set_key": "vt_primary_v1",
        "view_labels": {
            "M1": {"label": "positive", "specific_affect": "joy", "is_clear": True},
            "M2": {"label": "negative", "specific_affect": "anger", "is_clear": True},
            "M12": {"label": "neutral", "specific_affect": "calm", "is_clear": True},
        },
        "prompts": [{"prompt_id": prompt_id} for prompt_id in prompt_ids],
        "views": {
            view_key: {
                "prompts": {
                    prompt_id: {
                        "prompt_id": prompt_id,
                        "prompt_conditioned_state": _prompted_state(
                            root,
                            sample_id,
                            view_key,
                            prompt_id,
                            [value + prompt_index * 0.01 for value in vector],
                        ),
                    }
                    for prompt_index, prompt_id in enumerate(prompt_ids)
                }
            }
            for view_key, vector in base_vectors.items()
        },
        "metadata": {
            "source_dataset": "fake",
            "split_group_id": sample_id,
            "master_split": master_split,
            "representation_split": (
                "relation_val" if master_split == "val" else "relation_train"
            ),
            "calibration_split": "",
            "split_assignment_key": "fixture_v1",
            "split_assignment_sha256": "a" * 64,
        },
    }


def _read_jsonl(path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_representation_training_smoke_trains_exports_and_assigns_patterns(tmp_path) -> None:
    bundle_manifest = tmp_path / "bundle_manifest.jsonl"
    config_path = tmp_path / "train.yaml"
    output_root = tmp_path / "run"
    write_jsonl(
        bundle_manifest,
        [
            _bundle(
                tmp_path,
                f"sample-{index}",
                "Conflict" if index % 2 else "Aligned",
                index * 0.02,
                "val" if index >= 6 else "train",
            )
            for index in range(8)
        ],
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "mprisk_representation_training_v3",
                "key": "qwen3_vl_8b_tme_proxy_anchor_v1",
                "architecture_version": "layer_l2_gru_linear_relation_v1",
                "repr_key": "tme_proxy_anchor_v1",
                "model_key": "qwen3_vl_8b",
                "protocol": "vt",
                "classification_objective": "proxy_anchor_only",
                "prompt_set_key": "vt_primary_v1",
                "prompt_set_artifact_sha256": "b" * 64,
                "expected_prompt_count": 2,
                "expected_prompt_ids": ["vt_primary_v1_t01", "vt_primary_v1_t02"],
                "hidden_dim": 8,
                "condition_dim": 4,
                "relation_dim": 3,
                "dropout": 0.0,
                "max_epochs": 1,
                "batch_size": 4,
                "lr": 0.01,
                "weight_decay": 0.0,
                "proxy_alpha": 8.0,
                "proxy_margin": 0.1,
                "patience": 2,
                "min_delta": 0.0,
                "seed": 123,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity-bound calibration"):
        run_representation_training_smoke(
            bundle_manifest_path=bundle_manifest,
            config_path=config_path,
            model_key="qwen3_vl_8b",
            protocol="VT",
            prompt_set_key="vt_primary_v1",
            output_root=output_root,
            thresholds={"kappa": 0.5, "tau": 0.01},
        )
