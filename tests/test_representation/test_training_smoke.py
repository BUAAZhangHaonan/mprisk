from __future__ import annotations

import json

import numpy as np
import torch
import yaml
from safetensors.numpy import save_file


def _write_prompted_state(
    tmp_path, sample_id: str, condition: str, prompt_id: str, vector: list[float]
) -> dict:
    shard_path = f"prompted/{sample_id}-{condition}-{prompt_id}.safetensors"
    shard = tmp_path / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.zeros((1, 2, 3, len(vector)), dtype=np.float32)
    hidden_states[0, :, -1, :] = np.asarray(vector, dtype=np.float32)
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
        "hidden_dim": len(vector),
        "token_count": 3,
        "t0_token_index": -1,
        "cache_root": str(tmp_path),
        "metadata": {"tensor_key": "hidden_states"},
    }


def _write_representation_dataset(tmp_path) -> tuple[object, object]:
    dataset_path = tmp_path / "representation_dataset.jsonl"
    rows = []
    examples = [
        (f"sample-{index}", "Aligned" if index % 2 == 0 else "Conflict")
        for index in range(8)
    ]
    for index, (sample_id, sample_type) in enumerate(examples):
        for prompt_index in range(2):
            prompt_id = f"prompt-{prompt_index + 1}"
            base = [1.0, 0.1 + index * 0.01, 0.2]
            rows.append(
                {
                    "schema": "mprisk_relation_sample_v1",
                    "row_id": f"{sample_id}:{prompt_id}",
                    "sample_id": sample_id,
                    "sample_type": sample_type,
                    "label_id": int(sample_type == "Conflict"),
                    "model_key": "qwen3_vl_8b",
                    "protocol": "VT",
                    "prompt_id": prompt_id,
                    "prompt_set_key": "vt_primary_v1",
                    "split_group_id": sample_id,
                    "master_split": "val" if index >= 6 else "train",
                    "representation_split": (
                        "relation_val" if index >= 6 else "relation_train"
                    ),
                    "calibration_split": "",
                    "split_assignment_key": "fixture_v1",
                    "split_assignment_sha256": "a" * 64,
                    "conditions": {
                        condition: _write_prompted_state(
                            tmp_path,
                            sample_id,
                            condition,
                            prompt_id,
                            [
                                value + condition_index * 0.02 + prompt_index * 0.01
                                for value in base
                            ],
                        )
                        for condition_index, condition in enumerate(("M1", "M2", "M12"))
                    },
                }
            )
    dataset_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "mprisk_representation_training_v3",
                "key": "qwen3_vl_8b_tme_proxy_anchor_test_v1",
                "architecture_version": "layer_l2_gru_linear_relation_v1",
                "repr_key": "tme_proxy_anchor_v1",
                "model_key": "qwen3_vl_8b",
                "prompt_set_key": "vt_primary_v1",
                "prompt_set_artifact_sha256": "b" * 64,
                "expected_prompt_count": 2,
                "expected_prompt_ids": ["prompt-1", "prompt-2"],
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
    return dataset_path, config_path


def test_train_trajectory_encoder_cli_writes_loadable_artifacts(tmp_path) -> None:
    from scripts.train_trajectory_encoder import main

    dataset_path, config_path = _write_representation_dataset(tmp_path)
    output_dir = tmp_path / "run"

    assert (
        main(
            [
                "--dataset",
                str(dataset_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location="cpu")
    assert checkpoint["repr_key"] == "tme_proxy_anchor_v1"
    assert checkpoint["training_config"]["condition_dim"] == 4
    assert checkpoint["proxy_state_dict"]["proxies"].shape == (2, 3)
    assert "model_state_dict" in checkpoint
    metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
    assert metrics["repr_key"] == "tme_proxy_anchor_v1"
    assert metrics["final_epoch"] == 1
    assert metrics["device"] == "cpu"
    assert (output_dir / "train_config.yaml").exists()
    assert (output_dir / "train_log.jsonl").read_text(encoding="utf-8").strip()
