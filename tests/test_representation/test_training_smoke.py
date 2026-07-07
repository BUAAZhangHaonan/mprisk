from __future__ import annotations

import json

import numpy as np
import torch
import yaml
from safetensors.numpy import save_file


def _write_prompted_state(tmp_path, sample_id: str, prompt_id: str, vector: list[float]) -> dict:
    shard_path = f"prompted/{sample_id}-{prompt_id}.safetensors"
    shard = tmp_path / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.zeros((1, 2, 3, len(vector)), dtype=np.float32)
    hidden_states[0, :, -1, :] = np.asarray(vector, dtype=np.float32)
    save_file({"hidden_states": hidden_states}, shard)
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": "M1",
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
        ("sample-1", "safe", [1.0, 0.0, 0.0]),
        ("sample-2", "safe", [0.9, 0.1, 0.0]),
        ("sample-3", "risk", [0.0, 1.0, 0.0]),
        ("sample-4", "risk", [0.0, 0.9, 0.1]),
    ]
    for sample_id, label, vector in examples:
        for prompt_index in range(2):
            prompt_id = f"prompt-{prompt_index + 1}"
            rows.append(
                {
                    "row_id": f"{sample_id}-{prompt_id}",
                    "sample_id": sample_id,
                    "sample_type": "Conflict" if label == "risk" else "Aligned",
                    "model_key": "qwen3_vl_8b",
                    "protocol": "VT",
                    "view_key": "M1",
                    "prompt_id": prompt_id,
                    "prompt_set_key": "vt_primary_v1",
                    "label": label,
                    "specific_affect": "neutral",
                    "is_clear": True,
                    "prompt_conditioned_state": _write_prompted_state(
                        tmp_path,
                        sample_id,
                        prompt_id,
                        [value + prompt_index * 0.01 for value in vector],
                    ),
                    "split_group_id": sample_id,
                    "source_dataset": "fake",
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
                "embed_dim": 4,
                "hidden_dim": 8,
                "dropout": 0.0,
                "epochs": 1,
                "batch_size": 4,
                "lr": 0.01,
                "lambda_prompt": 0.5,
                "temperature": 0.2,
                "negative_budget_ratio": 1.0,
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
            ]
        )
        == 0
    )

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu")
    assert checkpoint["repr_key"] == "tme_supcon_v1"
    assert checkpoint["model_config"]["embed_dim"] == 4
    assert checkpoint["label_to_id"] == {"risk": 0, "safe": 1}
    assert "model_state_dict" in checkpoint
    metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
    assert metrics["repr_key"] == "tme_supcon_v1"
    assert metrics["epochs"] == 1
    assert (output_dir / "train_config.yaml").exists()
    assert (output_dir / "train_log.jsonl").read_text(encoding="utf-8").strip()
