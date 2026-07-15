from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from mprisk.representation import training as training_impl
from mprisk.representation.relation_models import (
    SINGLE_POINT_BINARY_V1,
    TRAJECTORY_MLP_BINARY_V1,
    build_representation_model,
)
from mprisk.representation.training import (
    TrainingConfig,
    export_frozen_baseline_representations,
)


def _condition_row(
    root: Path,
    *,
    sample_id: str,
    condition: str,
    prompt_id: str,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": condition,
        "prompt_set_key": "vt_primary_v1",
        "prompt_id": prompt_id,
        "shard_path": f"cache/{sample_id}-{condition}-{prompt_id}.safetensors",
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": 3,
        "token_count": 1,
        "t0_token_index": -1,
        "cache_root": str(root),
        "metadata": {"tensor_key": "hidden_states"},
    }


def _official_test_dataset(tmp_path: Path) -> Path:
    rows = []
    for sample_id, sample_type in (("sample-a", "Aligned"), ("sample-c", "Conflict")):
        for prompt_index in range(8):
            prompt_id = f"p{prompt_index + 1:02d}"
            rows.append(
                {
                    "schema": "mprisk_relation_sample_v1",
                    "row_id": f"{sample_id}:{prompt_id}",
                    "sample_id": sample_id,
                    "sample_type": sample_type,
                    "label_id": int(sample_type == "Conflict"),
                    "model_key": "qwen3_vl_8b",
                    "protocol": "VT",
                    "prompt_set_key": "vt_primary_v1",
                    "prompt_id": prompt_id,
                    "split_group_id": f"group-{sample_id}",
                    "master_split": "test",
                    "representation_split": "official_test",
                    "calibration_split": "",
                    "split_assignment_key": "fixture_v1",
                    "split_assignment_sha256": "a" * 64,
                    "conditions": {
                        condition: _condition_row(
                            tmp_path,
                            sample_id=sample_id,
                            condition=condition,
                            prompt_id=prompt_id,
                        )
                        for condition in ("M1", "M2", "M12")
                    },
                }
            )
    path = tmp_path / "official-test.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _checkpoint(tmp_path: Path, repr_key: str) -> Path:
    config = TrainingConfig(
        repr_key=repr_key,
        model_key="qwen3_vl_8b",
        hidden_dim=6,
        dropout=0.0,
        batch_size=3,
    )
    model = build_representation_model(
        repr_key,
        input_dim=3,
        layer_count=2,
        hidden_dim=6,
        dropout=0.0,
    )
    path = tmp_path / f"{repr_key}.pt"
    torch.save(
        {
            "schema": "mprisk_representation_checkpoint_v2",
            "repr_key": repr_key,
            "model_key": "qwen3_vl_8b",
            "model_config": {"input_dim": 3, "layer_count": 2},
            "training_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "proxy_state_dict": None,
        },
        path,
    )
    return path


@pytest.mark.parametrize("repr_key", [SINGLE_POINT_BINARY_V1, TRAJECTORY_MLP_BINARY_V1])
def test_baselines_expose_fixed_penultimate_features(repr_key: str) -> None:
    model = build_representation_model(
        repr_key,
        input_dim=3,
        layer_count=2,
        hidden_dim=6,
        dropout=0.0,
    )
    trajectories = torch.randn(4, 3, 2, 3)

    features = model.forward_features(trajectories)
    logits = model(trajectories)

    assert model.penultimate_dim == 6
    assert features.shape == (4, 6)
    assert logits.shape == (4, 2)
    torch.testing.assert_close(logits, model.classifier(features))
    assert not hasattr(model, "proxies")


@pytest.mark.parametrize("repr_key", [SINGLE_POINT_BINARY_V1, TRAJECTORY_MLP_BINARY_V1])
def test_baseline_export_streams_and_aggregates_held_out_prompts_once(
    tmp_path: Path, monkeypatch, repr_key: str
) -> None:
    dataset = _official_test_dataset(tmp_path)
    calls = []

    def fake_extract(entry):
        calls.append((entry.sample_id, entry.condition, entry.shard_path))
        offset = 0.0 if entry.sample_id == "sample-a" else 1.0
        return np.full((2, 3), offset + len(calls) / 1000.0, dtype=np.float32)

    monkeypatch.setattr(training_impl, "extract_t0_trajectory", fake_extract)
    result = export_frozen_baseline_representations(
        dataset_path=dataset,
        checkpoint_path=_checkpoint(tmp_path, repr_key),
        output_dir=tmp_path / f"export-{repr_key}",
        representation_split="official_test",
    )

    rows = [json.loads(line) for line in result.manifest_path.read_text().splitlines()]
    summary = json.loads(result.summary_path.read_text())
    assert len(calls) == 2 * 8 * 3
    assert [row["sample_id"] for row in rows] == ["sample-a", "sample-c"]
    assert all(row["prompt_count"] == 8 for row in rows)
    assert all(row["representation_split"] == "official_test" for row in rows)
    assert all(len(row["penultimate_feature"]) == 6 for row in rows)
    assert all(len(row["mean_logits"]) == 2 for row in rows)
    assert all("misread" not in json.dumps(row).casefold() for row in rows)
    assert summary["sample_count"] == 2
    assert summary["feature_dim"] == 6
    assert summary["aggregation"] == "mean_over_synchronized_prompts"
