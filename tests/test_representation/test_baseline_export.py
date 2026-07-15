from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

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
        prompt_set_key="vt_primary_v1",
        prompt_set_artifact_sha256="b" * 64,
        expected_prompt_count=8,
        expected_prompt_ids=tuple(f"p{index:02d}" for index in range(1, 9)),
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
            "architecture_version": repr_key,
            "model_key": "qwen3_vl_8b",
            "model_config": {"input_dim": 3, "layer_count": 2},
            "training_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "proxy_state_dict": None,
        },
        path,
    )
    return path


@pytest.mark.parametrize(
    ("repr_key", "expected_dim"),
    [(SINGLE_POINT_BINARY_V1, 9), (TRAJECTORY_MLP_BINARY_V1, 6)],
)
def test_baselines_expose_locked_frozen_features(
    repr_key: str, expected_dim: int
) -> None:
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

    assert model.penultimate_dim == expected_dim
    assert features.shape == (4, expected_dim)
    assert logits.shape == (4, 2)
    torch.testing.assert_close(logits, model.classifier(features))
    assert not hasattr(model, "proxies")


def test_single_point_feature_is_unprojected_three_condition_concat() -> None:
    model = build_representation_model(
        SINGLE_POINT_BINARY_V1,
        input_dim=3,
        layer_count=2,
        hidden_dim=128,
        dropout=0.5,
    )
    trajectories = torch.arange(36, dtype=torch.float32).reshape(2, 3, 2, 3)

    expected = trajectories[:, :, -1, :].flatten(start_dim=1)

    torch.testing.assert_close(model.forward_features(trajectories), expected)
    assert model.classifier.in_features == 9
    assert not hasattr(model, "feature_projection")


@pytest.mark.parametrize(
    "repr_key", [SINGLE_POINT_BINARY_V1, TRAJECTORY_MLP_BINARY_V1]
)
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
    expected_dim = 9 if repr_key == SINGLE_POINT_BINARY_V1 else 6
    assert all(len(row["penultimate_feature"]) == expected_dim for row in rows)
    assert all(len(row["mean_logits"]) == 2 for row in rows)
    expected_definition = (
        "mean_prompt_final_layer_m1_m2_m12_concat"
        if repr_key == SINGLE_POINT_BINARY_V1
        else "mean_prompt_first_linear_gelu_hidden"
    )
    assert all(row["feature_definition"] == expected_definition for row in rows)
    assert all("misread" not in json.dumps(row).casefold() for row in rows)
    assert summary["sample_count"] == 2
    assert summary["feature_dim"] == expected_dim
    assert summary["aggregation"] == "mean_over_synchronized_prompts"
    assert summary["feature_definition"] == expected_definition


def test_baseline_export_rejects_seven_prompt_samples(tmp_path: Path) -> None:
    dataset = _official_test_dataset(tmp_path)
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    dataset.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in rows
            if not (row["sample_id"] == "sample-c" and row["prompt_id"] == "p08")
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"sample-c.*exactly 8 prompts.*found 7"):
        export_frozen_baseline_representations(
            dataset_path=dataset,
            checkpoint_path=_checkpoint(tmp_path, TRAJECTORY_MLP_BINARY_V1),
            output_dir=tmp_path / "export-seven",
        )


def test_single_point_export_rejects_hidden_projection_checkpoint_drift(
    tmp_path: Path,
) -> None:
    checkpoint_path = _checkpoint(tmp_path, SINGLE_POINT_BINARY_V1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["architecture_version"] = SINGLE_POINT_BINARY_V1
    checkpoint["model_state_dict"]["feature_projection.weight"] = torch.randn(6, 9)
    checkpoint["model_state_dict"]["feature_projection.bias"] = torch.randn(6)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="Single-Point checkpoint architecture drift"):
        export_frozen_baseline_representations(
            dataset_path=_official_test_dataset(tmp_path),
            checkpoint_path=checkpoint_path,
            output_dir=tmp_path / "export-drift",
        )


@pytest.mark.parametrize(
    "config_name",
    [
        "representation_qwen2_5_omni_7b_single_point_v1.yaml",
        "representation_qwen3_vl_8b_single_point_v1.yaml",
        "representation_internvl3_5_8b_single_point_v1.yaml",
    ],
)
def test_single_point_configs_pin_direct_linear_architecture(config_name: str) -> None:
    config_path = Path("configs/experiments") / config_name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["repr_key"] == SINGLE_POINT_BINARY_V1
    assert payload["architecture_version"] == SINGLE_POINT_BINARY_V1
