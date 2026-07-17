from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from mprisk.representation.losses import ProxyAnchorLoss
from mprisk.representation.relation_dataset import build_relation_dataset
from mprisk.representation.relation_models import (
    SINGLE_POINT_BINARY_V1,
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
    TRAJECTORY_MLP_BINARY_V1,
    OrderedLinearRelationV1,
    SequentialTrajectoryEncoderV1,
    SphericalTMEV1,
    build_representation_model,
    ordered_relation_features,
)
from mprisk.representation.training import load_training_config


def _state(sample_id: str, condition: str, prompt_id: str) -> dict[str, object]:
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
        "hidden_dim": 4,
        "token_count": 3,
        "t0_token_index": -1,
        "cache_root": ".",
    }


def _bundle(sample_id: str = "sample-1", sample_type: str = "Conflict") -> dict[str, object]:
    prompt_ids = ["p1", "p2"]
    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "prompt_set_key": "vt_primary_v1",
        "prompts": [{"prompt_id": prompt_id} for prompt_id in prompt_ids],
        "views": {
            condition: {
                "prompts": {
                    prompt_id: {
                        "prompt_conditioned_state": _state(sample_id, condition, prompt_id)
                    }
                    for prompt_id in prompt_ids
                }
            }
            for condition in ("M1", "M2", "M12")
        },
        "metadata": {
            "split_group_id": f"group:{sample_id}",
            "master_split": "train",
            "representation_split": "relation_train",
            "calibration_split": "",
            "split_assignment_key": "fixture_v1",
            "split_assignment_sha256": "a" * 64,
        },
    }


def _prompt_contract() -> dict[str, object]:
    return {
        "prompt_set_key": "vt_primary_v1",
        "prompt_set_artifact_sha256": "b" * 64,
        "expected_prompt_count": 2,
        "expected_prompt_ids": ("p1", "p2"),
    }


def test_relation_dataset_uses_one_sample_label_for_all_three_conditions(tmp_path) -> None:
    source = tmp_path / "bundle.jsonl"
    source.write_text(json.dumps(_bundle()) + "\n", encoding="utf-8")

    result = build_relation_dataset(
        bundle_manifest_path=source,
        output_dir=tmp_path / "out",
        **_prompt_contract(),
    )
    rows = [json.loads(line) for line in result.dataset_path.read_text().splitlines()]
    summary = json.loads(result.summary_path.read_text())

    assert len(rows) == 2
    assert {row["sample_type"] for row in rows} == {"Conflict"}
    assert {row["label_id"] for row in rows} == {1}
    assert all(set(row["conditions"]) == {"M1", "M2", "M12"} for row in rows)
    assert all("view_labels" not in row and "label" not in row for row in rows)
    assert {row["master_split"] for row in rows} == {"train"}
    assert {row["representation_split"] for row in rows} == {"relation_train"}
    assert {row["split_assignment_sha256"] for row in rows} == {"a" * 64}
    assert summary["representation_split_counts"] == {"relation_train": 2}
    assert summary["split_assignment_sha256"] == "a" * 64


def test_relation_dataset_rejects_missing_registered_split(tmp_path) -> None:
    source = tmp_path / "bundle.jsonl"
    bundle = _bundle()
    del bundle["metadata"]["master_split"]
    source.write_text(json.dumps(bundle) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="master_split"):
        build_relation_dataset(
            bundle_manifest_path=source,
            output_dir=tmp_path / "out",
            **_prompt_contract(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "misread",
        "MISREAD",
        "binary_label",
        "final_decision",
        "misread_label",
        "misread_binary_label",
    ],
)
def test_relation_dataset_strictly_rejects_misread_fields(tmp_path, field: str) -> None:
    source = tmp_path / "bundle.jsonl"
    bundle = _bundle()
    bundle["metadata"][field] = 1
    source.write_text(json.dumps(bundle) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Misread fields are forbidden"):
        build_relation_dataset(
            bundle_manifest_path=source,
            output_dir=tmp_path / "out",
            **_prompt_contract(),
        )


def test_sequential_tme_normalizes_every_layer_and_all_outputs() -> None:
    encoder = SequentialTrajectoryEncoderV1(
        input_dim=4,
        sequence_hidden_dim=5,
        embed_dim=3,
        dropout=0.0,
    )
    trajectories = torch.tensor(
        [[[[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 12.0]]] * 3]
    )
    normalized = encoder.normalize_layers(trajectories)
    assert torch.linalg.vector_norm(normalized, dim=-1) == pytest.approx(torch.ones(1, 3, 2))

    model = SphericalTMEV1(
        input_dim=4,
        sequence_hidden_dim=5,
        condition_dim=3,
        relation_dim=2,
        dropout=0.0,
    )
    condition_z, relation_r = model(trajectories)
    assert model.architecture_version == TME_ARCHITECTURE_V1
    assert condition_z.shape == (1, 3, 3)
    assert relation_r.shape == (1, 2)
    assert torch.linalg.vector_norm(condition_z, dim=-1).detach() == pytest.approx(
        torch.ones(1, 3)
    )
    assert torch.linalg.vector_norm(relation_r, dim=-1).detach() == pytest.approx(torch.ones(1))


def test_tme_relation_is_exact_ordered_three_scalar_linear_projection() -> None:
    z1 = torch.tensor([[1.0, 0.0]])
    z2 = torch.tensor([[0.0, 1.0]])
    z12 = torch.tensor([[2**-0.5, 2**-0.5]])
    expected = torch.tensor([[1.0, 1.0 - 2**-0.5, 1.0 - 2**-0.5]])

    assert ordered_relation_features(z1, z2, z12) == pytest.approx(expected)
    relation = OrderedLinearRelationV1(relation_dim=4)
    assert list(relation.children()) == [relation.projection]
    assert relation.projection.in_features == 3


def test_three_representation_interfaces_have_binary_or_proxy_anchor_contracts() -> None:
    trajectories = torch.randn(2, 3, 4, 6)
    single = build_representation_model(
        SINGLE_POINT_BINARY_V1, input_dim=6, layer_count=4, hidden_dim=8
    )
    trajectory = build_representation_model(
        TRAJECTORY_MLP_BINARY_V1, input_dim=6, layer_count=4, hidden_dim=8
    )
    tme = build_representation_model(
        TME_PROXY_ANCHOR_V1,
        input_dim=6,
        layer_count=4,
        hidden_dim=8,
        condition_dim=5,
        relation_dim=4,
    )
    assert single(trajectories).shape == (2, 2)
    assert trajectory(trajectories).shape == (2, 2)
    assert tme(trajectories)[1].shape == (2, 4)


def test_proxy_anchor_is_two_proxy_tme_objective() -> None:
    objective = ProxyAnchorLoss(embed_dim=4, num_classes=2, alpha=16.0, margin=0.1)
    embeddings = torch.nn.functional.normalize(torch.randn(6, 4), dim=-1).requires_grad_()
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = objective(embeddings, labels)
    loss.backward()
    assert objective.proxies.shape == (2, 4)
    assert torch.isfinite(loss)
    assert embeddings.grad is not None


def test_tme_rejects_zero_norm_vectors_at_every_spherical_stage() -> None:
    encoder = SequentialTrajectoryEncoderV1(
        input_dim=3,
        sequence_hidden_dim=4,
        embed_dim=2,
        dropout=0.0,
    )
    zero_layer = torch.ones(1, 3, 2, 3)
    zero_layer[0, 1, 0] = 0.0
    with pytest.raises(ValueError, match=r"stage=tme_layer_input.*sample=sample-0"):
        encoder.normalize_layers(zero_layer, sample_ids=["sample-0"])

    with torch.no_grad():
        encoder.projection.weight.zero_()
        encoder.projection.bias.zero_()
    with pytest.raises(ValueError, match=r"stage=tme_z_projection.*sample=sample-0"):
        encoder(torch.ones(1, 3, 2, 3), sample_ids=["sample-0"])

    relation = OrderedLinearRelationV1(relation_dim=2)
    with torch.no_grad():
        relation.projection.weight.zero_()
        relation.projection.bias.zero_()
    z1 = torch.tensor([[1.0, 0.0]])
    z2 = torch.tensor([[0.0, 1.0]])
    z12 = torch.tensor([[2**-0.5, 2**-0.5]])
    with pytest.raises(
        ValueError, match=r"stage=ordered_relation_r_projection.*sample=sample-0"
    ):
        relation(z1, z2, z12, sample_ids=["sample-0"])

    with pytest.raises(ValueError, match=r"stage=ordered_relation_z1.*sample=sample-0"):
        ordered_relation_features(
            torch.zeros_like(z1), z2, z12, sample_ids=["sample-0"]
        )


def test_proxy_anchor_rejects_zero_embeddings_and_proxies_before_normalization() -> None:
    objective = ProxyAnchorLoss(embed_dim=2, num_classes=2)
    labels = torch.tensor([0, 1])
    with pytest.raises(
        ValueError, match=r"stage=proxy_anchor_embeddings.*sample=sample-a"
    ):
        objective(
            torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
            labels,
            sample_ids=["sample-a", "sample-c"],
        )

    with torch.no_grad():
        objective.proxies[0].zero_()
    with pytest.raises(
        ValueError, match=r"stage=proxy_anchor_proxies.*sample=proxy_class_0"
    ):
        objective(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            labels,
            sample_ids=["sample-a", "sample-c"],
        )


def test_tme_training_config_rejects_architecture_version_drift(tmp_path) -> None:
    config = {
        "schema": "mprisk_representation_training_v4",
        "key": "qwen3-vl-test",
        "architecture_version": "unversioned_gru",
        "repr_key": TME_PROXY_ANCHOR_V1,
        "model_key": "qwen3_vl_8b",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture_version"):
        load_training_config(path)


def test_delivery_tme_configs_separate_pa_only_from_state_supervision(tmp_path) -> None:
    root = Path(__file__).parents[2]
    config_root = root / "configs/experiments/delivery_20260716/seed20260717"
    configs = {path.name: load_training_config(path) for path in config_root.glob("*.yaml")}

    assert len(configs) == 6
    for name, config in configs.items():
        if "pa_only" in name:
            assert config.enable_state_supervision is False
            assert config.d_supervision_weight == 0.0
            assert config.angular_supervision_weight == 0.0
            assert config.d_aux_samples_per_class == 0
        else:
            assert config.enable_state_supervision is True
            assert config.d_supervision_weight > 0.0
            assert config.angular_supervision_weight > 0.0
            assert config.d_aux_samples_per_class > 0

    source = config_root / "qwen3_vl_8b_tme_pa_only_v1.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["d_supervision_weight"] = 0.1
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="PA-only TME requires"):
        load_training_config(invalid)
