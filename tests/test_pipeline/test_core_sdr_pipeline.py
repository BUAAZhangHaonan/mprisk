from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from mprisk.data.manifests import write_jsonl
from mprisk.prompts.prompt_cache_builder import prompt_cache_key
from mprisk.representation.relation_models import (
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
    SphericalTMEV1,
)
from mprisk.representation.training import TrainingConfig
from scripts.run_core_sdr_pipeline import run_core_sdr_pipeline


def _manifest_row(sample_id: str, sample_type: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_dataset": "fake_dataset",
        "source_id": f"{sample_id}-source",
        "protocol": "VT",
        "sample_type": sample_type,
        "split_group_id": sample_id,
        "split": "train",
        "media_paths": {"vision": "video.mp4", "text": "text.txt"},
        "views": {
            "M1": {
                "modality": "vision",
                "label": "positive",
                "specific_affect": "joy",
                "is_clear": True,
            },
            "M2": {
                "modality": "text",
                "label": "negative",
                "specific_affect": "anger",
                "is_clear": True,
            },
            "M12": {
                "modality": "vision+text",
                "label": "negative",
                "specific_affect": "frustration",
                "is_clear": True,
            },
        },
        "dominant_modality": "M2",
        "use_in_main": True,
    }


def _full_cache_entry(root, sample_id: str, condition: str) -> dict[str, object]:
    shard_path = f"outputs/full_cache/shards/{sample_id}-{condition}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.full((1, 2, 4, 3), 1.0, dtype=np.float32)
    save_file({"hidden_states": hidden_states}, shard)
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": condition,
        "dataset_key": "fake_dataset",
        "split": "test",
        "shard_path": shard_path,
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": 3,
        "token_count": 4,
        "metadata": {"t0_token_index": -1},
    }


def _write_full_cache_manifest(root, sample_ids: list[str]) -> None:
    manifest = root / "outputs/full_cache/manifests/unified_full_cache_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        _full_cache_entry(root, sample_id, condition)
        for sample_id in sample_ids
        for condition in ("M1", "M2", "M12")
    ]
    manifest.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    (manifest.parent / "extraction_ledger.csv").write_text("", encoding="utf-8")


def _prompt_set(path) -> None:
    path.write_text(
        """
schema: mprisk_equiv_prompt_set_v1
key: vt_primary_v1
protocol: vt
version: v1
active: true
templates:
  - prompt_id: vt_primary_v1_t01
    role: user
    enabled: true
    template_text: "Prompt one {sample_text}"
  - prompt_id: vt_primary_v1_t02
    role: user
    enabled: true
    template_text: "Prompt two {sample_text}"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _prompt_cache_row(prompt_id: str) -> dict[str, str]:
    return {
        "model_key": "qwen3_vl_8b",
        "prompt_set_key": "vt_primary_v1",
        "prompt_id": prompt_id,
        "protocol": "vt",
        "cache_key": prompt_cache_key(
            "qwen3_vl_8b",
            prompt_id,
            prompt_set_key="vt_primary_v1",
            protocol="vt",
        ),
    }


def _prompted_entry(
    root,
    sample_id: str,
    condition: str,
    prompt_id: str,
    value: float,
) -> dict[str, object]:
    shard_path = f"outputs/prompt_conditioned/{sample_id}-{condition}-{prompt_id}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.zeros((1, 2, 4, 3), dtype=np.float32)
    hidden_states[0, :, -1, :] = np.asarray([value, value + 0.1, value + 0.2], dtype=np.float32)
    save_file({"hidden_states": hidden_states}, shard)
    return {
        "sample_id": sample_id,
        "sample_type": "Conflict",
        "model_key": "qwen3_vl_8b",
        "protocol": "vt",
        "condition": condition,
        "prompt_set_key": "vt_primary_v1",
        "prompt_id": prompt_id,
        "shard_path": shard_path,
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": 3,
        "token_count": 4,
        "t0_token_index": -1,
        "cache_root": str(root),
        "metadata": {"tensor_key": "hidden_states"},
    }


def _prompted_rows(root, sample_ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index, sample_id in enumerate(sample_ids, start=1):
        for condition_index, condition in enumerate(("M1", "M2", "M12"), start=1):
            rows.append(
                _prompted_entry(
                    root,
                    sample_id,
                    condition,
                    "vt_primary_v1_t01",
                    float(sample_index + condition_index),
                )
            )
            rows.append(
                _prompted_entry(
                    root,
                    sample_id,
                    condition,
                    "vt_primary_v1_t02",
                    float(sample_index + condition_index + 1),
                )
            )
    return rows


def _read_jsonl(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _split_assignment(path, sample_ids: list[str]):
    write_jsonl(
        path,
        [
            {
                "schema": "mprisk_representation_split_assignment_v1",
                "config_key": "fixture_v1",
                "split_group_id": sample_id,
                "master_split": "train",
                "representation_split": "relation_train",
                "sample_ids": [sample_id],
                "sample_count": 1,
                "protocols": ["VT"],
                "source_datasets": ["fake_dataset"],
            }
            for sample_id in sample_ids
        ],
    )
    return path


def test_core_sdr_pipeline_rejects_raw_layernorm_as_final_representation(tmp_path) -> None:
    with pytest.raises(ValueError, match="raw_layernorm representations cannot stand in"):
        run_core_sdr_pipeline(
            model_key="qwen3_vl_8b",
            protocol="VT",
            prompt_set_key="vt_primary_v1",
            repr_key="raw_layernorm_mean",
            manifest_paths=[tmp_path / "missing.jsonl"],
            full_cache_root=tmp_path,
            prompt_cache_manifest=tmp_path / "prompt_cache_manifest.jsonl",
            prompt_conditioned_cache_manifest=tmp_path / "prompted_manifest.jsonl",
            prompt_set=tmp_path / "vt_primary_v1.yaml",
            split_assignment=tmp_path / "missing-split.jsonl",
            output_root=tmp_path,
            thresholds={"kappa": 0.5, "tau": 0.01},
        )


def test_core_sdr_pipeline_requires_checkpoint_for_tme_repr(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --checkpoint"):
        run_core_sdr_pipeline(
            model_key="qwen3_vl_8b",
            protocol="VT",
            prompt_set_key="vt_primary_v1",
            repr_key=TME_PROXY_ANCHOR_V1,
            manifest_paths=[tmp_path / "missing.jsonl"],
            full_cache_root=tmp_path,
            prompt_cache_manifest=tmp_path / "prompt_cache_manifest.jsonl",
            prompt_conditioned_cache_manifest=tmp_path / "prompted_manifest.jsonl",
            prompt_set=tmp_path / "vt_primary_v1.yaml",
            split_assignment=tmp_path / "missing-split.jsonl",
            output_root=tmp_path,
            thresholds={"kappa": 0.5, "tau": 0.01},
        )


def test_core_sdr_pipeline_rejects_unbound_thresholds_after_tme_export(tmp_path) -> None:
    labels = tmp_path / "manifests/final_manifest.jsonl"
    labels.parent.mkdir(parents=True)
    sample_ids = ["sample-conflict"]
    write_jsonl(labels, [_manifest_row("sample-conflict", "Conflict")])
    _write_full_cache_manifest(tmp_path, sample_ids)

    prompt_set = tmp_path / "vt_primary_v1.yaml"
    prompt_cache_manifest = tmp_path / "prompt_cache_manifest.jsonl"
    prompted_manifest = tmp_path / "prompt_conditioned_manifest.jsonl"
    checkpoint = tmp_path / "checkpoint.pt"
    _prompt_set(prompt_set)
    write_jsonl(
        prompt_cache_manifest,
        [_prompt_cache_row("vt_primary_v1_t01"), _prompt_cache_row("vt_primary_v1_t02")],
    )
    write_jsonl(prompted_manifest, _prompted_rows(tmp_path, sample_ids))
    model = SphericalTMEV1(
        input_dim=3,
        sequence_hidden_dim=8,
        condition_dim=4,
        relation_dim=3,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        repr_key=TME_PROXY_ANCHOR_V1,
        model_key="qwen3_vl_8b",
        protocol="vt",
        classification_objective="proxy_anchor_only",
        prompt_set_key="vt_primary_v1",
        prompt_set_artifact_sha256=hashlib.sha256(prompt_set.read_bytes()).hexdigest(),
        expected_prompt_count=2,
        expected_prompt_ids=("vt_primary_v1_t01", "vt_primary_v1_t02"),
        hidden_dim=8,
        condition_dim=4,
        relation_dim=3,
        dropout=0.0,
        d_supervision_weight=0.2,
        d_ranking_margin=0.25,
        angular_supervision_weight=0.2,
        angular_ranking_margin_rad=0.08726646259971647,
        d_aux_samples_per_class=1,
    )
    torch.save(
        {
            "repr_key": TME_PROXY_ANCHOR_V1,
            "architecture_version": TME_ARCHITECTURE_V1,
            "model_config": {"input_dim": 3, "layer_count": 2},
            "training_config": asdict(training_config),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="identity-bound calibration"):
        run_core_sdr_pipeline(
            model_key="qwen3_vl_8b",
            protocol="VT",
            prompt_set_key="vt_primary_v1",
            repr_key=TME_PROXY_ANCHOR_V1,
            manifest_paths=[labels],
            full_cache_root=tmp_path,
            prompt_cache_manifest=prompt_cache_manifest,
            prompt_conditioned_cache_manifest=prompted_manifest,
            prompt_set=prompt_set,
            split_assignment=_split_assignment(tmp_path / "split.jsonl", sample_ids),
            output_root=tmp_path,
            thresholds={"kappa": 0.5, "tau": 0.01},
            checkpoint=checkpoint,
        )
