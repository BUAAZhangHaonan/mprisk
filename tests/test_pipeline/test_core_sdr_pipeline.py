from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from mprisk.data.manifests import write_jsonl
from mprisk.prompts.prompt_cache_builder import prompt_cache_key
from mprisk.representation.trajectory_model import MLPProjection
from scripts.run_core_sdr_pipeline import run_core_sdr_pipeline


def _manifest_row(sample_id: str, sample_type: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_dataset": "fake_dataset",
        "source_id": f"{sample_id}-source",
        "protocol": "VT",
        "sample_type": sample_type,
        "split_group_id": sample_id,
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
    hidden_states[0, :, -1, :] = np.asarray(
        [value, value + 0.1, value + 0.2], dtype=np.float32
    )
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


def test_core_sdr_pipeline_runs_from_fake_manifests_and_writes_summary(tmp_path) -> None:
    labels = tmp_path / "manifests/final_manifest.jsonl"
    labels.parent.mkdir(parents=True)
    sample_ids = ["sample-conflict", "sample-aligned"]
    write_jsonl(
        labels,
        [
            _manifest_row("sample-conflict", "Conflict"),
            _manifest_row("sample-aligned", "Aligned"),
        ],
    )
    _write_full_cache_manifest(tmp_path, sample_ids)

    prompt_set = tmp_path / "vt_primary_v1.yaml"
    prompt_cache_manifest = tmp_path / "prompt_cache_manifest.jsonl"
    prompted_manifest = tmp_path / "prompt_conditioned_manifest.jsonl"
    _prompt_set(prompt_set)
    write_jsonl(
        prompt_cache_manifest,
        [_prompt_cache_row("vt_primary_v1_t01"), _prompt_cache_row("vt_primary_v1_t02")],
    )
    write_jsonl(prompted_manifest, _prompted_rows(tmp_path, sample_ids))

    result = run_core_sdr_pipeline(
        model_key="qwen3_vl_8b",
        protocol="VT",
        prompt_set_key="vt_primary_v1",
        repr_key="raw_layernorm_mean",
        manifest_paths=[labels],
        full_cache_root=tmp_path,
        prompt_cache_manifest=prompt_cache_manifest,
        prompt_conditioned_cache_manifest=prompted_manifest,
        prompt_set=prompt_set,
        output_root=tmp_path,
        thresholds={"kappa": 0.5, "tau": 0.01, "delta": 0.2},
    )

    expected_output_dir = (
        tmp_path / "outputs/states/qwen3_vl_8b/VT/vt_primary_v1/raw_layernorm_mean"
    )
    assert result.sdr_scores_path == expected_output_dir / "sdr_scores.jsonl"
    assert result.state_patterns_path == expected_output_dir / "state_patterns.jsonl"
    assert result.state_summary_path == expected_output_dir / "state_summary.json"
    assert result.core_summary_path == expected_output_dir / "CORE_SDR_SUMMARY.md"

    scores = _read_jsonl(result.sdr_scores_path)
    patterns = _read_jsonl(result.state_patterns_path)
    summary_text = result.core_summary_path.read_text(encoding="utf-8")

    assert [row["sample_id"] for row in scores] == sample_ids
    assert [row["sample_id"] for row in patterns] == sample_ids
    assert "State counts" in summary_text
    assert "Output paths" in summary_text
    assert "Conflict samples: 1" in summary_text
    assert "Aligned samples: 1" in summary_text
    assert str(result.sdr_scores_path) in summary_text
    assert str(result.state_patterns_path) in summary_text


def test_core_sdr_pipeline_requires_checkpoint_for_tme_repr(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --checkpoint"):
        run_core_sdr_pipeline(
            model_key="qwen3_vl_8b",
            protocol="VT",
            prompt_set_key="vt_primary_v1",
            repr_key="tme_supcon_v1",
            manifest_paths=[tmp_path / "missing.jsonl"],
            full_cache_root=tmp_path,
            prompt_cache_manifest=tmp_path / "prompt_cache_manifest.jsonl",
            prompt_conditioned_cache_manifest=tmp_path / "prompted_manifest.jsonl",
            prompt_set=tmp_path / "vt_primary_v1.yaml",
            output_root=tmp_path,
        )


def test_core_sdr_pipeline_uses_existing_checkpoint_for_tme_repr(tmp_path) -> None:
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
        },
        checkpoint,
    )

    result = run_core_sdr_pipeline(
        model_key="qwen3_vl_8b",
        protocol="VT",
        prompt_set_key="vt_primary_v1",
        repr_key="tme_supcon_v1",
        manifest_paths=[labels],
        full_cache_root=tmp_path,
        prompt_cache_manifest=prompt_cache_manifest,
        prompt_conditioned_cache_manifest=prompted_manifest,
        prompt_set=prompt_set,
        output_root=tmp_path,
        thresholds={"kappa": 0.5, "tau": 0.01, "delta": 0.2},
        checkpoint=checkpoint,
    )

    scores = _read_jsonl(result.sdr_scores_path)

    assert result.sdr_scores_path.name == "sdr_scores.jsonl"
    assert scores[0]["repr_key"] == "tme_supcon_v1"
    assert result.core_summary_path.exists()
