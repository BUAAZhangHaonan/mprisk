from __future__ import annotations

import json

import numpy as np
from safetensors.numpy import save_file

from mprisk.data.manifests import write_jsonl
from mprisk.prompts.prompt_cache_builder import prompt_cache_key
from scripts.run_state_measurement_smoke import run_state_measurement_smoke


def _state_entry(root, sample_id: str, condition: str, value: float) -> dict[str, object]:
    shard_path = f"outputs/full_cache/shards/{sample_id}-{condition}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.full((1, 2, 4, 3), value, dtype=np.float32)
    save_file({"hidden_states": hidden_states}, shard)
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


def _prompted_entry(
    root,
    sample_id: str,
    condition: str,
    prompt_id: str,
    vector: list[float],
) -> dict[str, object]:
    shard_path = f"outputs/prompt_conditioned/{sample_id}-{condition}-{prompt_id}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.zeros((1, 2, 4, 3), dtype=np.float32)
    hidden_states[0, :, -1, :] = np.asarray(vector, dtype=np.float32)
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


def _prompted_rows(root, sample_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in ("M1", "M2", "M12"):
        rows.append(
            _prompted_entry(root, sample_id, condition, "vt_primary_v1_t01", [1.0, 0.0, 0.0])
        )
        rows.append(
            _prompted_entry(root, sample_id, condition, "vt_primary_v1_t02", [0.0, 1.0, 0.0])
        )
    return rows


def _state_row(root, sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "sample_type": "Conflict",
        "source_dataset": "ch_sims_v2",
        "split_group_id": sample_id,
        "master_split": "train",
        "representation_split": "relation_train",
        "calibration_split": "",
        "split_assignment_key": "fixture_v1",
        "split_assignment_sha256": "a" * 64,
        "protocol": "VT",
        "model_key": "qwen3_vl_8b",
        "target_label": "negative",
        "dominant_modality": "M2",
        "view_labels": {
            "M1": {"label": "positive", "specific_affect": "joy", "is_clear": True},
            "M2": {"label": "negative", "specific_affect": "anger", "is_clear": True},
            "M12": {"label": "negative", "specific_affect": "sarcasm", "is_clear": True},
        },
        "m1_entry": _state_entry(root, sample_id, "M1", 1.0),
        "m2_entry": _state_entry(root, sample_id, "M2", 2.0),
        "m12_entry": _state_entry(root, sample_id, "M12", 3.0),
        "trajectory_meta": {"layer_count": 2, "hidden_dim": 3, "t0_token_index": -1},
    }


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


def test_state_measurement_smoke_exports_embeddings_sdr_and_patterns(tmp_path) -> None:
    state_manifest = tmp_path / "state_dataset_manifest.jsonl"
    prompt_set = tmp_path / "vt_primary_v1.yaml"
    prompt_cache_manifest = tmp_path / "prompt_cache_manifest.jsonl"
    prompt_conditioned_manifest = tmp_path / "prompt_conditioned_manifest.jsonl"
    write_jsonl(state_manifest, [_state_row(tmp_path, "sample-1")])
    _prompt_set(prompt_set)
    write_jsonl(
        prompt_cache_manifest,
        [_prompt_cache_row("vt_primary_v1_t01"), _prompt_cache_row("vt_primary_v1_t02")],
    )
    write_jsonl(prompt_conditioned_manifest, _prompted_rows(tmp_path, "sample-1"))

    result = run_state_measurement_smoke(
        state_dataset_manifest_path=state_manifest,
        prompt_cache_manifest_path=prompt_cache_manifest,
        prompt_conditioned_cache_manifest_path=prompt_conditioned_manifest,
        prompt_set_path=prompt_set,
        model_key="qwen3_vl_8b",
        protocol="VT",
        prompt_set_key="vt_primary_v1",
        repr_key="raw_layernorm_mean",
        output_root=tmp_path / "outputs",
        thresholds={"kappa": 0.5, "tau": 0.01, "delta": 0.2},
    )

    assert result.bundle_result.complete_count == 1
    assert result.embedding_count == 1
    assert result.sdr_count == 1
    assert result.pattern_count == 1
    assert result.report_path.name == "STATE_MEASUREMENT_SMOKE.md"

    pattern_rows = [
        json.loads(line)
        for line in result.patterns_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert pattern_rows[0]["sample_id"] == "sample-1"
    assert pattern_rows[0]["pattern"] in {"Confusion", "Consensus", "Balanced", "Dominant"}
    assert pattern_rows[0]["S_M1"] > 0
    assert summary["total_samples"] == 1
    assert result.report_path.exists()
