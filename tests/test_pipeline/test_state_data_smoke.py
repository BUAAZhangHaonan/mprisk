from __future__ import annotations

import json

import numpy as np
from safetensors.numpy import save_file

from mprisk.data.manifests import write_jsonl
from scripts.verify_state_data_pipeline import run_state_data_smoke


def _manifest_row(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_dataset": "ch_sims_v2",
        "source_id": f"{sample_id}-source",
        "protocol": "VT",
        "sample_type": "Aligned",
        "split_group_id": sample_id,
        "media_paths": {"vision": "video.mp4", "text": "text.txt"},
        "views": {
            "M1": {"modality": "vision", "label": "positive", "is_clear": True},
            "M2": {"modality": "text", "label": "positive", "is_clear": True},
            "M12": {"modality": "vision+text", "label": "positive", "is_clear": True},
        },
        "dominant_modality": "balanced",
        "use_in_main": True,
    }


def _cache_entry(root, sample_id: str, condition: str) -> dict[str, object]:
    shard_path = f"outputs/full_cache/shards/{sample_id}-{condition}.safetensors"
    shard = root / shard_path
    shard.parent.mkdir(parents=True, exist_ok=True)
    hidden_states = np.ones((1, 2, 4, 3), dtype=np.float32)
    save_file({"hidden_states": hidden_states}, shard)
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": condition,
        "dataset_key": "ch_sims_v2",
        "split": "test",
        "shard_path": shard_path,
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": 3,
        "token_count": 4,
        "metadata": {"tensor_key": "hidden_states", "t0_token_index": -1},
    }


def test_state_data_smoke_builds_manifest_and_checks_t0_trajectory(tmp_path) -> None:
    label_manifest = tmp_path / "data/processed/manifests/aligned_manifest.jsonl"
    label_manifest.parent.mkdir(parents=True)
    write_jsonl(label_manifest, [_manifest_row("sample-1")])
    cache_manifest = tmp_path / "outputs/full_cache/manifests/unified_full_cache_manifest.json"
    cache_manifest.parent.mkdir(parents=True)
    entries = [_cache_entry(tmp_path, "sample-1", condition) for condition in ("M1", "M2", "M12")]
    cache_manifest.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    (cache_manifest.parent / "extraction_ledger.csv").write_text("", encoding="utf-8")

    result = run_state_data_smoke(
        manifest_paths=[label_manifest],
        cache_root=tmp_path,
        model_key="qwen3_vl_8b",
        protocol="VT",
        output_dir=tmp_path / "outputs/state_data/qwen3_vl_8b/VT",
        reports_dir=tmp_path / "outputs/state_data/reports",
        trajectory_check_limit=1,
    )

    assert result.state_dataset.resolved_count == 1
    assert result.trajectory_checked_rows == 1
    assert result.trajectory_error_rows == 0
    assert result.report_path.name == "STATE_DATA_SMOKE.md"
    assert "sample-1" in result.report_path.read_text(encoding="utf-8")
