from __future__ import annotations

import json

from mprisk.data.manifests import write_jsonl
from mprisk.data.state_bundle import (
    build_state_bundles,
    iter_state_bundles,
    load_state_bundle,
)
from mprisk.prompts.prompt_cache_builder import prompt_cache_key


def _state_entry(sample_id: str, condition: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "vt",
        "condition": condition,
        "dataset_key": "ch_sims_v2",
        "split": "test",
        "shard_path": f"outputs/full_cache/shards/{sample_id}-{condition}.safetensors",
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": 3,
        "token_count": 4,
        "cache_root": ".",
        "checksum": f"{sample_id}-{condition}",
        "metadata": {"t0_token_index": -1},
    }


def _view_labels() -> dict[str, dict[str, object]]:
    return {
        "M1": {"label": "positive", "specific_affect": "joy", "is_clear": True},
        "M2": {"label": "negative", "specific_affect": "anger", "is_clear": True},
        "M12": {"label": "negative", "specific_affect": "frustration", "is_clear": True},
    }


def _state_row(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "sample_type": "Conflict",
        "source_dataset": "ch_sims_v2",
        "split_group_id": sample_id,
        "protocol": "VT",
        "model_key": "qwen3_vl_8b",
        "target_label": "negative",
        "view_labels": _view_labels(),
        "dominant_modality": "M2",
        "m1_entry": _state_entry(sample_id, "M1"),
        "m2_entry": _state_entry(sample_id, "M2"),
        "m12_entry": _state_entry(sample_id, "M12"),
        "trajectory_meta": {
            "layer_count": 2,
            "hidden_dim": 3,
            "t0_token_index": -1,
        },
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


def _prompted_state_row(sample_id: str, condition: str, prompt_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "sample_type": "Conflict",
        "model_key": "qwen3_vl_8b",
        "protocol": "vt",
        "condition": condition,
        "prompt_set_key": "vt_primary_v1",
        "prompt_id": prompt_id,
        "shard_path": f"outputs/prompted/{sample_id}-{condition}-{prompt_id}.safetensors",
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": 3,
        "token_count": 4,
        "t0_token_index": -1,
        "cache_root": ".",
        "checksum": f"{sample_id}-{condition}-{prompt_id}",
        "metadata": {"tensor_key": "hidden_states"},
    }


def _prompted_state_rows(sample_id: str) -> list[dict[str, object]]:
    return [
        _prompted_state_row(sample_id, condition, prompt_id)
        for condition in ("M1", "M2", "M12")
        for prompt_id in ("vt_primary_v1_t01", "vt_primary_v1_t02")
    ]


def _read_jsonl(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_build_state_bundles_writes_prompt_conditioned_manifest_and_summary(tmp_path) -> None:
    state_manifest = tmp_path / "state_dataset_manifest.jsonl"
    prompt_set = tmp_path / "vt_primary_v1.yaml"
    prompt_cache_manifest = tmp_path / "prompt_cache_manifest.jsonl"
    prompted_manifest = tmp_path / "prompt_conditioned_manifest.jsonl"
    write_jsonl(state_manifest, [_state_row("sample-ok")])
    _prompt_set(prompt_set)
    write_jsonl(
        prompt_cache_manifest,
        [_prompt_cache_row("vt_primary_v1_t01"), _prompt_cache_row("vt_primary_v1_t02")],
    )
    write_jsonl(prompted_manifest, _prompted_state_rows("sample-ok"))

    result = build_state_bundles(
        state_dataset_manifest_path=state_manifest,
        prompt_set_path=prompt_set,
        prompt_cache_manifest_path=prompt_cache_manifest,
        prompt_conditioned_cache_manifest_path=prompted_manifest,
        output_root=tmp_path / "outputs/state_bundles",
        model_key="qwen3_vl_8b",
        protocol="vt",
    )

    rows = _read_jsonl(result.manifest_path)
    missing_rows = _read_jsonl(result.missing_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.complete_count == 1
    assert result.missing_count == 0
    assert result.prompt_count == 2
    assert len(rows) == 1
    assert missing_rows == []
    assert summary["total_samples"] == 1
    assert summary["complete_samples"] == 1
    assert summary["missing_samples"] == 0
    assert summary["prompt_count"] == 2
    assert rows[0]["sample_id"] == "sample-ok"
    assert rows[0]["prompt_set_key"] == "vt_primary_v1"
    assert rows[0]["metadata"]["split_group_id"] == "sample-ok"
    assert rows[0]["view_labels"] == _view_labels()
    assert [prompt["prompt_id"] for prompt in rows[0]["prompts"]] == [
        "vt_primary_v1_t01",
        "vt_primary_v1_t02",
    ]
    for view_key in ("M1", "M2", "M12"):
        view = rows[0]["views"][view_key]
        assert view["state_cache"]["condition"] == view_key
        assert set(view["prompts"]) == {"vt_primary_v1_t01", "vt_primary_v1_t02"}
        assert view["prompts"]["vt_primary_v1_t01"]["prompt_cache"]["cache_key"]
        assert (
            view["prompts"]["vt_primary_v1_t01"]["prompt_conditioned_state"]["prompt_id"]
            == "vt_primary_v1_t01"
        )
        assert (
            view["prompts"]["vt_primary_v1_t02"]["prompt_conditioned_state"]["shard_path"]
            != view["prompts"]["vt_primary_v1_t01"]["prompt_conditioned_state"]["shard_path"]
        )


def test_build_state_bundles_records_missing_prompt_cache_rows_per_sample(tmp_path) -> None:
    state_manifest = tmp_path / "state_dataset_manifest.jsonl"
    prompt_set = tmp_path / "vt_primary_v1.yaml"
    prompt_cache_manifest = tmp_path / "prompt_cache_manifest.jsonl"
    prompted_manifest = tmp_path / "prompt_conditioned_manifest.jsonl"
    write_jsonl(state_manifest, [_state_row("sample-a"), _state_row("sample-b")])
    _prompt_set(prompt_set)
    write_jsonl(prompt_cache_manifest, [_prompt_cache_row("vt_primary_v1_t01")])
    write_jsonl(prompted_manifest, _prompted_state_rows("sample-a") + _prompted_state_rows("sample-b"))

    result = build_state_bundles(
        state_dataset_manifest_path=state_manifest,
        prompt_set_path=prompt_set,
        prompt_cache_manifest_path=prompt_cache_manifest,
        prompt_conditioned_cache_manifest_path=prompted_manifest,
        output_root=tmp_path / "outputs/state_bundles",
        model_key="qwen3_vl_8b",
        protocol="VT",
    )

    rows = _read_jsonl(result.manifest_path)
    missing_rows = _read_jsonl(result.missing_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert rows == []
    assert [row["sample_id"] for row in missing_rows] == ["sample-a", "sample-b"]
    assert missing_rows[0]["missing_prompt_ids"] == ["vt_primary_v1_t02"]
    assert summary["complete_samples"] == 0
    assert summary["missing_samples"] == 2


def test_iter_and_load_state_bundles_read_bundle_manifest(tmp_path) -> None:
    state_manifest = tmp_path / "state_dataset_manifest.jsonl"
    prompt_set = tmp_path / "vt_primary_v1.yaml"
    prompt_cache_manifest = tmp_path / "prompt_cache_manifest.jsonl"
    prompted_manifest = tmp_path / "prompt_conditioned_manifest.jsonl"
    write_jsonl(state_manifest, [_state_row("sample-ok")])
    _prompt_set(prompt_set)
    write_jsonl(
        prompt_cache_manifest,
        [_prompt_cache_row("vt_primary_v1_t01"), _prompt_cache_row("vt_primary_v1_t02")],
    )
    write_jsonl(prompted_manifest, _prompted_state_rows("sample-ok"))
    result = build_state_bundles(
        state_dataset_manifest_path=state_manifest,
        prompt_set_path=prompt_set,
        prompt_cache_manifest_path=prompt_cache_manifest,
        prompt_conditioned_cache_manifest_path=prompted_manifest,
        output_root=tmp_path / "outputs/state_bundles",
        model_key="qwen3_vl_8b",
        protocol="VT",
    )

    assert [bundle["sample_id"] for bundle in iter_state_bundles(result.manifest_path)] == [
        "sample-ok"
    ]
    assert load_state_bundle("sample-ok", result.manifest_path)["views"]["M12"]["state_cache"][
        "condition"
    ] == "M12"


def test_build_state_bundles_records_missing_prompt_conditioned_state_rows(tmp_path) -> None:
    state_manifest = tmp_path / "state_dataset_manifest.jsonl"
    prompt_set = tmp_path / "vt_primary_v1.yaml"
    prompt_cache_manifest = tmp_path / "prompt_cache_manifest.jsonl"
    prompted_manifest = tmp_path / "prompt_conditioned_manifest.jsonl"
    write_jsonl(state_manifest, [_state_row("sample-a")])
    _prompt_set(prompt_set)
    write_jsonl(
        prompt_cache_manifest,
        [_prompt_cache_row("vt_primary_v1_t01"), _prompt_cache_row("vt_primary_v1_t02")],
    )
    write_jsonl(prompted_manifest, [_prompted_state_row("sample-a", "M1", "vt_primary_v1_t01")])

    result = build_state_bundles(
        state_dataset_manifest_path=state_manifest,
        prompt_set_path=prompt_set,
        prompt_cache_manifest_path=prompt_cache_manifest,
        prompt_conditioned_cache_manifest_path=prompted_manifest,
        output_root=tmp_path / "outputs/state_bundles",
        model_key="qwen3_vl_8b",
        protocol="VT",
    )

    rows = _read_jsonl(result.manifest_path)
    missing_rows = _read_jsonl(result.missing_path)

    assert rows == []
    assert missing_rows[0]["sample_id"] == "sample-a"
    assert "M1:vt_primary_v1_t02" in missing_rows[0]["missing_prompt_conditioned_states"]
    assert "M12:vt_primary_v1_t02" in missing_rows[0]["missing_prompt_conditioned_states"]
