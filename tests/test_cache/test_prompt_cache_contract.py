from __future__ import annotations

import json

import pytest

from mprisk.cache.prompt_cache import (
    PromptCacheManifest,
    load_prompt_cache_manifest,
    read_prompt_cache_rows,
    write_prompt_cache_manifest,
)


def test_prompt_cache_manifest_round_trips_jsonl_and_indexes_contract_fields(tmp_path) -> None:
    manifest_path = tmp_path / "prompt_cache_manifest.jsonl"
    rows = [
        {
            "model_key": "qwen3_vl_8b",
            "prompt_set_key": "vt_primary_v1",
            "prompt_id": "vt_primary_v1_t01",
            "protocol": "VT",
            "cache_key": "cache-a",
            "artifact_uri": "outputs/prompt_cache/cache-a.safetensors",
        },
        {
            "model_key": "qwen3_vl_8b",
            "prompt_set_key": "vt_primary_v1",
            "prompt_id": "vt_primary_v1_t02",
            "protocol": "vt",
            "cache_key": "cache-b",
        },
    ]

    written = write_prompt_cache_manifest(manifest_path, rows)
    loaded = load_prompt_cache_manifest(written)

    assert [row["cache_key"] for row in read_prompt_cache_rows(written)] == ["cache-a", "cache-b"]
    assert loaded.lookup(
        model_key="qwen3_vl_8b",
        prompt_set_key="vt_primary_v1",
        prompt_id="vt_primary_v1_t01",
        protocol="vt",
    )["cache_key"] == "cache-a"
    assert loaded.lookup(
        model_key="qwen3_vl_8b",
        prompt_set_key="vt_primary_v1",
        prompt_id="vt_primary_v1_t02",
        protocol="VT",
    )["cache_key"] == "cache-b"


def test_prompt_cache_manifest_reports_missing_rows_by_prompt_id(tmp_path) -> None:
    manifest_path = tmp_path / "prompt_cache_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "model_key": "qwen3_vl_8b",
                "prompt_set_key": "vt_primary_v1",
                "prompt_id": "vt_primary_v1_t01",
                "protocol": "vt",
                "cache_key": "cache-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_prompt_cache_manifest(manifest_path)

    assert loaded.missing_prompt_ids(
        model_key="qwen3_vl_8b",
        prompt_set_key="vt_primary_v1",
        prompt_ids=["vt_primary_v1_t01", "vt_primary_v1_t02"],
        protocol="VT",
    ) == ["vt_primary_v1_t02"]


def test_prompt_cache_manifest_rejects_rows_missing_required_contract_fields() -> None:
    with pytest.raises(ValueError, match="missing required field cache_key"):
        PromptCacheManifest(
            [
                {
                    "model_key": "qwen3_vl_8b",
                    "prompt_set_key": "vt_primary_v1",
                    "prompt_id": "vt_primary_v1_t01",
                    "protocol": "vt",
                }
            ]
        )
