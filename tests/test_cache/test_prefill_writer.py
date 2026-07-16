from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from mprisk.cache.cache_manifest import load_full_cache_manifest
from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.cache.prefill_writer import write_prefill_result
from mprisk.models.base_wrapper import PrefillRequest, PrefillResult


def _result() -> PrefillResult:
    request = PrefillRequest(
        sample_id="sample:1",
        model_key="qwen2_5_omni_7b",
        protocol="VA",
        condition="m12",
        prompt_set_key="main_p8",
        prompt_id="p01",
        dataset_key="ch_sims_v2",
        split="test",
        messages=({"role": "user", "content": [{"type": "text", "text": "task"}]},),
        media_paths={"vision": "/media/sample.mp4", "audio": "/media/sample.mp4"},
        use_audio_in_video=True,
    )
    return PrefillResult(
        request=request,
        trajectory=np.arange(6, dtype=np.float32).reshape(2, 3),
        token_count=4,
        t0_token_index=3,
        provenance={"model_class": "Qwen2_5OmniThinkerForConditionalGeneration"},
    )


def test_prefill_writer_round_trips_through_full_cache_manifest(tmp_path) -> None:
    artifact = write_prefill_result(_result(), output_root=tmp_path)

    assert load_file(artifact.shard_path)["hidden_states"].shape == (2, 3)
    sidecar = json.loads(artifact.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["request"]["use_audio_in_video"] is True
    assert sidecar["request"]["prompt_set_key"] == "main_p8"
    assert sidecar["entry"]["prompt_id"] == "p01"
    assert sidecar["entry"]["metadata"]["t0_token_index"] == 3

    manifest = load_full_cache_manifest(
        tmp_path,
        manifest_path=artifact.manifest_path,
        ledger_path=tmp_path / "missing-ledger.csv",
    )
    entry = manifest.query("sample:1", "qwen2_5_omni_7b", "va", "M12")
    assert entry is not None
    np.testing.assert_array_equal(extract_t0_trajectory(entry), _result().trajectory)


def test_prefill_writer_refuses_implicit_overwrite(tmp_path) -> None:
    write_prefill_result(_result(), output_root=tmp_path)

    with pytest.raises(FileExistsError, match="already contains"):
        write_prefill_result(_result(), output_root=tmp_path)
