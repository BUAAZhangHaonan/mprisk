from __future__ import annotations

import json

import numpy as np

from mprisk.cache.prefill_cli import main
from mprisk.models.base_wrapper import PrefillResult


def _manifest(tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "source_dataset": "ch_sims_v2",
                "protocol": "VA",
                "split": "test",
                "media_paths": {"vision": str(media), "audio": str(media)},
                "text_content": "transcript",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _args(tmp_path):
    return [
        "--manifest",
        str(_manifest(tmp_path)),
        "--sample-id",
        "sample-1",
        "--protocol",
        "va",
        "--task-prompt",
        "Identify the emotion.",
        "--model-path",
        str(tmp_path / "model"),
        "--device",
        "cpu",
        "--output-root",
        str(tmp_path / "output"),
    ]


def test_prefill_cli_dry_run_exposes_three_explicit_conditions(tmp_path, capsys) -> None:
    assert main([*_args(tmp_path), "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    requests = payload["requests"]
    assert [item["condition"] for item in requests] == ["M1", "M2", "M12"]
    assert [item["use_audio_in_video"] for item in requests] == [False, False, True]


def test_prefill_cli_mock_writes_all_condition_artifacts(tmp_path, capsys) -> None:
    class FakeWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            return None

        def extract_prefill(self, request):
            return PrefillResult(
                request=request,
                trajectory=np.ones((2, 3), dtype=np.float32),
                token_count=4,
                t0_token_index=3,
                provenance={"elapsed_seconds": 0.1, "peak_gpu_memory_bytes": None},
            )

        def close(self):
            return None

    assert main(_args(tmp_path), wrapper_factory=FakeWrapper) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert [item["condition"] for item in payload["artifacts"]] == ["M1", "M2", "M12"]
    manifest = json.loads(
        (tmp_path / "output/manifests/unified_full_cache_manifest.json").read_text()
    )
    assert len(manifest["entries"]) == 3
