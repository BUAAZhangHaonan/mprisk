from __future__ import annotations

import json
from pathlib import Path

import pytest

from mprisk.state.spherical import compute_spherical_state
from mprisk.state.thresholds import calibrate_registered_aligned_thresholds
from mprisk.utils.io import write_jsonl
from scripts.assign_state_patterns import assign_state_patterns


def _identity(*, model_key: str = "qwen3_vl_8b") -> dict[str, str]:
    return {
        "model_key": model_key,
        "protocol": "VT",
        "prompt_set_key": "vt_main_p8_seed20260717",
        "prompt_set_artifact_sha256": "a" * 64,
        "repr_key": "tme_proxy_anchor_v1",
        "encoder_checkpoint_sha256": "b" * 64,
        "split_assignment_sha256": "c" * 64,
        "embedding_manifest_sha256": "d" * 64,
    }


def _score(sample_id: str, *, model_key: str = "qwen3_vl_8b") -> dict[str, object]:
    prompts = {f"p{index:02d}": [1.0, 0.0] for index in range(1, 9)}
    bundle = {
        "sample_id": sample_id,
        "sample_type": "Aligned",
        "calibration_split": "aligned_calibration",
        "representation_split": "aligned_calibration",
        "embeddings": {
            "M1": prompts,
            "M2": {key: [0.0, 1.0] for key in prompts},
            "M12": {key: [2**-0.5, 2**-0.5] for key in prompts},
        },
    }
    return {
        **compute_spherical_state(bundle),
        **_identity(model_key=model_key),
        "representation_split": "aligned_calibration",
        "calibration_split": "aligned_calibration",
    }


def test_calibration_binds_complete_encoder_and_embedding_identity() -> None:
    calibration = calibrate_registered_aligned_thresholds([_score("sample-a")])

    assert {key: calibration[key] for key in _identity()} == _identity()


def test_calibration_rejects_rows_from_different_backbones() -> None:
    with pytest.raises(ValueError, match="identity field model_key"):
        calibrate_registered_aligned_thresholds(
            [_score("sample-a"), _score("sample-b", model_key="internvl3_5_8b")]
        )


def test_pattern_assignment_rejects_thresholds_from_another_checkpoint(
    tmp_path: Path,
) -> None:
    row = _score("sample-a")
    scores_path = write_jsonl(tmp_path / "scores.jsonl", [row])
    thresholds = calibrate_registered_aligned_thresholds([row])
    thresholds["encoder_checkpoint_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="encoder_checkpoint_sha256"):
        assign_state_patterns(
            sdr_scores_path=scores_path,
            thresholds=thresholds,
            output_dir=tmp_path / "patterns",
        )


def test_sdr_export_hash_is_part_of_score_identity(tmp_path: Path) -> None:
    from scripts.compute_sdr_scores import compute_sdr_scores

    prompt_vectors = {f"p{index:02d}": [1.0, 0.0] for index in range(1, 9)}
    bundle = {
        "sample_id": "sample-a",
        "sample_type": "Aligned",
        "calibration_split": "aligned_calibration",
        "representation_split": "aligned_calibration",
        "embeddings": {
            "M1": prompt_vectors,
            "M2": {key: [0.0, 1.0] for key in prompt_vectors},
            "M12": {key: [2**-0.5, 2**-0.5] for key in prompt_vectors},
        },
        **{key: value for key, value in _identity().items() if key != "embedding_manifest_sha256"},
    }
    source = write_jsonl(tmp_path / "embeddings.jsonl", [bundle])
    result = compute_sdr_scores(
        embedding_manifest_path=source,
        output_dir=tmp_path / "sdr",
    )
    score = json.loads(result.scores_path.read_text(encoding="utf-8"))

    assert len(score["embedding_manifest_sha256"]) == 64
    assert result.summary_path.exists()
