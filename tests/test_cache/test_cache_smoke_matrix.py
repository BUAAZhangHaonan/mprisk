from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mprisk.cache.cache_smoke_matrix as smoke
from mprisk.cache.cache_smoke_matrix import (
    _evidence_matches,
    _sha256,
    _validate_frame_contract,
    _validate_media_contract,
)


@pytest.mark.parametrize(
    ("protocol", "condition", "message_types", "embedded_audio"),
    [
        ("vt", "M1", ("video",), False),
        ("vt", "M2", (), False),
        ("vt", "M12", ("video",), False),
        ("va", "M1", ("video",), False),
        ("va", "M2", ("audio",), False),
        ("va", "M12", ("video",), True),
        ("va", "M12", ("video", "audio"), False),
    ],
)
def test_validate_media_contract(
    tmp_path: Path,
    protocol: str,
    condition: str,
    message_types: tuple[str, ...],
    embedded_audio: bool,
) -> None:
    media = {"vision": str(tmp_path / "vision.mp4"), "audio": str(tmp_path / "audio.wav")}
    for path_value in media.values():
        path = Path(path_value)
        path.write_bytes(b"media")
    value, contains_video = _validate_media_contract(
        protocol,
        {
            "condition": condition,
            "media_paths": media,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *[{"type": item} for item in message_types],
                        {"type": "text", "text": "prompt"},
                    ],
                }
            ],
            "use_audio_in_video": embedded_audio,
        },
    )
    assert value.startswith(f"{protocol}:{condition}:")
    assert contains_video is ("video" in message_types)


def test_validate_media_contract_rejects_wrong_message_media(tmp_path: Path) -> None:
    vision = tmp_path / "vision.mp4"
    vision.write_bytes(b"video")
    with pytest.raises(ValueError, match="expected message media"):
        _validate_media_contract(
            "vt",
            {
                "condition": "M2",
                "media_paths": {"vision": str(vision)},
                "messages": [
                    {"role": "user", "content": [{"type": "video"}, {"type": "text"}]}
                ],
                "use_audio_in_video": False,
            },
        )


def test_validate_frame_contract_requires_exact_actual_frames() -> None:
    provenance = {
        "requested_frames": 8,
        "actual_frames": 8,
        "video_sampling_method": "uniform_midpoint_decord_v1",
        "video_frame_indices": [[1, 3, 5, 7, 9, 11, 13, 15]],
        "video_source_total_frames": [16],
    }
    assert (
        _validate_frame_contract(
            provenance,
            condition="M1",
            contains_video=True,
            expected_frames=8,
            expected_method="uniform_midpoint_decord_v1",
        )
        == 8
    )
    with pytest.raises(ValueError, match="frame contract mismatch"):
        _validate_frame_contract(
            {
                "requested_frames": 8,
                "actual_frames": 7,
                "video_sampling_method": "uniform_midpoint_decord_v1",
                "video_frame_indices": [[1, 3, 5, 7, 9, 11, 13]],
                "video_source_total_frames": [16],
            },
            condition="M12",
            contains_video=True,
            expected_frames=8,
            expected_method="uniform_midpoint_decord_v1",
        )
    assert (
        _validate_frame_contract(
            {"requested_frames": 0, "actual_frames": 0},
            condition="M2",
            contains_video=False,
            expected_frames=8,
            expected_method="uniform_midpoint_decord_v1",
        )
        == 0
    )


def test_validate_frame_contract_rejects_nonuniform_or_duplicate_indices() -> None:
    provenance = {
        "requested_frames": 8,
        "actual_frames": 8,
        "video_sampling_method": "processor_default",
        "video_frame_indices": [[1, 3, 5, 7, 9, 11, 13, 15]],
        "video_source_total_frames": [16],
    }
    with pytest.raises(ValueError, match="expected video_sampling_method"):
        _validate_frame_contract(
            provenance,
            condition="M1",
            contains_video=True,
            expected_frames=8,
            expected_method="uniform_midpoint_decord_v1",
        )
    provenance["video_sampling_method"] = "uniform_midpoint_decord_v1"
    provenance["video_frame_indices"] = [[1, 1, 3, 5, 7, 9, 11, 13]]
    with pytest.raises(ValueError, match="invalid video_frame_indices"):
        _validate_frame_contract(
            provenance,
            condition="M12",
            contains_video=True,
            expected_frames=8,
            expected_method="uniform_midpoint_decord_v1",
        )


def test_validate_frame_contract_accepts_backend_without_index_evidence() -> None:
    assert (
        _validate_frame_contract(
            {
                "requested_frames": 8,
                "actual_frames": 8,
                "video_sampling_method": "uniform_nframes_qwen_omni_utils_v1",
            },
            condition="M12",
            contains_video=True,
            expected_frames=8,
            expected_method="uniform_nframes_qwen_omni_utils_v1",
        )
        == 8
    )


def test_evidence_matches_all_runtime_signatures(tmp_path: Path, monkeypatch) -> None:
    prompt_set = tmp_path / "prompt.yaml"
    prompt_set.write_text("prompt", encoding="utf-8")
    asset_config = tmp_path / "assets.yaml"
    asset_config.write_text("assets", encoding="utf-8")
    smoke_manifest = tmp_path / "smoke.jsonl"
    smoke_manifest.write_text("{}\n", encoding="utf-8")
    model = SimpleNamespace(
        model_key="model",
        family="family",
        protocol="vt",
        python=tmp_path / "python",
        trajectory_shape=(2, 3),
        requested_frames=8,
        frame_protocol="fixed_uniform_temporal_samples_v1",
        video_sampling_method="uniform_midpoint_decord_v1",
        extra_args=(),
    )
    config = SimpleNamespace(prompt_sets={"vt": prompt_set}, asset_config=asset_config)
    job = SimpleNamespace(model=model, domain=SimpleNamespace(domain="target"))
    asset_signature = {"schema": "mprisk_cache_asset_signature_v1", "digest": "asset"}
    monkeypatch.setattr(
        smoke, "build_asset_signature", lambda config, model: asset_signature
    )
    evidence = {
        "schema": "mprisk_cache_smoke_evidence_v2",
        "status": "PASS",
        "model_key": "model",
        "family": "family",
        "protocol": "vt",
        "domain": "target",
        "expected_tasks": 48,
        "completed_tasks": 48,
        "failed_tasks": 0,
        "environment_python": str(model.python),
        "prompt_set_sha256": _sha256(prompt_set),
        "asset_config_sha256": _sha256(asset_config),
        "smoke_manifest_sha256": _sha256(smoke_manifest),
        "trajectory_shape": [2, 3],
        "extra_args": [],
        "requested_frames": 8,
        "frame_protocol": "fixed_uniform_temporal_samples_v1",
        "video_sampling_method": "uniform_midpoint_decord_v1",
        "asset_signature": asset_signature,
    }
    assert _evidence_matches(config, job, evidence, _sha256(smoke_manifest))
    for key in (
        "asset_config_sha256",
        "extra_args",
        "smoke_manifest_sha256",
        "requested_frames",
        "frame_protocol",
        "video_sampling_method",
        "asset_signature",
    ):
        stale = dict(evidence)
        stale[key] = None
        assert not _evidence_matches(config, job, stale, _sha256(smoke_manifest))
