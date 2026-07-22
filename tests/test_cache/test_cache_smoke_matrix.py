from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mprisk.cache.cache_smoke_matrix as smoke
from mprisk.cache.cache_smoke_matrix import (
    _evidence_matches,
    _sha256,
    _validate_frame_contract,
    _validate_gemma4_processor_media_contract,
    _validate_media_contract,
    build_parser,
)


def test_parser_accepts_explicit_tmux_session() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "matrix.yaml",
            "--domain",
            "target",
            "--model",
            "model",
            "--tmux-session",
            "target-smoke-gpu1",
            "--physical-gpu",
            "1",
            "--launch",
        ]
    )
    assert args.tmux_session == "target-smoke-gpu1"
    assert args.physical_gpu == 1


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


@pytest.mark.parametrize(
    (
        "condition",
        "message_types",
        "embedded_audio",
        "video_count",
        "audio_count",
        "audio_source",
    ),
    [
        ("M1", ("video",), False, 1, 0, "none"),
        ("M2", ("audio",), False, 0, 1, "explicit_audio_path"),
        ("M12", ("video", "audio"), True, 1, 1, "embedded_video_waveform"),
    ],
)
def test_validate_gemma4_media_contract_requires_exact_processor_inputs(
    tmp_path: Path,
    condition: str,
    message_types: tuple[str, ...],
    embedded_audio: bool,
    video_count: int,
    audio_count: int,
    audio_source: str,
) -> None:
    vision = tmp_path / "vision.mp4"
    audio = tmp_path / "audio.wav"
    vision.write_bytes(b"video")
    audio.write_bytes(b"audio")
    provenance = {
        "processor_media_contract": {
            "schema": "mprisk_gemma4_processor_media_contract_v1",
            "condition": condition,
            "video_input_count": video_count,
            "audio_input_count": audio_count,
            "audio_input_source": audio_source,
            "image_input_count": 0,
        }
    }

    value, contains_video = _validate_media_contract(
        "va",
        {
            "condition": condition,
            "media_paths": {"vision": str(vision), "audio": str(audio)},
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
        family="gemma4",
        provenance=provenance,
    )

    assert "processor_media_contract_v1" in value
    assert contains_video is (video_count == 1)


def test_validate_gemma4_media_contract_rejects_duplicate_audio() -> None:
    with pytest.raises(ValueError, match="processor_media_contract mismatch"):
        _validate_gemma4_processor_media_contract(
            "M12",
            {
                "processor_media_contract": {
                    "schema": "mprisk_gemma4_processor_media_contract_v1",
                    "condition": "M12",
                    "video_input_count": 1,
                    "audio_input_count": 2,
                    "audio_input_source": "embedded_video_waveform",
                    "image_input_count": 0,
                }
            },
        )


def test_validate_gemma4_media_contract_rejects_wrong_audio_source() -> None:
    with pytest.raises(ValueError, match="processor_media_contract mismatch"):
        _validate_gemma4_processor_media_contract(
            "M12",
            {
                "processor_media_contract": {
                    "schema": "mprisk_gemma4_processor_media_contract_v1",
                    "condition": "M12",
                    "video_input_count": 1,
                    "audio_input_count": 1,
                    "audio_input_source": "explicit_audio_path",
                    "image_input_count": 0,
                }
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


def test_validate_frame_contract_rejects_backend_without_index_evidence() -> None:
    with pytest.raises(ValueError, match="must provide both frame indices"):
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


def test_evidence_matches_all_runtime_signatures(tmp_path: Path, monkeypatch) -> None:
    prompt_set = tmp_path / "prompt.yaml"
    prompt_set.write_text("prompt", encoding="utf-8")
    asset_config = tmp_path / "assets.yaml"
    asset_config.write_text("assets", encoding="utf-8")
    smoke_manifest = tmp_path / "smoke.jsonl"
    smoke_manifest.write_text("{}\n", encoding="utf-8")
    environment = tmp_path / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "lib").mkdir()
    model = SimpleNamespace(
        model_key="model",
        family="family",
        protocol="vt",
        python=environment / "bin" / "python",
        python_no_user_site=False,
        env_isolation=False,
        dtype="bfloat16",
        trajectory_shape=(2, 3),
        requested_frames=8,
        frame_protocol="fixed_uniform_temporal_samples_v1",
        video_sampling_method="uniform_midpoint_decord_v1",
        extra_args=(),
    )
    config = SimpleNamespace(prompt_sets={"vt": prompt_set}, asset_config=asset_config)
    job = SimpleNamespace(model=model, domain=SimpleNamespace(domain="target"))
    asset_signature = {"schema": "mprisk_cache_asset_signature_v2", "digest": "asset"}
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
        "python_no_user_site": False,
        "env_isolation": False,
        "runtime_library_path": str((environment / "lib").resolve()),
        "prompt_set_sha256": _sha256(prompt_set),
        "asset_config_sha256": _sha256(asset_config),
        "smoke_manifest_sha256": _sha256(smoke_manifest),
        "trajectory_shape": [2, 3],
        "extra_args": [],
        "dtype": "bfloat16",
        "requested_frames": 8,
        "frame_protocol": "fixed_uniform_temporal_samples_v1",
        "video_sampling_method": "uniform_midpoint_decord_v1",
        "asset_signature": asset_signature,
    }
    assert _evidence_matches(config, job, evidence, _sha256(smoke_manifest))
    for key in (
        "asset_config_sha256",
        "extra_args",
        "dtype",
        "runtime_library_path",
        "python_no_user_site",
        "env_isolation",
        "smoke_manifest_sha256",
        "requested_frames",
        "frame_protocol",
        "video_sampling_method",
        "asset_signature",
    ):
        stale = dict(evidence)
        stale[key] = None
        assert not _evidence_matches(config, job, stale, _sha256(smoke_manifest))
