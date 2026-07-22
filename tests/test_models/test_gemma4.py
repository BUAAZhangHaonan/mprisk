from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.gemma4 import (
    Gemma4Wrapper,
    _validate_media_contract,
    build_va_request,
)


def _model_dir(tmp_path):
    model_dir = tmp_path / "gemma-4-12B-it"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_unified",
                "dtype": "bfloat16",
                "text_config": {
                    "num_hidden_layers": 48,
                    "hidden_size": 3840,
                },
            }
        ),
        encoding="utf-8",
    )
    return model_dir


class _Processor:
    def __init__(self):
        self.call_kwargs = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered prompt"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[0, 10, 11, 12]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
        }


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=48, hidden_size=3840)
        )

    def __call__(self, **kwargs):
        return SimpleNamespace(
            hidden_states=tuple(
                torch.full((1, 4, 3840), float(index), dtype=torch.float32)
                for index in range(49)
            )
        )


def test_gemma4_va_requests_keep_conditions_separate(tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    common = dict(
        sample_id="sample-1",
        model_key="gemma4_12b",
        dataset_key="dataset",
        split="test",
        media_paths={"vision": str(media), "audio": str(media)},
        text_content="not used in VA",
        task_prompt="Describe the emotion.",
    )

    m1 = build_va_request(condition="M1", **common)
    m2 = build_va_request(condition="M2", **common)
    m12 = build_va_request(condition="M12", **common)

    assert [item["type"] for item in m1.messages[0]["content"]] == ["video", "text"]
    assert [item["type"] for item in m2.messages[0]["content"]] == ["audio", "text"]
    assert [item["type"] for item in m12.messages[0]["content"]] == ["video", "audio", "text"]
    assert m1.use_audio_in_video is False
    assert m2.use_audio_in_video is False
    assert m12.use_audio_in_video is True


def test_gemma4_m12_rejects_missing_audio():
    with pytest.raises(ValueError, match="both video and audio"):
        _validate_media_contract(
            "M12",
            {
                "videos": [np.zeros((1, 2, 2, 3), dtype=np.uint8)],
                "video_metadata": [{"total_num_frames": 1, "fps": 1.0, "frames_indices": [0]}],
                "audio": None,
                "audio_waveforms": None,
            },
        )


@pytest.mark.parametrize(
    ("condition", "media", "expected"),
    [
        (
            "M1",
            {
                "videos": [np.zeros((4, 2, 2, 3), dtype=np.uint8)],
                "video_metadata": [{"total_num_frames": 4}],
                "audio": None,
                "audio_waveforms": None,
                "images": None,
            },
            (1, 0, "none"),
        ),
        (
            "M2",
            {
                "videos": None,
                "video_metadata": None,
                "audio": ["sample.wav"],
                "audio_waveforms": None,
                "images": None,
            },
            (0, 1, "explicit_audio_path"),
        ),
        (
            "M12",
            {
                "videos": [np.zeros((4, 2, 2, 3), dtype=np.uint8)],
                "video_metadata": [{"total_num_frames": 4}],
                "audio": None,
                "audio_waveforms": [(np.ones(1600, dtype=np.float32), 16000)],
                "images": None,
            },
            (1, 1, "embedded_video_waveform"),
        ),
    ],
)
def test_gemma4_processor_media_contract_is_exact(condition, media, expected):
    contract = _validate_media_contract(condition, media)

    assert contract == {
        "schema": "mprisk_gemma4_processor_media_contract_v1",
        "condition": condition,
        "video_input_count": expected[0],
        "audio_input_count": expected[1],
        "audio_input_source": expected[2],
        "image_input_count": 0,
    }


def test_gemma4_rejects_duplicate_audio_processor_inputs():
    with pytest.raises(ValueError, match="both explicit audio paths"):
        _validate_media_contract(
            "M12",
            {
                "videos": [np.zeros((4, 2, 2, 3), dtype=np.uint8)],
                "video_metadata": [{"total_num_frames": 4}],
                "audio": ["sample.wav"],
                "audio_waveforms": [(np.ones(1600, dtype=np.float32), 16000)],
                "images": None,
            },
        )


def test_gemma4_extracts_joint_video_and_audio(monkeypatch, tmp_path):
    processor = _Processor()
    wrapper = Gemma4Wrapper(
        model_key="gemma4_12b",
        model_path=_model_dir(tmp_path),
        device="cpu",
        model=_Model(),
        processor=processor,
        video_num_segments=4,
        runtime_versions={"transformers": "test"},
    )
    request = PrefillRequest(
        sample_id="sample-1",
        model_key="gemma4_12b",
        protocol="va",
        condition="M12",
        prompt_set_key="p8",
        prompt_id="p1",
        dataset_key="dataset",
        split="test",
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "sample.mp4"},
                    {"type": "audio", "audio": "sample.mp4"},
                    {"type": "text", "text": "Describe the emotion."},
                ],
            },
        ),
        media_paths={"vision": "sample.mp4", "audio": "sample.mp4"},
        use_audio_in_video=True,
    )
    monkeypatch.setattr(
        "mprisk.models.gemma4._collect_media_inputs",
        lambda _, *, max_frames: {
            "audio": None,
            "audio_waveforms": [(np.ones(1600, dtype=np.float32), 16000)],
            "videos": [np.zeros((4, 2, 2, 3), dtype=np.uint8)],
            "video_metadata": [{"total_num_frames": 4, "fps": 1.0, "frames_indices": [0, 1, 2, 3]}],
            "images": None,
            "temporary_paths": [],
        },
    )

    result = wrapper.extract_prefill(request)

    assert result.trajectory.shape == (48, 3840)
    np.testing.assert_allclose(result.trajectory[:, 0], np.arange(1, 49))
    assert processor.call_kwargs["sampling_rate"] == 16000
    assert len(processor.call_kwargs["audio"]) == 1
    assert len(processor.call_kwargs["videos"]) == 1
    assert result.provenance["media_keys"] == ["videos", "audio"]
    assert result.provenance["processor_media_contract"] == {
        "schema": "mprisk_gemma4_processor_media_contract_v1",
        "condition": "M12",
        "video_input_count": 1,
        "audio_input_count": 1,
        "audio_input_source": "embedded_video_waveform",
        "image_input_count": 0,
    }
    assert result.provenance["requested_frames"] == 4
    assert result.provenance["actual_frames"] == 4
