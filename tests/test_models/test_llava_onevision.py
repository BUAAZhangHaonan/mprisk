from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.llava_onevision import (
    LlavaOneVisionWrapper,
    _validate_native_video_processor_output,
)


def _model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    payload = {
        "model_type": "llava_onevision",
        "architectures": ["LlavaOnevisionForConditionalGeneration"],
        "torch_dtype": "float16",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 3,
            "max_position_embeddings": 32768,
        },
    }
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return path


def _request(media, *, condition="M12"):
    content = []
    if condition != "M2":
        content.append({"type": "video", "video": str(media), "fps": 1.0})
    content.append({"type": "text", "text": "Describe the overall affect."})
    return PrefillRequest(
        sample_id="sample",
        model_key="llava_onevision_qwen2_7b",
        protocol="vt",
        condition=condition,
        dataset_key="source",
        split="all",
        messages=({"role": "user", "content": content},),
        media_paths={"vision": str(media)},
        use_audio_in_video=False,
    )


class LlavaOnevisionProcessor:
    def __init__(self, *, token_count=1602, frames=8):
        self.token_count = token_count
        self.frames = frames

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        input_ids = torch.arange(self.token_count).unsqueeze(0)
        output = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        if messages[0]["content"][0]["type"] == "video":
            output["pixel_values_videos"] = torch.zeros(
                (1, self.frames, 3, 384, 384)
            )
        return output


def _model():
    language = SimpleNamespace(num_hidden_layers=2, hidden_size=3)

    def call(self, **kwargs):
        self.kwargs = kwargs
        length = int(kwargs["attention_mask"].shape[-1])
        states = tuple(
            torch.full((1, length, 3), float(index)) for index in range(3)
        )
        return SimpleNamespace(hidden_states=states)

    cls = type(
        "LlavaOnevisionForConditionalGeneration",
        (),
        {
            "__init__": lambda self: setattr(
                self, "config", SimpleNamespace(text_config=language)
            ),
            "__call__": call,
        },
    )
    return cls()


def _wrapper(tmp_path, processor):
    return LlavaOneVisionWrapper(
        model_key="llava_onevision_qwen2_7b",
        model_path=_model_dir(tmp_path),
        device="cpu",
        dtype="float16",
        model=_model(),
        processor=processor,
        runtime_versions={"transformers": "test"},
    )


def test_native_video_f8_preserves_frames_and_extracts_full_t0_trajectory(
    tmp_path,
    monkeypatch,
):
    indices = [6, 18, 31, 43, 56, 68, 81, 93]
    monkeypatch.setattr(
        "mprisk.models.video_frame_utils.uniform_video_sample",
        lambda path, count: (
            [Image.new("RGB", (2, 2), color=index) for index in range(count)],
            {
                "frames_indices": indices,
                "total_num_frames": 100,
                "fps": 25.0,
                "width": 2,
                "height": 2,
                "duration": 4.0,
                "video_backend": "decord",
            },
        ),
    )
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    processor = LlavaOnevisionProcessor()
    result = _wrapper(tmp_path, processor).extract_prefill(_request(media))

    assert result.trajectory.shape == (2, 3)
    np.testing.assert_allclose(result.trajectory[:, 0], [1.0, 2.0])
    content = processor.messages[0]["content"]
    assert [item["type"] for item in content] == ["video", "text"]
    assert len(content[0]["video"]) == 8
    assert processor.kwargs["processor_kwargs"] == {
        "videos_kwargs": {"do_sample_frames": False}
    }
    assert result.token_count == 1602
    assert result.provenance["video_input_mode"] == "native_video"
    assert result.provenance["requested_frames"] == 8
    assert result.provenance["actual_frames"] == 8
    assert result.provenance["video_frame_indices"] == [indices]
    assert result.provenance["context_limit"] == 32768
    assert result.provenance["no_truncation"] is True


def test_text_only_m2_has_no_video_tensor(tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    processor = LlavaOnevisionProcessor(token_count=31)
    result = _wrapper(tmp_path, processor).extract_prefill(
        _request(media, condition="M2")
    )

    assert [item["type"] for item in processor.messages[0]["content"]] == ["text"]
    assert "processor_kwargs" not in processor.kwargs
    assert result.provenance["requested_frames"] == 0
    assert result.provenance["actual_frames"] == 0


@pytest.mark.parametrize(
    "model_inputs,match",
    [
        (
            {
                "attention_mask": torch.ones((1, 32769), dtype=torch.long),
                "pixel_values_videos": torch.zeros((1, 8, 3, 2, 2)),
            },
            "exceeds the checkpoint limit",
        ),
        (
            {
                "attention_mask": torch.ones((1, 100), dtype=torch.long),
                "pixel_values_videos": torch.zeros((1, 7, 3, 2, 2)),
            },
            r"shape \[1,8,C,H,W\]",
        ),
        (
            {
                "attention_mask": torch.ones((1, 100), dtype=torch.long),
                "pixel_values_videos": torch.zeros((1, 8, 3, 2, 2)),
                "pixel_values": torch.zeros((8, 3, 2, 2)),
            },
            "produced image tensors",
        ),
    ],
)
def test_native_video_output_fails_closed(model_inputs, match):
    with pytest.raises(ValueError, match=match):
        _validate_native_video_processor_output(
            model_inputs,
            expected_frames=8,
            max_position_embeddings=32768,
        )


def test_native_video_output_accepts_exact_context_limit():
    token_count, frame_count = _validate_native_video_processor_output(
        {
            "attention_mask": torch.ones((1, 32768), dtype=torch.long),
            "pixel_values_videos": torch.zeros((1, 8, 3, 2, 2)),
        },
        expected_frames=8,
        max_position_embeddings=32768,
    )

    assert token_count == 32768
    assert frame_count == 8


def test_wrapper_rejects_non_f8_protocol(tmp_path):
    with pytest.raises(ValueError, match="frozen native-video F8"):
        LlavaOneVisionWrapper(
            model_key="llava_onevision_qwen2_7b",
            model_path=_model_dir(tmp_path),
            device="cpu",
            dtype="float16",
            video_num_segments=4,
            model=_model(),
            processor=LlavaOnevisionProcessor(),
            runtime_versions={"transformers": "test"},
        )
