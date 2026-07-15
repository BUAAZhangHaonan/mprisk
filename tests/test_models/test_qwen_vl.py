from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.qwen_omni import build_condition_request
from mprisk.models.qwen_vl import QwenVlWrapper


def _model_dir(tmp_path):
    model_dir = tmp_path / "Qwen3-VL-8B-Instruct"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config": {
                    "dtype": "bfloat16",
                    "num_hidden_layers": 36,
                    "hidden_size": 4096,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return model_dir


class _FakeProcessor:
    def __init__(self) -> None:
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[0, 10, 11, 12]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
            "pixel_values_videos": torch.ones((1, 3, 2, 2)),
        }


class _FakeQwen3VL:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=36, hidden_size=4096)
        )
        self.call_kwargs = None

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        hidden_states = tuple(
            torch.full((1, 4, 4096), float(index), dtype=torch.float32)
            for index in range(37)
        )
        return SimpleNamespace(hidden_states=hidden_states)


def _wrapper(tmp_path, processor=None, model=None):
    return QwenVlWrapper(
        model_key="qwen3_vl_8b",
        model_path=_model_dir(tmp_path),
        device="cpu",
        model=model or _FakeQwen3VL(),
        processor=processor or _FakeProcessor(),
        runtime_versions={"transformers": "test"},
    )


def test_qwen3_vl_vt_conditions_do_not_leak_modalities(tmp_path) -> None:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    common = {
        "sample_id": "sample-1",
        "model_key": "qwen3_vl_8b",
        "protocol": "vt",
        "prompt_set_key": "main_p8",
        "prompt_id": "p01",
        "dataset_key": "ch_sims_v2",
        "split": "test",
        "media_paths": {"vision": str(media)},
        "transcript": "private transcript",
        "task_prompt": "Describe the emotion.",
    }

    m1 = build_condition_request(condition="M1", **common)
    m2 = build_condition_request(condition="M2", **common)
    m12 = build_condition_request(condition="M12", **common)

    assert [item["type"] for item in m1.messages[0]["content"]] == ["video", "text"]
    assert "private transcript" not in m1.messages[0]["content"][-1]["text"]
    assert [item["type"] for item in m2.messages[0]["content"]] == ["text"]
    assert "private transcript" in m2.messages[0]["content"][-1]["text"]
    assert [item["type"] for item in m12.messages[0]["content"]] == ["video", "text"]
    assert "private transcript" in m12.messages[0]["content"][-1]["text"]


def test_qwen3_vl_extracts_all_language_blocks_at_last_non_padding_token(tmp_path) -> None:
    processor = _FakeProcessor()
    model = _FakeQwen3VL()
    wrapper = _wrapper(tmp_path, processor=processor, model=model)
    request = build_condition_request(
        sample_id="sample-1",
        model_key="qwen3_vl_8b",
        protocol="vt",
        condition="M12",
        prompt_set_key="main_p8",
        prompt_id="p01",
        dataset_key="dataset",
        split="test",
        media_paths={"vision": str(tmp_path / "sample.mp4")},
        transcript="spoken words",
        task_prompt="Identify the emotion.",
    )

    result = wrapper.extract_prefill(request)

    assert result.trajectory.shape == (36, 4096)
    np.testing.assert_allclose(result.trajectory[:, 0], np.arange(1, 37))
    assert result.token_count == 4
    assert result.t0_token_index == 3
    assert processor.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "fps": 1.0,
    }
    assert model.call_kwargs["use_cache"] is False
    assert model.call_kwargs["output_hidden_states"] is True
    assert model.call_kwargs["return_dict"] is True
    assert model.call_kwargs["logits_to_keep"] == 1
    assert result.provenance["hidden_state_index_offset"] == 1


def test_qwen3_vl_preserves_native_video_and_multiple_image_content(tmp_path) -> None:
    processor = _FakeProcessor()
    wrapper = _wrapper(tmp_path, processor=processor)
    request = PrefillRequest(
        sample_id="sample-images",
        model_key="qwen3_vl_8b",
        protocol="vt",
        condition="M1",
        prompt_set_key="main_p8",
        prompt_id="p02",
        dataset_key="dataset",
        split="test",
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "frame-1.jpg"},
                    {"type": "image", "image": "frame-2.jpg"},
                    {"type": "text", "text": "Describe the emotion."},
                ],
            },
        ),
        media_paths={"vision": "frame-1.jpg"},
        use_audio_in_video=False,
    )

    wrapper.extract_prefill(request)

    assert [item["type"] for item in processor.messages[0]["content"]] == [
        "image",
        "image",
        "text",
    ]
