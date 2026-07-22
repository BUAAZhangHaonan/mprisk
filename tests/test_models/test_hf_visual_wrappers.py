from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.gemma3 import Gemma3Wrapper
from mprisk.models.glm4v import Glm4vWrapper
from mprisk.models.qwen2_5_vl import Qwen2_5VlWrapper
from mprisk.models.wrapper_registry import get_wrapper


SPECS = (
    (
        Qwen2_5VlWrapper,
        "qwen2_5_vl_7b",
        "qwen2_5_vl",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2_5_VLProcessor",
        "root",
        3,
        4,
    ),
    (
        Glm4vWrapper,
        "glm4_6v_flash",
        "glm4v",
        "Glm4vForConditionalGeneration",
        "Glm46VProcessor",
        "text_config",
        4,
        5,
    ),
)


def _model_dir(tmp_path, *, model_type, architecture, location, layers, hidden):
    path = tmp_path / model_type
    path.mkdir()
    language = {
        "dtype": "bfloat16",
        "num_hidden_layers": layers,
        "hidden_size": hidden,
    }
    payload = {
        "model_type": model_type,
        "architectures": [architecture],
    }
    if location == "root":
        payload.update(language)
    else:
        payload["text_config"] = language
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return path


def _fake_model(architecture, *, location, layers, hidden):
    config = SimpleNamespace(num_hidden_layers=layers, hidden_size=hidden)
    if location != "root":
        config = SimpleNamespace(text_config=config)

    def call(self, **kwargs):
        self.call_kwargs = kwargs
        seq = int(kwargs["attention_mask"].shape[-1])
        states = tuple(
            torch.full((1, seq, hidden), float(index), dtype=torch.float32)
            for index in range(layers + 1)
        )
        return SimpleNamespace(hidden_states=states)

    return type(architecture, (), {"__init__": lambda self: setattr(self, "config", config), "__call__": call})()


def _fake_template_processor(processor_class):
    def init(self):
        self.kwargs = None
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[0, 10, 11]]),
            "attention_mask": torch.tensor([[0, 1, 1]]),
        }

    return type(processor_class, (), {"__init__": init, "apply_chat_template": apply_chat_template})()


def _request(model_key, *, content=None):
    return PrefillRequest(
        sample_id="sample-1",
        model_key=model_key,
        protocol="vt",
        condition="M2" if content is None else "M12",
        dataset_key="source",
        split="test",
        prompt_set_key="p8",
        prompt_id="p01",
        messages=(
            {
                "role": "user",
                "content": content or [{"type": "text", "text": "Describe the emotion."}],
            },
        ),
        media_paths={},
        use_audio_in_video=False,
    )


@pytest.mark.parametrize(
    "wrapper_cls,model_key,model_type,architecture,processor_class,location,layers,hidden",
    SPECS,
)
def test_native_visual_wrappers_extract_all_blocks_at_t0(
    tmp_path,
    wrapper_cls,
    model_key,
    model_type,
    architecture,
    processor_class,
    location,
    layers,
    hidden,
):
    processor = _fake_template_processor(processor_class)
    model = _fake_model(architecture, location=location, layers=layers, hidden=hidden)
    wrapper = wrapper_cls(
        model_key=model_key,
        model_path=_model_dir(
            tmp_path,
            model_type=model_type,
            architecture=architecture,
            location=location,
            layers=layers,
            hidden=hidden,
        ),
        device="cpu",
        model=model,
        processor=processor,
        runtime_versions={"transformers": "test"},
    )

    result = wrapper.extract_prefill(_request(model_key))

    assert result.trajectory.shape == (layers, hidden)
    np.testing.assert_allclose(result.trajectory[:, 0], np.arange(1, layers + 1))
    assert result.t0_token_index == 2
    assert result.provenance["hidden_state_index_offset"] == 1
    assert result.provenance["thinking_enabled"] is False
    assert model.call_kwargs["use_cache"] is False
    assert model.call_kwargs["output_hidden_states"] is True
    assert model.call_kwargs["logits_to_keep"] == 1
    if wrapper_cls is Glm4vWrapper:
        assert processor.kwargs["enable_thinking"] is False
    else:
        assert "enable_thinking" not in processor.kwargs


class Gemma3Processor:
    def __init__(self):
        self.messages = None
        self.images = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "prompt"

    def __call__(self, **kwargs):
        self.images = kwargs.get("images")
        return {
            "input_ids": torch.tensor([[0, 10, 11, 12]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
        }


class Gemma3ForConditionalGeneration:
    def __init__(self):
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=2, hidden_size=3)
        )
        self.call_kwargs = None

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        states = tuple(
            torch.full((1, 4, 3), float(index), dtype=torch.float32)
            for index in range(3)
        )
        return SimpleNamespace(hidden_states=states)


def test_gemma3_converts_video_to_ordered_image_placeholders(tmp_path, monkeypatch):
    path = _model_dir(
        tmp_path,
        model_type="gemma3",
        architecture="Gemma3ForConditionalGeneration",
        location="text_config",
        layers=2,
        hidden=3,
    )
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    frames = [Image.new("RGB", (2, 2), color=index) for index in range(3)]
    monkeypatch.setattr("mprisk.models.gemma3._uniform_video_frames", lambda p, n: frames)
    processor = Gemma3Processor()
    model = Gemma3ForConditionalGeneration()
    wrapper = Gemma3Wrapper(
        model_key="gemma3_4b",
        model_path=path,
        device="cpu",
        video_num_segments=3,
        model=model,
        processor=processor,
        runtime_versions={"transformers": "test"},
    )
    request = _request(
        "gemma3_4b",
        content=[
            {"type": "video", "video": str(media), "fps": 1.0},
            {"type": "text", "text": "Describe the emotion."},
        ],
    )

    result = wrapper.extract_prefill(request)

    assert result.trajectory.shape == (2, 3)
    assert [item["type"] for item in processor.messages[0]["content"]] == [
        "image",
        "image",
        "image",
        "text",
    ]
    assert len(processor.images[0]) == 3
    assert result.provenance["video_num_segments"] == 3
    assert result.provenance["video_frame_count"] == 3


def test_visual_wrappers_reject_audio_leakage(tmp_path):
    path = _model_dir(
        tmp_path,
        model_type="qwen2_5_vl",
        architecture="Qwen2_5_VLForConditionalGeneration",
        location="root",
        layers=2,
        hidden=3,
    )
    wrapper = Qwen2_5VlWrapper(
        model_key="qwen2_5_vl_7b",
        model_path=path,
        device="cpu",
        model=_fake_model(
            "Qwen2_5_VLForConditionalGeneration", location="root", layers=2, hidden=3
        ),
        processor=_fake_template_processor("Qwen2_5_VLProcessor"),
        runtime_versions={"transformers": "test"},
    )
    request = PrefillRequest(
        sample_id="sample",
        model_key="qwen2_5_vl_7b",
        protocol="vt",
        condition="M1",
        dataset_key="source",
        split="test",
        messages=({"role": "user", "content": [{"type": "audio", "audio": "x.wav"}]},),
        media_paths={},
        use_audio_in_video=False,
    )
    with pytest.raises(ValueError, match="Unsupported"):
        wrapper.extract_prefill(request)


def test_new_visual_families_are_registered() -> None:
    assert get_wrapper("gemma3") is Gemma3Wrapper
    assert get_wrapper("glm4v") is Glm4vWrapper
    assert get_wrapper("qwen2_5_vl") is Qwen2_5VlWrapper
