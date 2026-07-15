from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mprisk.models.base_wrapper import GenerationRequest
from mprisk.models.qwen_omni import QwenOmniWrapper, build_condition_request


def _model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_5_omni",
                "thinker_config": {
                    "torch_dtype": "bfloat16",
                    "text_config": {"num_hidden_layers": 28, "hidden_size": 3584},
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return model_dir


def _content_types(request):
    return [item["type"] for item in request.messages[0]["content"]]


def test_build_va_conditions_prevents_audio_leakage(tmp_path) -> None:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    common = {
        "sample_id": "sample-1",
        "model_key": "qwen2_5_omni_7b",
        "protocol": "va",
        "dataset_key": "ch_sims_v2",
        "split": "test",
        "media_paths": {"vision": str(media), "audio": str(media)},
        "transcript": "unused",
        "task_prompt": "Identify the emotion.",
    }

    m1 = build_condition_request(condition="M1", **common)
    m2 = build_condition_request(condition="M2", **common)
    m12 = build_condition_request(condition="M12", **common)

    assert _content_types(m1) == ["video", "text"]
    assert m1.use_audio_in_video is False
    assert _content_types(m2) == ["audio", "text"]
    assert m2.use_audio_in_video is False
    assert _content_types(m12) == ["video", "text"]
    assert m12.use_audio_in_video is True


def test_build_vta_and_separate_audio_conditions_are_explicit(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    request = build_condition_request(
        sample_id="sample-1",
        model_key="qwen2_5_omni_7b",
        protocol="vta",
        condition="M12",
        dataset_key="dataset",
        split="test",
        media_paths={"vision": str(video), "audio": str(audio)},
        transcript="shared words",
        task_prompt="Identify the emotion.",
        joint_audio_mode="separate_file",
    )

    assert _content_types(request) == ["video", "audio", "text"]
    assert request.use_audio_in_video is False
    assert "shared words" in request.messages[0]["content"][-1]["text"]

    with pytest.raises(ValueError, match="same media file"):
        build_condition_request(
            sample_id="sample-1",
            model_key="qwen2_5_omni_7b",
            protocol="vta",
            condition="M12",
            dataset_key="dataset",
            split="test",
            media_paths={"vision": str(video), "audio": str(audio)},
            transcript="shared words",
            task_prompt="Identify the emotion.",
            joint_audio_mode="embedded_video",
        )


class _FakeProcessor:
    def __init__(self) -> None:
        self.call_kwargs = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages
        return "rendered prompt"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[0, 10, 11, 12]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
        }


class _FakeThinker:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=28, hidden_size=3584)
        )
        self.call_kwargs = None

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        states = tuple(
            torch.full((1, 4, 3584), float(index), dtype=torch.float32) for index in range(29)
        )
        return SimpleNamespace(hidden_states=states)


def test_wrapper_generation_decodes_only_new_tokens_and_uses_greedy_kwargs(tmp_path) -> None:
    class Processor(_FakeProcessor):
        tokenizer = SimpleNamespace(eos_token_id=99)

        def batch_decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert skip_special_tokens is True
            assert clean_up_tokenization_spaces is False
            assert token_ids.tolist() == [[42, 99]]
            return ["The person appears emotionally unsettled."]

    class Thinker(_FakeThinker):
        generation_config = SimpleNamespace(eos_token_id=None)

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return torch.tensor([[0, 10, 11, 12, 42, 99]])

    model = Thinker()
    wrapper = QwenOmniWrapper(
        model_key="qwen2_5_omni_7b",
        model_path=_model_dir(tmp_path),
        device="cpu",
        model=model,
        processor=Processor(),
        process_mm_info_fn=lambda messages, use_audio_in_video: (None, None, None),
        runtime_versions={"transformers": "test", "qwen-omni-utils": "test"},
    )
    request = GenerationRequest(
        sample_id="sample-1",
        model_key="qwen2_5_omni_7b",
        protocol="va",
        condition="M12",
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "sample.mp4"},
                    {"type": "text", "text": "Prompt"},
                ],
            },
        ),
        media_paths={"vision": "sample.mp4", "audio": "sample.mp4"},
        use_audio_in_video=True,
        generation_kwargs={"do_sample": False, "num_beams": 1, "max_new_tokens": 32},
    )

    result = wrapper.generate_conditioned(request)

    assert result.token_ids == (42, 99)
    assert result.text == "The person appears emotionally unsettled."
    assert result.eos_token_ids == (99,)
    assert result.finish_reason == "eos"
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["num_beams"] == 1
    assert model.generate_kwargs["max_new_tokens"] == 32
    assert model.generate_kwargs["eos_token_id"] == 99
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs


def test_wrapper_extracts_last_non_padding_token_from_28_thinker_blocks(tmp_path) -> None:
    model = _FakeThinker()
    processor = _FakeProcessor()
    process_calls = []

    def process_mm_info(messages, *, use_audio_in_video):
        process_calls.append((messages, use_audio_in_video))
        return None, None, None

    wrapper = QwenOmniWrapper(
        model_key="qwen2_5_omni_7b",
        model_path=_model_dir(tmp_path),
        device="cpu",
        model=model,
        processor=processor,
        process_mm_info_fn=process_mm_info,
        runtime_versions={"transformers": "test", "qwen-omni-utils": "test"},
    )
    request = build_condition_request(
        sample_id="sample-1",
        model_key="qwen2_5_omni_7b",
        protocol="vt",
        condition="M2",
        dataset_key="dataset",
        split="test",
        media_paths={},
        transcript="spoken words",
        task_prompt="Identify the emotion.",
    )

    result = wrapper.extract_prefill(request)

    assert result.trajectory.shape == (28, 3584)
    np.testing.assert_allclose(result.trajectory[:, 0], np.arange(1, 29))
    assert result.t0_token_index == 3
    assert result.token_count == 4
    assert process_calls[0][1] is False
    assert model.call_kwargs["use_cache"] is False
    assert model.call_kwargs["output_hidden_states"] is True
    assert model.call_kwargs["return_dict"] is True
    assert model.call_kwargs["use_audio_in_video"] is False
    assert result.provenance["hidden_state_index_offset"] == 1
    assert result.provenance["talker_loaded"] is False


def test_wrapper_rejects_wrong_hidden_state_count(tmp_path) -> None:
    class BadThinker(_FakeThinker):
        def __call__(self, **kwargs):
            return SimpleNamespace(hidden_states=(torch.zeros((1, 4, 3584)),))

    wrapper = QwenOmniWrapper(
        model_key="qwen2_5_omni_7b",
        model_path=_model_dir(tmp_path),
        device="cpu",
        model=BadThinker(),
        processor=_FakeProcessor(),
        process_mm_info_fn=lambda messages, use_audio_in_video: (None, None, None),
        runtime_versions={"transformers": "test", "qwen-omni-utils": "test"},
    )
    request = build_condition_request(
        sample_id="sample-1",
        model_key="qwen2_5_omni_7b",
        protocol="vt",
        condition="M2",
        dataset_key="dataset",
        split="test",
        media_paths={},
        transcript="spoken words",
        task_prompt="Identify the emotion.",
    )

    with pytest.raises(ValueError, match="29 hidden-state tensors"):
        wrapper.extract_prefill(request)
