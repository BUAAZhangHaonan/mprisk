from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from mprisk.models.internvl import InternVlWrapper
from mprisk.models.qwen_omni import build_condition_request


def _model_dir(tmp_path):
    model_dir = tmp_path / "InternVL3_5-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "internvl_chat",
                "architectures": ["InternVLChatModel"],
                "dtype": "bfloat16",
                "force_image_size": 448,
                "llm_config": {
                    "model_type": "qwen3",
                    "num_hidden_layers": 3,
                    "hidden_size": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return model_dir


class _FakeConversation:
    roles = ("user", "assistant")
    sep = "<|end|>"

    def __init__(self) -> None:
        self.system_message = "system"
        self.messages = []

    def append_message(self, role, value):
        self.messages.append((role, value))

    def get_prompt(self):
        return "\n".join("" if value is None else value for _, value in self.messages)


class _FakeTokenizer:
    padding_side = "right"

    def __init__(self) -> None:
        self.query = None

    def convert_tokens_to_ids(self, token):
        return {"<IMG_CONTEXT>": 99, "<|end|>": 2}[token]

    def __call__(self, query, **kwargs):
        self.query = query
        image_count = query.count("<IMG_CONTEXT>")
        ids = [0] + [99] * image_count + [10, 11]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([[0] + [1] * (len(ids) - 1)]),
        }


class _FakeEmbedding:
    def __call__(self, input_ids):
        return torch.zeros((*input_ids.shape, 4), dtype=torch.float32)


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.call_kwargs = None

    def get_input_embeddings(self):
        return _FakeEmbedding()

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        seq = int(kwargs["inputs_embeds"].shape[1])
        states = tuple(
            torch.full((1, seq, 4), float(index), dtype=torch.float32) for index in range(4)
        )
        return SimpleNamespace(hidden_states=states)


class _FakeInternVl:
    num_image_token = 1
    system_message = "system"

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            force_image_size=448,
            llm_config=SimpleNamespace(num_hidden_layers=3, hidden_size=4),
        )
        self.conv_template = _FakeConversation()
        self.language_model = _FakeLanguageModel()
        self.chat_called = False

    def extract_feature(self, pixel_values):
        return torch.arange(pixel_values.shape[0] * 4, dtype=torch.float32).reshape(
            pixel_values.shape[0], 1, 4
        )

    def chat(self, *args, **kwargs):
        self.chat_called = True
        raise AssertionError("chat must not be used for hidden-state extraction")


def _load_video(path, *, input_size, max_num, num_segments):
    assert path.endswith("sample.mp4")
    assert input_size == 448
    assert max_num == 1
    assert num_segments == 2
    return torch.ones((2, 3, 2, 2)), [1, 1]


def test_internvl_uses_official_video_prefix_and_explicit_language_forward(tmp_path) -> None:
    model = _FakeInternVl()
    tokenizer = _FakeTokenizer()
    wrapper = InternVlWrapper(
        model_key="internvl3_5_8b",
        model_path=_model_dir(tmp_path),
        device="cpu",
        video_num_segments=2,
        model=model,
        tokenizer=tokenizer,
        load_video_fn=_load_video,
        runtime_versions={"transformers": "test", "decord": "test"},
    )
    request = build_condition_request(
        sample_id="sample-1",
        model_key="internvl3_5_8b",
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

    assert result.trajectory.shape == (3, 4)
    np.testing.assert_allclose(result.trajectory[:, 0], np.arange(1, 4))
    assert "Frame1: <img><IMG_CONTEXT></img>" in tokenizer.query
    assert "Frame2: <img><IMG_CONTEXT></img>" in tokenizer.query
    assert model.chat_called is False
    assert model.language_model.call_kwargs["use_cache"] is False
    assert model.language_model.call_kwargs["output_hidden_states"] is True
    assert model.language_model.call_kwargs["return_dict"] is True
    assert model.language_model.call_kwargs["logits_to_keep"] == 1
    assert result.t0_token_index == result.token_count - 1
    assert result.provenance["num_patches_list"] == [1, 1]


def test_internvl_text_only_condition_never_loads_visual_input(tmp_path) -> None:
    calls = []

    def forbidden_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("M2 must not load video")

    model = _FakeInternVl()
    tokenizer = _FakeTokenizer()
    wrapper = InternVlWrapper(
        model_key="internvl3_5_8b",
        model_path=_model_dir(tmp_path),
        device="cpu",
        video_num_segments=2,
        model=model,
        tokenizer=tokenizer,
        load_video_fn=forbidden_loader,
        runtime_versions={"transformers": "test", "decord": "test"},
    )
    request = build_condition_request(
        sample_id="sample-1",
        model_key="internvl3_5_8b",
        protocol="vt",
        condition="M2",
        prompt_set_key="main_p8",
        prompt_id="p01",
        dataset_key="dataset",
        split="test",
        media_paths={"vision": str(tmp_path / "sample.mp4")},
        transcript="spoken words",
        task_prompt="Identify the emotion.",
    )

    result = wrapper.extract_prefill(request)

    assert calls == []
    assert "<IMG_CONTEXT>" not in tokenizer.query
    assert result.trajectory.shape == (3, 4)
