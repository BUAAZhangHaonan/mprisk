from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.llava import LlavaOneVisionWrapper, LlavaV15Wrapper
from mprisk.models.minicpm_v import MiniCpmVWrapper
from mprisk.models.phi3_vision import Phi3VisionWrapper
from mprisk.models.wrapper_registry import get_wrapper
from mprisk.models.video_frame_utils import validate_video_grid_frames


def _model_dir(tmp_path, *, model_type, architecture, dtype, location="root"):
    path = tmp_path / architecture
    path.mkdir()
    language = {
        "num_hidden_layers": 2,
        "hidden_size": 3,
        "torch_dtype": dtype,
    }
    payload = {
        "model_type": model_type,
        "architectures": [architecture],
        "torch_dtype": dtype,
    }
    if location == "root":
        payload.update(language)
    else:
        payload["text_config"] = language
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return path


def _request(model_key, media, *, condition="M12"):
    content = []
    if condition != "M2":
        content.append({"type": "video", "video": str(media), "fps": 1.0})
    content.append({"type": "text", "text": "Describe the overall affect."})
    return PrefillRequest(
        sample_id="sample",
        model_key=model_key,
        protocol="vt",
        condition=condition,
        dataset_key="source",
        split="train",
        prompt_set_key="p8",
        prompt_id="p001",
        messages=({"role": "user", "content": content},),
        media_paths={"vision": str(media)},
        use_audio_in_video=False,
    )


def _fake_model(name, *, location="root"):
    language = SimpleNamespace(num_hidden_layers=2, hidden_size=3)
    config = language if location == "root" else SimpleNamespace(text_config=language)

    def call(self, **kwargs):
        self.kwargs = kwargs
        mask = kwargs.get("attention_mask")
        if mask is None:
            mask = kwargs["data"]["input_ids"].new_ones(kwargs["data"]["input_ids"].shape)
        states = tuple(
            torch.full((1, int(mask.shape[-1]), 3), float(index))
            for index in range(3)
        )
        return SimpleNamespace(hidden_states=states)

    return type(
        name,
        (),
        {"__init__": lambda self: setattr(self, "config", config), "__call__": call},
    )()


class LlavaProcessor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "prompt"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class LlavaOnevisionProcessor(LlavaProcessor):
    pass


@pytest.mark.parametrize(
    "wrapper_cls,model_key,model_type,architecture,processor_cls,frame_count",
    [
        (
            LlavaV15Wrapper,
            "llava_v1_5_7b",
            "llava",
            "LlavaForConditionalGeneration",
            LlavaProcessor,
            7,
        ),
        (
            LlavaOneVisionWrapper,
            "llava_onevision_qwen2_7b",
            "llava_onevision",
            "LlavaOnevisionForConditionalGeneration",
            LlavaOnevisionProcessor,
            8,
        ),
    ],
)
def test_llava_video_uses_family_limit_and_preserves_frame_order(
    tmp_path,
    monkeypatch,
    wrapper_cls,
    model_key,
    model_type,
    architecture,
    processor_cls,
    frame_count,
):
    frames = [Image.new("RGB", (2, 2), color=index) for index in range(frame_count)]
    monkeypatch.setattr(
        "mprisk.models.video_frame_utils.uniform_video_sample",
        lambda path, count: (
            frames,
            {
                "frames_indices": list(range(frame_count)),
                "total_num_frames": 100,
            },
        ),
    )
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    processor = processor_cls()
    model = _fake_model(architecture, location="text_config")
    model_path = _model_dir(
        tmp_path,
        model_type=model_type,
        architecture=architecture,
        dtype="float16",
        location="text_config",
    )
    if wrapper_cls is LlavaV15Wrapper:
        from safetensors.torch import save_file

        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        config["text_config"]["model_type"] = "llama"
        (model_path / "config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        tensors = {
            "language_model.model.embed_tokens.weight": torch.zeros((7, 3)),
            "language_model.model.layers.0.input_layernorm.weight": torch.zeros(3),
            "language_model.model.layers.1.input_layernorm.weight": torch.zeros(3),
        }
        save_file(tensors, model_path / "model-00001-of-00001.safetensors")
        (model_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        key: "model-00001-of-00001.safetensors" for key in tensors
                    }
                }
            ),
            encoding="utf-8",
        )
    wrapper = wrapper_cls(
        model_key=model_key,
        model_path=model_path,
        device="cpu",
        dtype="float16",
        model=model,
        processor=processor,
        runtime_versions={"transformers": "test"},
    )

    result = wrapper.extract_prefill(_request(model_key, media))

    assert result.trajectory.shape == (2, 3)
    np.testing.assert_allclose(result.trajectory[:, 0], [1.0, 2.0])
    assert [item["type"] for item in processor.messages[0]["content"]] == [
        *(["image"] * frame_count),
        "text",
    ]
    assert len(processor.call_kwargs["images"]) == frame_count
    assert result.provenance["video_frame_count"] == frame_count
    assert result.provenance["requested_frames"] == frame_count
    assert result.provenance["actual_frames"] == frame_count


class _MiniTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def convert_tokens_to_ids(self, token):
        return len(token)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "prompt"

    def __call__(self, *args, **kwargs):
        return {
            "input_ids": torch.tensor([[4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class MiniCPMVProcessor:
    def __init__(self):
        self.tokenizer = _MiniTokenizer()

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": [[]],
            "tgt_sizes": [],
            "image_bound": [torch.empty((0, 2), dtype=torch.long)],
        }


def test_minicpm_text_only_has_explicit_empty_visual_contract(tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    processor = MiniCPMVProcessor()
    model = _fake_model("MiniCPMV")
    wrapper = MiniCpmVWrapper(
        model_key="minicpm_v_2_6",
        model_path=_model_dir(
            tmp_path,
            model_type="minicpmv",
            architecture="MiniCPMV",
            dtype="bfloat16",
        ),
        device="cpu",
        model=model,
        processor=processor,
        runtime_versions={"transformers": "test"},
    )

    result = wrapper.extract_prefill(
        _request("minicpm_v_2_6", media, condition="M2")
    )

    assert result.trajectory.shape == (2, 3)
    assert model.kwargs["data"]["pixel_values"] == [[]]
    assert model.kwargs["data"]["tgt_sizes"] == []
    assert "position_ids" in model.kwargs["data"]


class Phi3VProcessor:
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


def test_phi35_prompt_numbers_all_video_frames(tmp_path, monkeypatch):
    frames = [Image.new("RGB", (2, 2), color=index) for index in range(8)]
    monkeypatch.setattr(
        "mprisk.models.video_frame_utils.uniform_video_sample",
        lambda path, count: (
            frames,
            {"frames_indices": list(range(8)), "total_num_frames": 100},
        ),
    )
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    processor = Phi3VProcessor()
    model = _fake_model("Phi3VForCausalLM")
    wrapper = Phi3VisionWrapper(
        model_key="phi3_5_vision",
        model_path=_model_dir(
            tmp_path,
            model_type="phi3_v",
            architecture="Phi3VForCausalLM",
            dtype="bfloat16",
        ),
        device="cpu",
        model=model,
        processor=processor,
        runtime_versions={"transformers": "test"},
    )

    result = wrapper.extract_prefill(_request("phi3_5_vision", media))

    assert result.trajectory.shape == (2, 3)
    assert processor.kwargs["text"].count("<|image_") == 8
    assert processor.kwargs["text"].index("<|image_1|>") < processor.kwargs[
        "text"
    ].index("<|image_8|>")
    assert len(processor.kwargs["images"]) == 8


def test_video_frame_families_are_registered():
    assert get_wrapper("llava_v15") is LlavaV15Wrapper
    assert get_wrapper("llava_onevision") is LlavaOneVisionWrapper
    assert get_wrapper("minicpm_v") is MiniCpmVWrapper
    assert get_wrapper("phi3_vision") is Phi3VisionWrapper


def test_video_grid_validation_fails_closed_on_processor_resampling():
    processor = SimpleNamespace(
        video_processor=SimpleNamespace(temporal_patch_size=2)
    )
    with pytest.raises(ValueError, match="requested 8.*retained 6"):
        validate_video_grid_frames(
            {"video_grid_thw": torch.tensor([[3, 2, 2]])},
            processor=processor,
            requested_frames=8,
            family="test_family",
        )
