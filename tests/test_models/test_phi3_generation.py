from __future__ import annotations

import json
import types

import torch

from mprisk.models.base_wrapper import GenerationRequest
from mprisk.models.phi3_vision import Phi3VisionWrapper


class Phi3VProcessor:
    def __init__(self) -> None:
        self.tokenizer = types.SimpleNamespace(eos_token_id=99)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def batch_decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return ["The person appears emotionally unsettled."]


class Phi3VForCausalLM:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(num_hidden_layers=2, hidden_size=3)
        self.generate_kwargs: dict[str, object] | None = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 20, 99]])


def test_phi3_generation_reuses_prefill_prepare_inputs_and_forwards_policy(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "phi3_v",
                "architectures": ["Phi3VForCausalLM"],
                "num_hidden_layers": 2,
                "hidden_size": 3,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    model = Phi3VForCausalLM()
    processor = Phi3VProcessor()
    monkeypatch.setattr(
        "mprisk.models.phi3_vision.request_text_and_frames",
        lambda request, video_num_segments: (
            request.messages[0]["content"][-1]["text"],
            [object()] * video_num_segments,
            {"requested_frames": video_num_segments, "actual_frames": video_num_segments},
        ),
    )
    wrapper = Phi3VisionWrapper(
        model_key="phi3_5_vision",
        model_path=model_path,
        device="cpu",
        dtype="bfloat16",
        video_num_segments=2,
        model=model,
        processor=processor,
        runtime_versions={"transformers": "test"},
    )
    request = GenerationRequest(
        sample_id="sample",
        model_key="phi3_5_vision",
        protocol="vt",
        condition="M12",
        messages=(
            {
                "role": "user",
                "content": (
                    {"type": "video", "video": "sample.mp4", "fps": 1.0},
                    {"type": "text", "text": "Describe affect in plain English."},
                ),
            },
        ),
        media_paths={"vision": "sample.mp4"},
        use_audio_in_video=False,
        generation_kwargs={"do_sample": False, "num_beams": 1, "max_new_tokens": 256},
    )

    result = wrapper.generate_conditioned(request)

    assert result.token_ids == (20, 99)
    assert result.finish_reason == "eos"
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["max_new_tokens"] == 256
    assert "Describe affect in plain English.<|end|>" in processor.calls[-1]["text"]
