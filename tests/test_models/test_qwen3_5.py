from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from mprisk.models.qwen3_5 import Qwen3_5Wrapper


def _model_dir(tmp_path):
    model_dir = tmp_path / "Qwen3.5-4B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "dtype": "bfloat16",
                    "num_hidden_layers": 32,
                    "hidden_size": 2560,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return model_dir


class _Processor:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[0, 10, 11, 12]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
        }


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=32, hidden_size=2560)
        )
        self.call_kwargs = None

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return SimpleNamespace(
            hidden_states=tuple(
                torch.full((1, 4, 2560), float(index), dtype=torch.float32)
                for index in range(33)
            )
        )


def test_qwen3_5_extracts_all_blocks_and_disables_thinking(tmp_path):
    from mprisk.models.base_wrapper import PrefillRequest

    model = _Model()
    processor = _Processor()
    wrapper = Qwen3_5Wrapper(
        model_key="qwen3_5_4b",
        model_path=_model_dir(tmp_path),
        device="cpu",
        model=model,
        processor=processor,
        runtime_versions={"transformers": "test"},
    )
    request = PrefillRequest(
        sample_id="sample-1",
        model_key="qwen3_5_4b",
        protocol="vt",
        condition="M12",
        prompt_set_key="p8",
        prompt_id="p1",
        dataset_key="dataset",
        split="test",
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "sample.mp4", "fps": 1.0},
                    {"type": "text", "text": "Describe the emotion."},
                ],
            },
        ),
        media_paths={"vision": "sample.mp4"},
        use_audio_in_video=False,
    )

    result = wrapper.extract_prefill(request)

    assert result.trajectory.shape == (32, 2560)
    np.testing.assert_allclose(result.trajectory[:, 0], np.arange(1, 33))
    assert result.token_count == 4
    assert result.t0_token_index == 3
    assert processor.kwargs["enable_thinking"] is False
    assert processor.kwargs["add_generation_prompt"] is True
    assert processor.kwargs["processor_kwargs"] == {
        "videos_kwargs": {"fps": 1.0}
    }
    assert model.call_kwargs["use_cache"] is False
    assert model.call_kwargs["logits_to_keep"] == 1
    assert result.provenance["hidden_state_index_offset"] == 1
