from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

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
        self.messages = None
        self.video_processor = SimpleNamespace(temporal_patch_size=2)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[0, 10, 11, 12]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
            "video_grid_thw": torch.tensor([[4, 2, 2]]),
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


def test_qwen3_5_extracts_all_blocks_and_disables_thinking(tmp_path, monkeypatch):
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
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    frames = [Image.new("RGB", (2, 2), color=index) for index in range(8)]
    monkeypatch.setattr(
        "mprisk.models.video_frame_utils.uniform_video_sample",
        lambda path, count: (
            frames,
            {
                "frames_indices": list(range(8)),
                "total_num_frames": 100,
                "fps": 25.0,
                "width": 2,
                "height": 2,
                "duration": 4.0,
                "video_backend": "test",
            },
        ),
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
                    {"type": "video", "video": str(media), "fps": 1.0},
                    {"type": "text", "text": "Describe the emotion."},
                ],
            },
        ),
        media_paths={"vision": str(media)},
        use_audio_in_video=False,
    )

    result = wrapper.extract_prefill(request)

    assert result.trajectory.shape == (32, 2560)
    np.testing.assert_allclose(result.trajectory[:, 0], np.arange(1, 33))
    assert result.token_count == 4
    assert result.t0_token_index == 3
    assert processor.kwargs["enable_thinking"] is False
    assert processor.kwargs["add_generation_prompt"] is True
    videos_kwargs = processor.kwargs["processor_kwargs"]["videos_kwargs"]
    assert videos_kwargs["do_sample_frames"] is False
    assert len(processor.messages[0]["content"][0]["video"]) == 8
    assert model.call_kwargs["use_cache"] is False
    assert model.call_kwargs["logits_to_keep"] == 1
    assert result.provenance["hidden_state_index_offset"] == 1
    assert result.provenance["requested_frames"] == 8
    assert result.provenance["actual_frames"] == 8
