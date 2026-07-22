from __future__ import annotations

import json
import types

import numpy as np
import pytest
import torch

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.phi4_mm import Phi4MmWrapper
from mprisk.models.wrapper_registry import get_wrapper


def test_phi4_is_registered() -> None:
    assert get_wrapper("phi4_multimodal") is Phi4MmWrapper


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *, text, images, audios, return_tensors):
        assert return_tensors == "pt"
        self.calls.append({"text": text, "images": images, "audios": audios})
        mode = (1 if images else 0) + (2 if audios else 0)
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "input_mode": torch.tensor([mode]),
            "input_image_embeds": torch.ones((1, 1)) if images else torch.tensor([]),
            "input_audio_embeds": torch.ones((1, 1)) if audios else torch.tensor([]),
        }


class _FakeModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(num_hidden_layers=2, hidden_size=3)

    def __call__(self, **kwargs):
        assert kwargs["use_cache"] is False
        assert kwargs["output_hidden_states"] is True
        hidden_states = tuple(
            torch.full((1, 5, 3), float(index)) for index in range(3)
        )
        return types.SimpleNamespace(hidden_states=hidden_states)


def _request(tmp_path, condition: str) -> PrefillRequest:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    content = []
    use_audio_in_video = False
    if condition in {"M1", "M12"}:
        content.append({"type": "video", "video": str(video), "fps": 1.0})
    if condition == "M2":
        content.append({"type": "audio", "audio": str(video)})
    if condition == "M12":
        use_audio_in_video = True
    content.append({"type": "text", "text": "Describe the overall affect."})
    return PrefillRequest(
        sample_id=f"sample-{condition}",
        model_key="phi4_multimodal",
        protocol="va",
        condition=condition,
        dataset_key="fixture",
        split="test",
        messages=({"role": "user", "content": content},),
        media_paths={"vision": str(video), "audio": str(video)},
        use_audio_in_video=use_audio_in_video,
        prompt_set_key="fixture-p8",
        prompt_id="p01",
    )


@pytest.mark.parametrize(
    ("condition", "expected_images", "expected_audio", "expected_tokens"),
    [
        ("M1", 2, 0, ("<|image_1|>", "<|image_2|>")),
        ("M2", 0, 1, ("<|audio_1|>",)),
        ("M12", 2, 1, ("<|image_1|>", "<|image_2|>", "<|audio_1|>")),
    ],
)
def test_phi4_va_prefill_contract(
    tmp_path,
    monkeypatch,
    condition,
    expected_images,
    expected_audio,
    expected_tokens,
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Phi4MMForCausalLM"],
                "num_hidden_layers": 2,
                "hidden_size": 3,
            }
        ),
        encoding="utf-8",
    )
    (model_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "mprisk.models.phi4_mm._uniform_video_frames",
        lambda path, count: [object(), object()],
    )
    monkeypatch.setattr(
        "mprisk.models.phi4_mm._decode_audio",
        lambda path: (np.ones(160, dtype=np.float32), 16000),
    )
    processor = _FakeProcessor()
    wrapper = Phi4MmWrapper(
        model_key="phi4_multimodal",
        model_path=model_path,
        device="cpu",
        video_num_segments=2,
        model=_FakeModel(),
        processor=processor,
        runtime_versions={
            "transformers": "4.48.2",
            "peft": "0.13.2",
            "torch": torch.__version__,
        },
    )

    result = wrapper.extract_prefill(_request(tmp_path, condition))

    assert result.trajectory.shape == (2, 3)
    assert result.t0_token_index == 4
    assert result.token_count == 5
    assert result.provenance["hidden_state_index_offset"] == 1
    call = processor.calls[-1]
    assert len(call["images"] or []) == expected_images
    assert len(call["audios"] or []) == expected_audio
    assert all(token in call["text"] for token in expected_tokens)
    assert call["text"].endswith("<|end|><|assistant|>")


def test_phi4_rejects_non_va_protocol(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Phi4MMForCausalLM"],
                "num_hidden_layers": 2,
                "hidden_size": 3,
            }
        ),
        encoding="utf-8",
    )
    wrapper = Phi4MmWrapper(
        model_key="phi4_multimodal",
        model_path=model_path,
        device="cpu",
        model=_FakeModel(),
        processor=_FakeProcessor(),
        runtime_versions={"transformers": "4.48.2", "peft": "0.13.2"},
    )
    request = PrefillRequest(
        sample_id="x",
        model_key="phi4_multimodal",
        protocol="vt",
        condition="M1",
        dataset_key="fixture",
        split="test",
        messages=({"role": "user", "content": ({"type": "text", "text": "x"},)},),
        media_paths={},
        use_audio_in_video=False,
    )
    with pytest.raises(ValueError, match="supports VA only"):
        wrapper.extract_prefill(request)
