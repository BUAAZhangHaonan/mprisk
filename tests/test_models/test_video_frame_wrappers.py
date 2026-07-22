from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.llava import (
    LlavaOneVisionWrapper,
    LlavaV15Wrapper,
    _llava_v15_runtime_contracts,
    _validate_llava_v15_processor_tokens,
    _validate_llava_v15_sampled_frames,
)
from mprisk.models.minicpm_v import MiniCpmVWrapper
from mprisk.models.phi3_vision import Phi3VisionWrapper
from mprisk.models.video_frame_utils import validate_video_grid_frames
from mprisk.models.wrapper_registry import get_wrapper


def _model_dir(tmp_path, *, model_type, architecture, dtype, location="root"):
    path = tmp_path / architecture
    path.mkdir()
    language = {
        "num_hidden_layers": 2,
        "hidden_size": 3,
        "max_position_embeddings": 4096,
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


def _request(model_key, media, *, condition="M12", runtime_contracts=None):
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
        runtime_contracts={} if runtime_contracts is None else runtime_contracts,
    )


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _llava_runtime_contracts(
    *,
    selected_frames,
    candidate_counts,
    total_frames=100,
    sample_id="sample",
):
    prompt_ids = [f"p{index:03d}" for index in range(1, 9)]
    prompt_sha = _canonical_sha256(prompt_ids)
    video_path = "/tmp/sample.mp4"
    indices = [
        min(
            total_frames - 1,
            int((index + 0.5) * total_frames / selected_frames),
        )
        for index in range(selected_frames)
    ]
    return {
        "context_budget_contract": {
            "schema": "mprisk_llava_v15_context_budget_contract_v1",
            "mode": "per_sample_shared_max_legal",
            "sample_id": sample_id,
            "max_position_embeddings": 4096,
            "max_candidate_frames": 8,
            "selected_frames": selected_frames,
            "conditions": ["M1", "M12"],
            "prompt_set_key": "p8",
            "prompt_ids": prompt_ids,
            "prompt_ids_sha256": prompt_sha,
            "candidate_max_token_counts": {
                str(key): value for key, value in candidate_counts.items()
            },
            "candidate_condition_max_token_counts": {
                str(key): {"M1": value, "M12": value}
                for key, value in candidate_counts.items()
            },
            "selected_max_token_count": candidate_counts[selected_frames],
            "selection_rule": (
                "largest_f_with_all_p8_m1_m12_tokens_lte_context"
            ),
            "no_truncation": True,
        },
        "frame_selection_contract": {
            "schema": "mprisk_llava_v15_shared_frame_selection_v1",
            "sample_id": sample_id,
            "sampling_method": "uniform_midpoint_decord_v1",
            "video_path": video_path,
            "selected_frames": selected_frames,
            "source_total_frames": total_frames,
            "frame_indices": indices,
            "frame_indices_sha256": _canonical_sha256(indices),
            "shared_conditions": ["M1", "M12"],
            "prompt_ids_sha256": prompt_sha,
        },
    }


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
    def __init__(self, token_count=3):
        self.token_count = token_count

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "prompt"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        input_ids = torch.arange(self.token_count).unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
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
    candidate_counts = {
        1: 1000,
        2: 1500,
        3: 2000,
        4: 2500,
        5: 3000,
        6: 3500,
        7: 4096,
        8: 4600,
    }
    runtime_contracts = _llava_runtime_contracts(
        selected_frames=frame_count,
        candidate_counts=candidate_counts,
    )
    expected_indices = (
        runtime_contracts["frame_selection_contract"]["frame_indices"]
        if wrapper_cls is LlavaV15Wrapper
        else list(range(frame_count))
    )
    monkeypatch.setattr(
        "mprisk.models.video_frame_utils.uniform_video_sample",
        lambda path, count: (
            frames,
            {
                "frames_indices": expected_indices,
                "total_num_frames": 100,
            },
        ),
    )
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    runtime_contracts["frame_selection_contract"]["video_path"] = str(
        media.resolve()
    )
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

    request_contracts = runtime_contracts if wrapper_cls is LlavaV15Wrapper else None
    result = wrapper.extract_prefill(
        _request(model_key, media, runtime_contracts=request_contracts)
    )

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
    if wrapper_cls is LlavaV15Wrapper:
        assert result.provenance["context_budget_contract"] == runtime_contracts[
            "context_budget_contract"
        ]
        assert result.provenance["frame_selection_contract"] == runtime_contracts[
            "frame_selection_contract"
        ]


@pytest.mark.parametrize(
    ("selected_frames", "candidate_counts", "processor_tokens"),
    [
        (
            7,
            {1: 1000, 2: 1500, 3: 2000, 4: 2500, 5: 3000, 6: 3500, 7: 4096, 8: 4600},
            4096,
        ),
        (
            6,
            {1: 1000, 2: 1500, 3: 2000, 4: 2500, 5: 3000, 6: 4000, 7: 4106, 8: 4600},
            4000,
        ),
    ],
)
def test_llava_v15_consumes_source_f7_and_target_f6_plans(
    tmp_path,
    monkeypatch,
    selected_frames,
    candidate_counts,
    processor_tokens,
):
    contracts = _llava_runtime_contracts(
        selected_frames=selected_frames,
        candidate_counts=candidate_counts,
    )
    expected_indices = contracts["frame_selection_contract"]["frame_indices"]
    monkeypatch.setattr(
        "mprisk.models.video_frame_utils.uniform_video_sample",
        lambda path, count: (
            [Image.new("RGB", (2, 2), color=index) for index in range(count)],
            {"frames_indices": expected_indices, "total_num_frames": 100},
        ),
    )
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    contracts["frame_selection_contract"]["video_path"] = str(media.resolve())
    model_path = _model_dir(
        tmp_path,
        model_type="llava",
        architecture="LlavaForConditionalGeneration",
        dtype="float16",
        location="text_config",
    )
    from safetensors.torch import save_file

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config["text_config"]["model_type"] = "llama"
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
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
    wrapper = LlavaV15Wrapper(
        model_key="llava_v1_5_7b",
        model_path=model_path,
        device="cpu",
        dtype="float16",
        model=_fake_model("LlavaForConditionalGeneration", location="text_config"),
        processor=LlavaProcessor(token_count=processor_tokens),
        runtime_versions={"transformers": "test"},
    )

    result = wrapper.extract_prefill(
        _request("llava_v1_5_7b", media, runtime_contracts=contracts)
    )

    assert result.token_count == processor_tokens
    assert result.provenance["actual_frames"] == selected_frames
    assert result.provenance["video_frame_indices"] == [expected_indices]


def test_llava_v15_rejects_missing_or_nonmaximal_frame_plan(tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    request = _request("llava_v1_5_7b", media)
    with pytest.raises(ValueError, match="requires exactly"):
        _llava_v15_runtime_contracts(
            request,
            max_position_embeddings=4096,
            max_candidate_frames=8,
        )

    counts = {1: 1000, 2: 1500, 3: 2000, 4: 2500, 5: 3000, 6: 3500, 7: 4000, 8: 4600}
    contracts = _llava_runtime_contracts(selected_frames=6, candidate_counts=counts)
    contracts["frame_selection_contract"]["video_path"] = str(media.resolve())
    request = _request("llava_v1_5_7b", media, runtime_contracts=contracts)
    with pytest.raises(ValueError, match="not the largest legal"):
        _llava_v15_runtime_contracts(
            request,
            max_position_embeddings=4096,
            max_candidate_frames=8,
        )


def test_llava_v15_rejects_processor_context_overflow(tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    counts = {1: 1000, 2: 1500, 3: 2000, 4: 2500, 5: 3000, 6: 3500, 7: 4096, 8: 4600}
    contracts = _llava_runtime_contracts(selected_frames=7, candidate_counts=counts)
    contracts["frame_selection_contract"]["video_path"] = str(media.resolve())
    request = _request("llava_v1_5_7b", media, runtime_contracts=contracts)
    with pytest.raises(ValueError, match="4097 tokens.*limit 4096"):
        _validate_llava_v15_processor_tokens(
            request,
            model_inputs={"attention_mask": torch.ones((1, 4097), dtype=torch.long)},
            context_contract=contracts["context_budget_contract"],
        )


def test_llava_v15_rejects_frame_indices_that_differ_from_plan(tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    counts = {
        1: 1000,
        2: 1500,
        3: 2000,
        4: 2500,
        5: 3000,
        6: 3500,
        7: 4096,
        8: 4600,
    }
    contracts = _llava_runtime_contracts(selected_frames=7, candidate_counts=counts)
    contracts["frame_selection_contract"]["video_path"] = str(media.resolve())
    request = _request("llava_v1_5_7b", media, runtime_contracts=contracts)

    with pytest.raises(ValueError, match="decoded frame indices differ"):
        _validate_llava_v15_sampled_frames(
            request,
            provenance={
                "actual_frames": 7,
                "video_frame_indices": [[0, 1, 2, 3, 4, 5, 6]],
                "video_source_total_frames": [100],
            },
            frame_contract=contracts["frame_selection_contract"],
        )


def test_llava_v15_rejects_condition_maximum_that_does_not_form_global_max(
    tmp_path,
):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    counts = {
        1: 1000,
        2: 1500,
        3: 2000,
        4: 2500,
        5: 3000,
        6: 3500,
        7: 4096,
        8: 4600,
    }
    contracts = _llava_runtime_contracts(selected_frames=7, candidate_counts=counts)
    contracts["frame_selection_contract"]["video_path"] = str(media.resolve())
    contracts["context_budget_contract"][
        "candidate_condition_max_token_counts"
    ]["7"] = {"M1": 4000, "M12": 4000}
    request = _request("llava_v1_5_7b", media, runtime_contracts=contracts)

    with pytest.raises(ValueError, match="must equal its M1/M12 maximum"):
        _llava_v15_runtime_contracts(
            request,
            max_position_embeddings=4096,
            max_candidate_frames=8,
        )


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
