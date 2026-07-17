from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mprisk.cache.kv_prefill import (
    QwenVlPromptKvPrefillExtractor,
    _assert_identical_token_prefix,
    _clone_pristine_dynamic_cache,
    _longest_common_prefix_length,
    _require_isolated_python_environment,
)


class _Layer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values
        self.is_initialized = True

    def update(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = torch.cat((self.keys, keys), dim=-2)
        self.values = torch.cat((self.values, values), dim=-2)


class _Cache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.layers = [_Layer(keys, values)]

    def get_seq_length(self) -> int:
        return int(self.layers[0].keys.shape[-2])


def test_python_environment_must_disable_user_site(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    monkeypatch.setattr("mprisk.cache.kv_prefill.site.ENABLE_USER_SITE", True)
    with pytest.raises(RuntimeError, match="PYTHONNOUSERSITE=1"):
        _require_isolated_python_environment()

    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr("mprisk.cache.kv_prefill.site.ENABLE_USER_SITE", False)
    _require_isolated_python_environment()


def test_prefix_contract_checks_every_token_aligned_field() -> None:
    common = {
        "input_ids": torch.tensor([[11, 12, 13, 21]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "mm_token_type_ids": torch.tensor([[0, 2, 2, 0]]),
    }
    other = {key: value.clone() for key, value in common.items()}
    other["input_ids"][0, 3] = 22
    _assert_identical_token_prefix([common, other], 3)

    other["mm_token_type_ids"][0, 2] = 0
    with pytest.raises(RuntimeError, match="exact mm_token_type_ids prefix"):
        _assert_identical_token_prefix([common, other], 3)


def test_pristine_cache_clone_is_order_independent_and_source_is_immutable() -> None:
    prefix_keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
    prefix_values = prefix_keys + 100
    prefix = _Cache(prefix_keys, prefix_values)

    def run(order: tuple[int, ...]) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        for suffix_id in order:
            cloned = _clone_pristine_dynamic_cache(prefix)
            suffix = torch.full((1, 1, 1, 2), float(suffix_id))
            cloned.layers[0].update(suffix, suffix + 100)
            outputs[suffix_id] = cloned.layers[0].keys.clone()
        return outputs

    forward = run((7, 9))
    reverse = run((9, 7))
    assert torch.equal(forward[7], reverse[7])
    assert torch.equal(forward[9], reverse[9])
    assert prefix.get_seq_length() == 3
    assert prefix.layers[0].keys.data_ptr() == prefix_keys.data_ptr()
    assert prefix.layers[0].values.data_ptr() == prefix_values.data_ptr()


def test_longest_common_prefix_is_prompt_order_invariant() -> None:
    sequences = [(1, 2, 3, 4), (1, 2, 8), (1, 2, 3, 9)]
    assert _longest_common_prefix_length(sequences) == 2
    assert _longest_common_prefix_length(list(reversed(sequences))) == 2


def test_exact_mrope_positions_and_original_mask_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr("mprisk.cache.kv_prefill.site.ENABLE_USER_SITE", False)

    recorded: dict[str, object] = {}
    full_position_ids = torch.arange(15).reshape(3, 1, 5)

    class _InnerModel:
        @staticmethod
        def get_rope_index(input_ids: torch.Tensor, **kwargs: object):
            recorded["rope_input_ids"] = input_ids.clone()
            recorded["rope_kwargs"] = kwargs
            return full_position_ids.clone(), torch.tensor([[-2]])

    class _Model:
        def __init__(self) -> None:
            self.model = _InnerModel()

        def __call__(self, **kwargs: object):
            recorded["forward_kwargs"] = kwargs
            suffix_len = int(kwargs["input_ids"].shape[-1])  # type: ignore[union-attr]
            states = tuple(torch.full((1, suffix_len, 4), float(i)) for i in range(3))
            return SimpleNamespace(hidden_states=states)

    wrapper = SimpleNamespace(
        family="qwen_vl",
        model_key="qwen3_vl_8b",
        model_path="/models/qwen3-vl",
        device="cpu",
        dtype_name="bfloat16",
        attn_implementation="sdpa",
        expected_layer_count=2,
        expected_hidden_dim=4,
        model=_Model(),
        processor=SimpleNamespace(),
    )
    extractor = QwenVlPromptKvPrefillExtractor(wrapper, verbose=False)
    full_inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "mm_token_type_ids": torch.tensor([[0, 2, 2, 0, 0]]),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    exact_positions = extractor._full_position_ids(full_inputs)
    prefix_inputs = extractor._build_prefix_inputs(
        full_inputs,
        3,
        full_position_ids=exact_positions,
    )
    assert torch.equal(prefix_inputs["position_ids"], full_position_ids[..., :3])

    cache = _Cache(
        torch.zeros((1, 1, 3, 2), dtype=torch.float32),
        torch.zeros((1, 1, 3, 2), dtype=torch.float32),
    )
    request = SimpleNamespace(messages=())
    result = extractor._suffix_forward(
        request=request,
        full_inputs=full_inputs,
        full_position_ids=exact_positions,
        prefix_len=3,
        prefix_identity="a" * 64,
        past_key_values=cache,
    )
    forwarded = recorded["forward_kwargs"]
    assert torch.equal(forwarded["attention_mask"], full_inputs["attention_mask"])
    assert torch.equal(forwarded["position_ids"], full_position_ids[..., 3:])
    assert forwarded["use_cache"] is True
    assert result.token_count == 5
    assert result.t0_token_index == 4
    assert result.provenance["prefill_strategy"] == "qwen_vl_prompt_kv"
    assert result.provenance["prefill_strategy_version"] == "v1"
    assert result.provenance["prefix_identity"] == "a" * 64
