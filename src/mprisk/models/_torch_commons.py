"""Shared torch helpers for backbone wrappers.

These helpers were previously duplicated across internvl/qwen_omni/gemma_4/
qwen3_5/qwen_vl. They cover the parts of the prefill-extraction pipeline that
are identical regardless of the underlying HF model:

* move_inputs_to_device: push tokenized inputs to the model device
* require_attention_mask: enforce a 2-D, batch-1 attention mask
* token_position: count non-pad tokens and find the last conditioning index
* trajectory_from_outputs: stack hidden-state trajectory at one token index
* load_config_json: read config.json with a model_type guard
* validate_contract_dims: enforce num_hidden_layers/hidden_size positivity

Each wrapper still owns its own _load_model_contract because the layout of
the underlying config (which sub-dict holds the language block, which
model_type / architectures string is expected, where the dtype lives) is
model-specific. The two helpers below let those contracts share their
plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from collections.abc import Mapping

import torch


def load_config_json(
    model_path: Path,
    *,
    expected_model_type: str | None = None,
    wrapper_label: str,
) -> dict[str, Any]:
    """Read <model_path>/config.json as a dict, optionally guarding model_type."""
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{wrapper_label} config is missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if expected_model_type is not None and payload.get("model_type") != expected_model_type:
        raise ValueError(
            f"Unexpected model_type in {config_path}: {payload.get('model_type')!r}"
        )
    return payload


def validate_contract_dims(contract: dict[str, Any], *, wrapper_label: str) -> None:
    """Raise ValueError if num_hidden_layers / hidden_size are not positive."""
    if contract["num_hidden_layers"] <= 0 or contract["hidden_size"] <= 0:
        raise ValueError(f"Invalid {wrapper_label} dimensions: {contract}")


def move_inputs_to_device(model_inputs: Any, device: str, *, wrapper_label: str) -> Any:
    """Recursively push a tokenizer/processor output onto device.

    Handles both BatchFeature-like objects (with a .to) and plain mappings.
    """
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    if not isinstance(model_inputs, Mapping):
        raise TypeError(
            f"{wrapper_label} processor output must be a BatchFeature or mapping"
        )
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def require_attention_mask(model_inputs: Any, *, wrapper_label: str) -> Any:
    """Return model_inputs['attention_mask'] enforcing a 2-D batch-1 mask."""
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None or attention_mask.ndim != 2 or int(attention_mask.shape[0]) != 1:
        raise ValueError(
            f"{wrapper_label} extraction requires one two-dimensional attention_mask"
        )
    return attention_mask


def token_position(attention_mask: Any) -> tuple[int, int]:
    """Return (token_count, last_non_pad_index) for a 2-D batch-1 mask."""
    token_count = int(attention_mask.shape[-1])
    non_padding = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
    if non_padding.numel() == 0:
        raise ValueError("attention_mask contains no conditioning tokens")
    return token_count, int(non_padding[-1].item())


def trajectory_from_outputs(
    outputs: Any,
    *,
    t0_token_index: int,
    layer_count: int,
    hidden_dim: int,
    wrapper_label: str,
) -> torch.Tensor:
    """Stack language-block hidden states (skip the embedding layer) at one token."""
    hidden_states = getattr(outputs, "hidden_states", None)
    expected_state_count = layer_count + 1
    if hidden_states is None or len(hidden_states) != expected_state_count:
        actual = None if hidden_states is None else len(hidden_states)
        raise ValueError(
            f"Expected {expected_state_count} hidden-state tensors, got {actual}"
        )
    trajectory = torch.stack(
        [state[0, t0_token_index, :] for state in hidden_states[1:]], dim=0
    )
    if tuple(trajectory.shape) != (layer_count, hidden_dim):
        raise ValueError(
            f"Expected {wrapper_label} trajectory shape {(layer_count, hidden_dim)}, "
            f"got {tuple(trajectory.shape)}"
        )
    if not torch.isfinite(trajectory).all().item():
        raise ValueError(f"{wrapper_label} trajectory contains non-finite values")
    return trajectory
