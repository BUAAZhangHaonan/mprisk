"""Extract pre-generation t0 trajectories from hidden-state cache shards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from mprisk.cache.hidden_state_cache import HiddenStateEntry


Trajectory = list[list[float]]


@dataclass(frozen=True)
class TrajectoryBundle:
    sample_id: str
    model_key: str
    protocol: str
    m1_trajectory: Trajectory
    m2_trajectory: Trajectory
    m12_trajectory: Trajectory
    trajectory_meta: dict[str, int]


def t0_token_index(entry: HiddenStateEntry | None = None) -> int:
    """Return the token index used for pre-generation state extraction."""
    if entry is None:
        return -1
    metadata = entry.metadata or {}
    if "t0_token_index" in metadata and metadata["t0_token_index"] not in (None, ""):
        return int(metadata["t0_token_index"])
    return -1


def extract_t0_trajectory(entry: HiddenStateEntry) -> Trajectory:
    """Read one cache entry and return a [layer_count, hidden_dim] trajectory."""
    hidden_states = _load_entry_hidden_states(entry)
    trajectory = _slice_t0(hidden_states, entry)
    _validate_trajectory(trajectory, entry)
    return trajectory.astype(np.float32).tolist()


def bundle_three_views(
    m1_entry: HiddenStateEntry,
    m2_entry: HiddenStateEntry,
    m12_entry: HiddenStateEntry,
) -> TrajectoryBundle:
    """Extract and bundle M1, M2, and M12 trajectories for one sample."""
    entries = (m1_entry, m2_entry, m12_entry)
    sample_ids = {entry.sample_id for entry in entries}
    model_keys = {entry.model_key for entry in entries}
    protocols = {entry.protocol for entry in entries}
    shapes = {(entry.layer_count, entry.hidden_dim) for entry in entries}
    if len(sample_ids) != 1 or len(model_keys) != 1 or len(protocols) != 1:
        raise ValueError("M1, M2, and M12 entries must refer to the same sample/model/protocol")
    if len(shapes) != 1:
        raise ValueError("M1, M2, and M12 entries must have the same layer_count and hidden_dim")

    t0_indices = {t0_token_index(entry) for entry in entries}
    if len(t0_indices) != 1:
        raise ValueError("M1, M2, and M12 entries must use the same t0_token_index")

    return TrajectoryBundle(
        sample_id=m1_entry.sample_id,
        model_key=m1_entry.model_key,
        protocol=m1_entry.protocol,
        m1_trajectory=extract_t0_trajectory(m1_entry),
        m2_trajectory=extract_t0_trajectory(m2_entry),
        m12_trajectory=extract_t0_trajectory(m12_entry),
        trajectory_meta={
            "layer_count": m1_entry.layer_count,
            "hidden_dim": m1_entry.hidden_dim,
            "t0_token_index": t0_token_index(m1_entry),
        },
    )


def _load_entry_hidden_states(entry: HiddenStateEntry) -> np.ndarray:
    if not entry.shard_file.exists():
        raise FileNotFoundError(f"Cache shard does not exist: {entry.shard_file}")
    tensors = load_file(entry.shard_file)
    tensor_key = _select_tensor_key(tensors, entry.metadata or {})
    tensor = np.asarray(tensors[tensor_key])
    return _select_sample_tensor(tensor, entry)


def _select_tensor_key(tensors: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    requested = metadata.get("tensor_key")
    if requested:
        requested_key = str(requested)
        if requested_key not in tensors:
            raise KeyError(f"Tensor key {requested_key!r} not found in cache shard")
        return requested_key
    if "hidden_states" in tensors:
        return "hidden_states"
    if len(tensors) == 1:
        return next(iter(tensors))
    keys = ", ".join(sorted(tensors))
    raise ValueError(f"Cache shard has multiple tensors; set metadata.tensor_key. Keys: {keys}")


def _select_sample_tensor(tensor: np.ndarray, entry: HiddenStateEntry) -> np.ndarray:
    if tensor.ndim == 4:
        if entry.index_in_shard >= tensor.shape[0]:
            raise IndexError(
                f"index_in_shard {entry.index_in_shard} is out of range for {entry.shard_file}"
            )
        return tensor[entry.index_in_shard]
    if tensor.ndim in {2, 3}:
        return tensor
    raise ValueError(
        "Hidden-state tensor must have shape [sample, layer, token, hidden], "
        "[layer, token, hidden], or [layer, hidden]"
    )


def _slice_t0(hidden_states: np.ndarray, entry: HiddenStateEntry) -> np.ndarray:
    if hidden_states.ndim == 2:
        return hidden_states
    if hidden_states.ndim != 3:
        raise ValueError("Selected hidden states must be [layer, token, hidden] or [layer, hidden]")
    token_index = t0_token_index(entry)
    token_count = hidden_states.shape[1]
    if not -token_count <= token_index < token_count:
        raise IndexError(f"t0_token_index {token_index} is out of range for {token_count} tokens")
    return hidden_states[:, token_index, :]


def _validate_trajectory(trajectory: np.ndarray, entry: HiddenStateEntry) -> None:
    if trajectory.ndim != 2:
        raise ValueError("t0 trajectory must have shape [layer_count, hidden_dim]")
    layer_count, hidden_dim = trajectory.shape
    if layer_count != entry.layer_count:
        raise ValueError(
            f"t0 trajectory layer_count mismatch: expected {entry.layer_count}, got {layer_count}"
        )
    if hidden_dim != entry.hidden_dim:
        raise ValueError(
            f"t0 trajectory hidden_dim mismatch: expected {entry.hidden_dim}, got {hidden_dim}"
        )
    if not np.isfinite(trajectory).all():
        raise ValueError("t0 trajectory must contain only finite values")
