"""Lightweight torch helpers for trajectory representation batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def encode_labels(labels: Sequence[object]) -> tuple[torch.Tensor, dict[object, int]]:
    """Encode hashable labels into stable integer ids in first-seen order."""
    label_to_id: dict[object, int] = {}
    encoded: list[int] = []
    for label in labels:
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)
        encoded.append(label_to_id[label])
    return torch.tensor(encoded, dtype=torch.long), label_to_id


def collate_trajectory_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate trajectory dicts without depending on dataset classes."""
    if not items:
        raise ValueError("items must be non-empty")
    trajectories = torch.as_tensor(
        [item["trajectory"] for item in items],
        dtype=torch.float32,
    )
    labels = [item["label"] for item in items]
    label_ids, label_to_id = encode_labels(labels)
    return {
        "trajectories": trajectories,
        "labels": labels,
        "label_ids": label_ids,
        "label_to_id": label_to_id,
        "sample_ids": [item["sample_id"] for item in items],
        "view_keys": [item["view_key"] for item in items],
        "prompt_keys": [item["prompt_key"] for item in items],
    }
