"""Generalization summary helpers."""

from __future__ import annotations


def group_metric_keys(dataset_key: str, model_key: str, protocol: str) -> str:
    return f"{dataset_key}/{model_key}/{protocol}"
