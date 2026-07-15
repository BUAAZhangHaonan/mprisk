from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mprisk.representation.training import TrainingConfig, _baseline_class_weights


def _config(repr_key: str, objective: str) -> TrainingConfig:
    return TrainingConfig(
        repr_key=repr_key,
        model_key="model",
        protocol="vt",
        classification_objective=objective,
        prompt_set_key="p8",
        prompt_set_artifact_sha256="a" * 64,
        expected_prompt_count=1,
        expected_prompt_ids=("p1",),
    )


def test_baseline_weights_are_inverse_train_sample_frequency() -> None:
    samples = [
        SimpleNamespace(sample_id="a1", label_id=0),
        SimpleNamespace(sample_id="a2", label_id=0),
        SimpleNamespace(sample_id="a3", label_id=0),
        SimpleNamespace(sample_id="c1", label_id=1),
    ]
    weights = _baseline_class_weights(
        samples,
        config=_config("single_point_binary_v1", "inverse_frequency_cross_entropy"),
        device=torch.device("cpu"),
    )
    assert weights is not None
    assert weights.tolist() == pytest.approx([4 / 6, 2.0])


def test_tme_remains_proxy_anchor_only_without_cross_entropy_weights() -> None:
    weights = _baseline_class_weights(
        [SimpleNamespace(sample_id="a", label_id=0), SimpleNamespace(sample_id="c", label_id=1)],
        config=_config("tme_proxy_anchor_v1", "proxy_anchor_only"),
        device=torch.device("cpu"),
    )
    assert weights is None
