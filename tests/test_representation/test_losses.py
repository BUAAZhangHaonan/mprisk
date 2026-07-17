from __future__ import annotations

import math

import pytest
import torch

from mprisk.representation.losses import (
    SUPPORTED_LOSSES,
    ModalitySplitRankingLoss,
    ProxyAnchorLoss,
)
from mprisk.state.spherical import compute_spherical_state


def test_proxy_anchor_is_the_only_metric_objective_for_tme() -> None:
    assert SUPPORTED_LOSSES == ("proxy_anchor", "cross_entropy")
    objective = ProxyAnchorLoss(embed_dim=3, num_classes=2, alpha=8.0, margin=0.1)
    embeddings = torch.tensor(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.1, 0.9, 0.0]],
        requires_grad=True,
    )
    loss = objective(embeddings, torch.tensor([0, 0, 1, 1]))
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert objective.proxies.shape == (2, 3)


def test_proxy_anchor_rejects_non_ac_class_contract() -> None:
    with pytest.raises(ValueError, match="exactly two classes"):
        ProxyAnchorLoss(embed_dim=3, num_classes=3)


def _direction(degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    return torch.tensor([math.cos(radians), math.sin(radians)])


def _condition_bundle(split_degrees: list[float]) -> torch.Tensor:
    samples = []
    for split in split_degrees:
        prompts = []
        for jitter in (-2.0, 2.0):
            prompts.append(
                torch.stack(
                    (
                        _direction(jitter),
                        _direction(split + jitter),
                        _direction(split / 2.0 + jitter),
                    )
                )
            )
        samples.append(torch.stack(prompts))
    return torch.stack(samples)


def test_modality_split_ranking_enforces_d_and_raw_angle_margins() -> None:
    objective = ModalitySplitRankingLoss(
        d_margin=0.25,
        angular_margin_rad=math.radians(5.0),
    )
    labels = torch.tensor([0, 0, 1, 1])
    separated = _condition_bundle([8.0, 10.0, 35.0, 40.0])
    d_loss, angular_loss, diagnostics = objective(separated, labels)
    assert d_loss == pytest.approx(0.0, abs=1e-6)
    assert angular_loss == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["d_pair_satisfaction"] == pytest.approx(1.0)
    assert diagnostics["angular_pair_satisfaction"] == pytest.approx(1.0)
    assert diagnostics["split_angle_rad"] * 180.0 / math.pi == pytest.approx(
        torch.tensor([8.0, 10.0, 35.0, 40.0]), abs=1e-3
    )


def test_modality_split_ranking_has_finite_gradients_without_denominator_gaming() -> None:
    objective = ModalitySplitRankingLoss(
        d_margin=2.0,
        angular_margin_rad=math.radians(5.0),
    )
    labels = torch.tensor([0, 0, 1, 1])
    condition_z = _condition_bundle([8.0, 10.0, 11.0, 12.0]).requires_grad_()
    d_loss, angular_loss, diagnostics = objective(condition_z, labels)
    denominator_gradient = torch.autograd.grad(
        d_loss,
        diagnostics["dispersion_denominator"],
        allow_unused=True,
        retain_graph=True,
    )[0]
    assert denominator_gradient is None
    (d_loss + angular_loss).backward()
    assert condition_z.grad is not None
    assert torch.isfinite(condition_z.grad).all()
    assert float(condition_z.grad.abs().sum()) > 0.0


def test_modality_split_ranking_d_matches_exact_state_definition() -> None:
    objective = ModalitySplitRankingLoss(
        d_margin=0.25,
        angular_margin_rad=math.radians(5.0),
    )
    condition_z = _condition_bundle([10.0, 35.0])
    _d_loss, _angular_loss, diagnostics = objective(
        condition_z,
        torch.tensor([0, 1]),
    )
    for sample_index in range(2):
        bundle = {
            "sample_id": f"sample-{sample_index}",
            "sample_type": "Aligned" if sample_index == 0 else "Conflict",
            "calibration_split": "",
            "embeddings": {
                condition: {
                    f"p{prompt_index}": condition_z[
                        sample_index, prompt_index, condition_index
                    ].tolist()
                    for prompt_index in range(condition_z.shape[1])
                }
                for condition_index, condition in enumerate(("M1", "M2", "M12"))
            },
        }
        expected = compute_spherical_state(bundle)
        assert diagnostics["D"][sample_index] == pytest.approx(expected["D"], rel=1e-4)
