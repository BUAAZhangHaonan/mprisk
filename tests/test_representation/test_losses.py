from __future__ import annotations

import torch

from mprisk.representation.losses import (
    combined_trajectory_loss,
    prompt_consistency_loss,
    supervised_contrastive_loss,
)
from mprisk.representation.trajectory_model import MLPProjection


def test_prompt_consistency_loss_uses_same_sample_and_view_prompt_pairs() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        requires_grad=True,
    )

    loss = prompt_consistency_loss(
        embeddings,
        sample_ids=["s1", "s1", "s1", "s2"],
        view_keys=["m1", "m1", "m2", "m1"],
        prompt_keys=["p1", "p2", "p1", "p1"],
        temperature=0.2,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_supervised_contrastive_loss_backpropagates_with_negative_budget() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
        ],
        requires_grad=True,
    )

    loss = supervised_contrastive_loss(
        embeddings,
        labels=["safe", "safe", "risk", "risk", "other", "other"],
        temperature=0.2,
        negative_budget_ratio=0.5,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_combined_loss_supports_two_step_synthetic_training() -> None:
    torch.manual_seed(13)
    model = MLPProjection(input_dim=5, embed_dim=4, hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trajectories = torch.randn(6, 3, 5)

    for _ in range(2):
        optimizer.zero_grad()
        embeddings = model(trajectories)
        loss = combined_trajectory_loss(
            embeddings,
            labels=["a", "a", "b", "b", "c", "c"],
            sample_ids=["s1", "s1", "s2", "s2", "s3", "s3"],
            view_keys=["m1", "m1", "m1", "m1", "m2", "m2"],
            prompt_keys=["p1", "p2", "p1", "p2", "p1", "p2"],
            temperature=0.2,
            negative_budget_ratio=0.5,
        )
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)
