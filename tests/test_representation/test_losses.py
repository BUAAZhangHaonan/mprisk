from __future__ import annotations

import pytest
import torch

from mprisk.representation.losses import SUPPORTED_LOSSES, ProxyAnchorLoss


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
