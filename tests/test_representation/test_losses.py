from __future__ import annotations

import pytest
import torch

from mprisk.representation.losses import (
    SUPPORTED_LOSSES,
    ProxyAnchorLoss,
    SphericalSDRHingeLoss,
)


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


def _unit_sphere_z(batch: int = 4, dim: int = 8, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, 3, dim, generator=gen)
    return z / z.norm(dim=-1, keepdim=True)


def test_sdr_hinge_forward_returns_scalar_hinges_and_diag() -> None:
    loss = SphericalSDRHingeLoss(margin_D=0.6, margin_R=0.4)
    z = _unit_sphere_z(batch=4, dim=8).requires_grad_(True)
    labels = torch.tensor([0, 1, 0, 1])
    hinge_D, hinge_R, diag = loss(z, labels)

    assert hinge_D.ndim == 0
    assert hinge_R.ndim == 0
    assert torch.isfinite(hinge_D) and torch.isfinite(hinge_R)
    # Gradients reach condition_z.
    (hinge_D + hinge_R).backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    # All diagnostic keys present.
    for key in (
        "mean_d_C", "mean_d_A", "mean_absR_C", "mean_absR_A",
        "hinge_D", "hinge_R", "r_signed", "d_m1_m2",
    ):
        assert key in diag, f"missing diag key: {key}"


def test_sdr_hinge_grouping_uses_aligned_zero_conflict_one() -> None:
    # Construct a batch where Conflict samples have tiny d(M1, M2) and
    # Aligned samples have large d(M1, M2). The expected mean_d_C < mean_d_A
    # so hinge_D > 0 (margin - (mean_d_C - mean_d_A) > margin > 0).
    loss = SphericalSDRHingeLoss(margin_D=0.6, margin_R=0.4)
    dim = 8
    # Aligned (label=0): orthogonal M1, M2 -> d = pi/2 ~ 1.57 rad
    a_m1 = torch.tensor([1.0] + [0.0] * (dim - 1))
    a_m2 = torch.tensor([0.0, 1.0] + [0.0] * (dim - 2))
    # Conflict (label=1): near-identical M1, M2 -> d ~ 0
    c_m1 = torch.tensor([1.0] + [0.0] * (dim - 1))
    c_m2 = torch.tensor([0.9999] + [0.0] * (dim - 1))
    c_m2 = c_m2 / c_m2.norm()
    # M12 = midpoint; r_signed ~ 0 in both groups so hinge_R stays near margin_R.
    m12 = (a_m1 + a_m2) / 2
    a_m12 = m12 / m12.norm()
    c_m12 = (c_m1 + c_m2) / 2
    c_m12 = c_m12 / c_m12.norm()
    z = torch.stack(
        [
            torch.stack([a_m1, a_m2, a_m12]),  # aligned
            torch.stack([c_m1, c_m2, c_m12]),  # conflict
            torch.stack([a_m1, a_m2, a_m12]),
            torch.stack([c_m1, c_m2, c_m12]),
        ]
    )
    labels = torch.tensor([0, 1, 0, 1])
    hinge_D, hinge_R, diag = loss(z, labels)

    # mean_d_A >> mean_d_C so the gap (mean_d_C - mean_d_A) is very negative.
    assert diag["mean_d_A"].item() > diag["mean_d_C"].item()
    assert (diag["mean_d_C"] - diag["mean_d_A"]).item() < 0.0
    # Hinge_D = relu(0.6 - negative) = 0.6 + |gap| > 0.6
    assert hinge_D.item() > 0.6
    # Labels are exactly the (0,1) mainline convention.
    assert diag["mean_d_C"].item() >= 0.0


def test_sdr_hingle_single_class_batch_is_zero_no_op() -> None:
    loss = SphericalSDRHingeLoss(margin_D=0.6, margin_R=0.4)
    z = _unit_sphere_z(batch=4, dim=8)
    # Only Aligned (or only Conflict) present -> hinge is zero no-op so the
    # aux contribution can never blow up on a class-imbalanced batch.
    labels = torch.tensor([0, 0, 0, 0])
    hinge_D, hinge_R, diag = loss(z, labels)
    assert hinge_D.item() == 0.0
    assert hinge_R.item() == 0.0
    # Diagnostic tensors still exist for caller convenience.
    assert "mean_d_C" in diag and "mean_d_A" in diag


def test_sdr_hinge_rejects_wrong_shape_and_bad_labels() -> None:
    loss = SphericalSDRHingeLoss(margin_D=0.6, margin_R=0.4)
    z = _unit_sphere_z(batch=4, dim=8)
    # Wrong condition_z rank (missing the 3-prompt axis).
    bad_z = torch.randn(4, 8)
    with pytest.raises(ValueError, match="condition_z must have shape"):
        loss(bad_z, torch.tensor([0, 1, 0, 1]))
    # Bad labels (out of {0, 1}).
    with pytest.raises(ValueError, match="labels must be Aligned=0"):
        loss(z, torch.tensor([0, 2, 0, 1]))
    # Negative margins.
    with pytest.raises(ValueError, match="margins must be non-negative"):
        SphericalSDRHingeLoss(margin_D=-0.1)
