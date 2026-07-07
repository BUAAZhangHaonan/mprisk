from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mprisk.representation.adapters import get_trajectory_encoder
from mprisk.representation.trajectory_model import LayerMeanPool, MLPProjection


def test_layer_mean_pool_normalizes_each_layer_before_pooling() -> None:
    trajectories = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 2.0]],
            [[5.0, 0.0], [0.0, 12.0]],
        ]
    )

    pooled = LayerMeanPool()(trajectories)

    expected = torch.stack(
        [
            torch.tensor([[0.6, 0.8], [0.0, 1.0]]).mean(dim=0),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]).mean(dim=0),
        ]
    )
    torch.testing.assert_close(pooled, expected)


def test_mlp_projection_returns_expected_shape_and_unit_norm() -> None:
    torch.manual_seed(7)
    model = MLPProjection(input_dim=4, embed_dim=3, hidden_dim=6, dropout=0.0)
    trajectories = torch.randn(5, 3, 4)

    embeddings = model(trajectories)

    assert embeddings.shape == (5, 3)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(5), atol=1e-6, rtol=1e-6)


def test_mlp_projection_can_skip_output_normalization() -> None:
    torch.manual_seed(11)
    model = MLPProjection(
        input_dim=4,
        embed_dim=3,
        hidden_dim=6,
        dropout=0.0,
        normalize_output=False,
    )
    trajectories = torch.randn(4, 2, 4)

    embeddings = model(trajectories)

    assert embeddings.shape == (4, 3)
    assert not torch.allclose(F.normalize(embeddings, dim=-1), embeddings)


def test_adapters_recognize_tme_supcon_v1_without_breaking_raw_encoder() -> None:
    raw_encoder = get_trajectory_encoder("raw_layernorm_mean")
    tme_encoder = get_trajectory_encoder("tme_supcon_v1")

    assert raw_encoder.encode([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(
        [2**-0.5, 2**-0.5]
    )
    assert tme_encoder.repr_key == "tme_supcon_v1"
