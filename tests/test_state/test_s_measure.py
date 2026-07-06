from __future__ import annotations

import pytest

from mprisk.state.s_measure import compute_s, compute_s_for_bundle


def _bundle() -> dict[str, object]:
    return {
        "sample_id": "case-1",
        "sample_type": "Conflict",
        "model_key": "toy-model",
        "protocol": "VT",
        "prompt_set_key": "vt_primary_v1",
        "repr_key": "raw_layernorm_mean",
        "embeddings": {
            "M1": {"p1": [0.0, 0.0], "p2": [0.0, 2.0]},
            "M2": {"p1": [3.0, 0.0], "p2": [3.0, 4.0]},
            "M12": {"p1": [1.0, 1.0], "p2": [1.0, 3.0]},
        },
    }


def test_compute_s_for_bundle_returns_view_dispersion_and_mean() -> None:
    scores = compute_s_for_bundle(_bundle())

    assert scores["S_M1"] == pytest.approx(1.0)
    assert scores["S_M2"] == pytest.approx(2.0)
    assert scores["S_M12"] == pytest.approx(1.0)
    assert scores["S_mean"] == pytest.approx(4.0 / 3.0)


def test_compute_s_accepts_embedding_bundle_for_new_api() -> None:
    scores = compute_s(_bundle())

    assert scores["S_mean"] == pytest.approx(4.0 / 3.0)


def test_compute_s_keeps_legacy_template_variance_mean() -> None:
    assert compute_s([1.0, 2.0, 3.0]) == pytest.approx(2.0)
