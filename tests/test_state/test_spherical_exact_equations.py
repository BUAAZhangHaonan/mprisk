from __future__ import annotations

import math

import pytest

from mprisk.state.spherical import compute_spherical_state, spherical_distance
from mprisk.utils.io import write_jsonl
from scripts.assign_state_patterns import assign_state_patterns


def _unit(angle: float) -> list[float]:
    return [math.cos(angle), math.sin(angle)]


def _symmetric(center: float, spread: float) -> dict[str, list[float]]:
    return {"p1": _unit(center - spread), "p2": _unit(center + spread)}


def test_spherical_distance_is_geodesic_angle_for_known_vectors() -> None:
    assert spherical_distance(_unit(0.0), _unit(0.0)) == pytest.approx(0.0)
    assert spherical_distance(_unit(0.0), _unit(math.pi / 2.0)) == pytest.approx(
        math.pi / 2.0
    )
    assert spherical_distance(_unit(0.0), _unit(math.pi / 3.0)) == pytest.approx(
        math.pi / 3.0
    )


def test_sdr_uses_squared_dispersion_normalized_d_and_m1_m2_r_denominator() -> None:
    spread_m1 = math.pi / 6.0
    spread_m2 = math.pi / 12.0
    spread_m12 = math.pi / 18.0
    joint_center = math.pi / 6.0
    bundle = {
        "sample_id": "analytic",
        "sample_type": "Conflict",
        "embeddings": {
            "M1": _symmetric(0.0, spread_m1),
            "M2": _symmetric(math.pi / 2.0, spread_m2),
            "M12": _symmetric(joint_center, spread_m12),
        },
    }

    state = compute_spherical_state(bundle)

    assert state["S_M1"] == pytest.approx(spread_m1**2)
    assert state["S_M2"] == pytest.approx(spread_m2**2)
    assert state["S_M12"] == pytest.approx(spread_m12**2)
    assert state["S_mean"] == pytest.approx(
        (spread_m1**2 + spread_m2**2 + spread_m12**2) / 3.0
    )
    assert state["D"] == pytest.approx(
        (math.pi / 2.0) / math.sqrt(spread_m1**2 + spread_m2**2),
        rel=1e-10,
    )
    assert state["R"] == pytest.approx(1.0 / 3.0, rel=1e-10)
    assert state["lean"] == "V"


def test_synchronous_bootstrap_recomputes_exact_r_from_shared_prompt_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mprisk.state.spherical as spherical

    bundle = {
        "sample_id": "bootstrap-analytic",
        "sample_type": "Aligned",
        "embeddings": {
            "M1": {"p1": _unit(0.0), "p2": _unit(math.pi / 6.0)},
            "M2": {"p1": _unit(math.pi / 2.0), "p2": _unit(2.0 * math.pi / 3.0)},
            "M12": {"p1": _unit(math.pi / 8.0), "p2": _unit(math.pi / 2.0)},
        },
    }

    class FixedRng:
        def __init__(self) -> None:
            self._draws = iter(([0, 0], [1, 1], [0, 1]))

        def integers(self, low: int, high: int, size: int) -> list[int]:
            assert (low, high, size) == (0, 2, 2)
            return list(next(self._draws))

    monkeypatch.setattr(spherical, "BOOTSTRAP_REPLICATES", 3)
    monkeypatch.setattr(spherical.np.random, "default_rng", lambda seed: FixedRng())

    state = spherical.compute_spherical_state(bundle)

    expected_r = [0.5, -1.0 / 3.0, 1.0 / 12.0]
    expected_se = spherical.np.asarray(expected_r, dtype=float).std(ddof=1)
    assert state["R_bootstrap_se"] == pytest.approx(expected_se)
    assert state["delta_i"] == pytest.approx(1.96 * expected_se)


def test_pattern_assignment_rejects_stale_sdr_equations(tmp_path) -> None:
    scores = write_jsonl(
        tmp_path / "scores.jsonl",
        [
            {
                "sample_id": "stale",
                "sample_type": "Aligned",
                "sdr_schema": "mprisk_spherical_sdr_v1",
                "distance_metric": "cosine_distance_v1",
                "S_mean": 0.1,
                "D": 0.2,
                "R": 0.0,
                "delta_i": 0.1,
            }
        ],
    )

    with pytest.raises(ValueError, match="exact spherical SDR v2"):
        assign_state_patterns(
            sdr_scores_path=scores,
            thresholds={"kappa": 0.5, "tau": 0.3},
            output_dir=tmp_path / "patterns",
        )
