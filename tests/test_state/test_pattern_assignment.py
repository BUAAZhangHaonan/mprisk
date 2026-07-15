from __future__ import annotations

import json
import math

import pytest

from mprisk.state.aggregation import aggregate_state_rows, compute_state_row
from mprisk.state.patterns import StatePattern, StateThresholds, load_thresholds_config


def _bundle(sample_id: str = "case-1") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "sample_type": "Conflict",
        "model_key": "toy-model",
        "protocol": "VT",
        "prompt_set_key": "vt_primary_v1",
        "repr_key": "tme_proxy_anchor_v1",
        "embeddings": {
            "M1": {"p1": [1.0, 0.0], "p2": [1.0, 0.0]},
            "M2": {"p1": [0.0, 1.0], "p2": [0.0, 1.0]},
            "M12": {
                "p1": [0.7071067811865476, 0.7071067811865476],
                "p2": [0.7071067811865476, 0.7071067811865476],
            },
        },
    }


def test_compute_state_row_emits_metadata_measures_and_pattern() -> None:
    thresholds = StateThresholds(kappa=2.0, tau=2.0e12, delta=0.2)

    row = compute_state_row(_bundle(), thresholds)

    assert row["sample_id"] == "case-1"
    assert row["sample_type"] == "Conflict"
    assert row["model_key"] == "toy-model"
    assert row["protocol"] == "VT"
    assert row["prompt_set_key"] == "vt_primary_v1"
    assert row["repr_key"] == "tme_proxy_anchor_v1"
    assert row["S_M1"] == pytest.approx(0.0)
    assert row["S_M2"] == pytest.approx(0.0)
    assert row["S_M12"] == pytest.approx(0.0)
    assert row["S_mean"] == pytest.approx(0.0)
    assert row["D"] == pytest.approx((math.pi / 2.0) / 1e-12)
    assert row["R"] == pytest.approx(0.0)
    assert row["pattern"] == StatePattern.CONSENSUS.value


def test_aggregate_state_rows_processes_each_sample() -> None:
    thresholds = StateThresholds(kappa=2.0, tau=3.0, delta=0.1)

    rows = aggregate_state_rows([_bundle("case-1"), _bundle("case-2")], thresholds)

    assert [row["sample_id"] for row in rows] == ["case-1", "case-2"]
    assert all(row["pattern"] in {pattern.value for pattern in StatePattern} for row in rows)


def test_load_thresholds_config_from_dict_or_json_file(tmp_path) -> None:
    thresholds = load_thresholds_config({"kappa": 1.0, "tau": 2.0, "delta": 0.3})
    assert thresholds == StateThresholds(kappa=1.0, tau=2.0, delta=0.3)

    config_path = tmp_path / "thresholds.json"
    config_path.write_text(json.dumps({"kappa": 4.0, "tau": 5.0, "delta": 0.6}))

    assert load_thresholds_config(config_path) == StateThresholds(kappa=4.0, tau=5.0, delta=0.6)
