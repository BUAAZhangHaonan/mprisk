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
        "repr_key": "raw_layernorm_mean",
        "embeddings": {
            "M1": {"p1": [0.0, 0.0], "p2": [0.0, 2.0]},
            "M2": {"p1": [3.0, 0.0], "p2": [3.0, 4.0]},
            "M12": {"p1": [1.0, 1.0], "p2": [1.0, 3.0]},
        },
    }


def test_compute_state_row_emits_metadata_measures_and_pattern() -> None:
    thresholds = StateThresholds(kappa=2.0, tau=3.0, delta=0.2)

    row = compute_state_row(_bundle(), thresholds)

    assert row["sample_id"] == "case-1"
    assert row["sample_type"] == "Conflict"
    assert row["model_key"] == "toy-model"
    assert row["protocol"] == "VT"
    assert row["prompt_set_key"] == "vt_primary_v1"
    assert row["repr_key"] == "raw_layernorm_mean"
    assert row["S_M1"] == pytest.approx(1.0)
    assert row["S_M2"] == pytest.approx(2.0)
    assert row["S_M12"] == pytest.approx(1.0)
    assert row["S_mean"] == pytest.approx(4.0 / 3.0)
    assert row["D"] == pytest.approx(math.sqrt(10.0))
    assert row["R"] == pytest.approx((2.0 - math.sqrt(2.0)) / (2.0 + math.sqrt(2.0)))
    assert row["pattern"] == StatePattern.BALANCED.value


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
