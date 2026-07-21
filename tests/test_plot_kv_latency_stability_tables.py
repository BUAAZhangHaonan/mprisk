from __future__ import annotations

import csv

from mprisk.viz.plot_kv_latency_stability_tables import (
    _knee_point,
    _pareto_rows,
    aggregate_timing,
    load_stability,
)


def test_aggregate_timing_sums_three_conditions(tmp_path) -> None:
    path = tmp_path / "timing.csv"
    fields = [
        "p",
        "condition",
        "full_prefill_mean_ms",
        "prompt_kv_prefill_mean_ms",
        "generation_call_mean_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in ("M1", "M2", "M12"):
            writer.writerow(
                {
                    "p": 8,
                    "condition": condition,
                    "full_prefill_mean_ms": 100,
                    "prompt_kv_prefill_mean_ms": 50,
                    "generation_call_mean_ms": 200,
                }
            )
    row = aggregate_timing(path)[0]
    assert row["full_prefill_seconds"] == 0.3
    assert row["prompt_kv_prefill_seconds"] == 0.15
    assert row["generation_seconds"] == 0.6
    assert row["prompt_kv_end_to_end_seconds"] == 0.75


def test_pareto_and_knee_are_computed_without_p1() -> None:
    timing = [
        {
            "prompt_count": p,
            "prompt_kv_end_to_end_seconds": float(p),
        }
        for p in (1, 2, 4, 8, 16)
    ]
    stability = [
        {"prompt_count": p, "state_index_mae": value}
        for p, value in ((1, None), (2, 1.0), (4, 0.25), (8, 0.08), (16, 0.0))
    ]
    rows = _pareto_rows(timing, stability)
    assert [row["prompt_count"] for row in rows] == [2, 4, 8, 16]
    assert _knee_point(rows)["prompt_count"] in {4, 8}
