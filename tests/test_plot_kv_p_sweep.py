from __future__ import annotations

import json

from mprisk.viz.plot_kv_p_sweep import export_curve, normalize_rows


def test_normalize_nested_rows_preserves_missing_metrics() -> None:
    rows = normalize_rows(
        [
            {
                "P": 2,
                "latency": {"median_seconds": 0.2, "p95_seconds": 0.3, "n": 5},
                "stability": {"pattern_agreement": 0.8},
            },
            {"prompt_count": 1, "latency_median_seconds": 0.1},
        ]
    )
    assert [row["prompt_count"] for row in rows] == [1, 2]
    assert rows[0]["latency_p95_seconds"] is None
    assert rows[0]["pattern_agreement"] is None
    assert rows[1]["pattern_agreement"] == 0.8


def test_export_marks_missing_p_values_without_fabricating_data(tmp_path) -> None:
    source = tmp_path / "sweep.json"
    source.write_text(
        json.dumps({"schema": "runner", "rows": [{"P": 1, "latency": {"median": 0.1}}]}),
        encoding="utf-8",
    )
    result = export_curve(source, tmp_path / "out")
    payload = json.loads(
        (tmp_path / "out" / "kv_p_sweep_curve.json").read_text(encoding="utf-8")
    )
    assert result["missing_prompt_counts"] == [2, 4, 8, 16, 32, 64]
    assert payload["rows"][0]["latency_median_seconds"] == 0.1
    assert payload["rows"][0]["latency_p95_seconds"] is None
    assert (tmp_path / "out" / "kv_p_sweep_latency_stability.pdf").exists()

