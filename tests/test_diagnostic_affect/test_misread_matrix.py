from __future__ import annotations

from collections import Counter

from mprisk.diagnostic_affect.matrix import _planned_request_records


def test_matrix_request_plan_has_unique_global_call_ids() -> None:
    records = _planned_request_records(
        run_id="run",
        job_id="target_model",
        model_key="model",
        protocol="VT",
        sample_ids=["a", "b"],
        flash_model="deepseek-v4-flash",
        pro_model="deepseek-v4-pro",
    )

    assert len(records) == 8
    assert len({row["call_id"] for row in records}) == 8
    assert Counter(row["role"] for row in records) == {"flash": 6, "pro": 2}
    assert all(row["api_request_issued"] is False for row in records)
    assert all(row["request_sha256"] is None for row in records)
    assert all(row["conditional"] is (row["role"] == "pro") for row in records)
    assert all(
        row["request_materialization_status"] == "awaiting_diagnostic_affect_description"
        for row in records
    )
