from __future__ import annotations

import json

from mprisk.evaluation.predictions import read_prediction_jsonl
from mprisk.evaluation.stage1_correctness import (
    compute_correctness,
    summarize_correctness,
    write_stage1_outputs,
)


def _row(
    sample_id: str,
    prediction: str,
    target_label: str,
    *,
    model_key: str = "demo-model",
    protocol: str = "posthoc",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "model_key": model_key,
        "protocol": protocol,
        "prediction": prediction,
        "target_label": target_label,
        "confidence_text": "0.5",
        "raw_response": prediction,
        "source": "unit-test",
    }


def test_compute_correctness_marks_correct_errors_and_abstains() -> None:
    rows = [
        _row("correct", "positive", "positive"),
        _row("error", "negative", "positive"),
        _row("abstain", "unclear", "neutral"),
    ]

    scored = compute_correctness(rows)

    assert [row["is_correct"] for row in scored] == [True, False, False]
    assert [row["is_abstain"] for row in scored] == [False, False, True]
    assert [row["is_evaluated"] for row in scored] == [True, True, True]


def test_unclear_target_labels_are_excluded_from_main_denominators() -> None:
    scored = compute_correctness(
        [
            _row("included", "positive", "positive"),
            _row("excluded", "positive", "unclear"),
        ]
    )

    summary = summarize_correctness(scored)

    assert scored[1]["is_evaluated"] is False
    assert summary["total_rows"] == 2
    assert summary["evaluated_count"] == 1
    assert summary["correct_count"] == 1
    assert summary["error_count"] == 0
    assert summary["abstain_count"] == 0
    assert summary["accuracy"] == 1.0


def test_summary_counts_errors_abstains_and_rates() -> None:
    scored = compute_correctness(
        [
            _row("correct", "positive", "positive"),
            _row("error", "negative", "positive"),
            _row("abstain", "not sure", "neutral"),
            _row("excluded", "positive", "ambiguous"),
        ]
    )

    summary = summarize_correctness(scored)

    assert summary == {
        "total_rows": 4,
        "evaluated_count": 3,
        "correct_count": 1,
        "error_count": 1,
        "abstain_count": 1,
        "accuracy": 1 / 3,
        "error_rate": 1 / 3,
        "abstain_rate": 1 / 3,
    }


def test_write_stage1_outputs_uses_model_and_protocol_directories(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _row("correct", "positive", "positive", model_key="m1", protocol="t0"),
                _row("error", "negative", "positive", model_key="m1", protocol="t0"),
                _row("abstain", "invalid", "neutral", model_key="m1", protocol="t0"),
                _row("excluded", "positive", "unclear", model_key="m1", protocol="t0"),
            ]
        ),
        encoding="utf-8",
    )
    scored = compute_correctness(read_prediction_jsonl(input_path))

    predictions_path, summary_path = write_stage1_outputs(scored, tmp_path / "outputs")

    assert predictions_path == (
        tmp_path
        / "outputs"
        / "evaluation"
        / "stage1"
        / "m1"
        / "t0"
        / "predictions_normalized.jsonl"
    )
    assert summary_path == (
        tmp_path / "outputs" / "evaluation" / "stage1" / "m1" / "t0" / "correctness_summary.json"
    )
    written_rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert written_rows[0]["normalized_prediction"] == "positive"
    assert written_rows[1]["is_correct"] is False
    assert written_rows[2]["is_abstain"] is True
    assert written_rows[3]["is_evaluated"] is False
    assert written_summary["total_rows"] == 4
    assert written_summary["evaluated_count"] == 3
    assert written_summary["correct_count"] == 1
    assert written_summary["error_count"] == 1
    assert written_summary["abstain_count"] == 1


def test_write_stage1_outputs_computes_correctness_from_raw_rows(tmp_path) -> None:
    predictions_path, summary_path = write_stage1_outputs(
        [
            _row("correct", "positive", "positive", model_key="m2", protocol="posthoc"),
            _row("abstain", "not sure", "negative", model_key="m2", protocol="posthoc"),
        ],
        tmp_path / "outputs",
    )

    written_rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [row["is_correct"] for row in written_rows] == [True, False]
    assert [row["is_abstain"] for row in written_rows] == [False, True]
    assert written_summary["evaluated_count"] == 2
    assert written_summary["correct_count"] == 1
    assert written_summary["abstain_count"] == 1
