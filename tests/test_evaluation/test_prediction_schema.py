from __future__ import annotations

import json

import pytest

from mprisk.evaluation.predictions import (
    normalize_label,
    normalize_prediction_row,
    read_prediction_jsonl,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": "sample-1",
        "model_key": "demo-model",
        "protocol": "posthoc",
        "prediction": "positive",
        "target_label": "positive",
        "is_correct": False,
        "confidence_text": "0.82",
        "raw_response": "The answer is positive.",
        "source": "unit-test",
    }
    row.update(overrides)
    return row


def test_normalize_label_handles_case_whitespace_and_simple_synonyms() -> None:
    assert normalize_label("  POS ") == "positive"
    assert normalize_label("negative.") == "negative"
    assert normalize_label(" neither ") == "neutral"
    assert normalize_label("not sure") == "uncertain"
    assert normalize_label("not-a-label") == "invalid"


def test_normalize_prediction_row_validates_required_schema_fields() -> None:
    row = _row()
    del row["raw_response"]

    with pytest.raises(ValueError, match="raw_response"):
        normalize_prediction_row(row)


def test_normalize_prediction_row_validates_field_types() -> None:
    row = _row(is_correct="false")

    with pytest.raises(TypeError, match="is_correct"):
        normalize_prediction_row(row)


def test_normalize_prediction_row_adds_normalized_labels() -> None:
    normalized = normalize_prediction_row(
        _row(prediction="  Yes ", target_label="NEG", confidence_text=None)
    )

    assert normalized["normalized_prediction"] == "positive"
    assert normalized["normalized_target_label"] == "negative"
    assert normalized["confidence_text"] is None


def test_read_prediction_jsonl_returns_normalized_rows(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_row(sample_id="sample-1", prediction="positive")),
                "",
                json.dumps(_row(sample_id="sample-2", prediction="I cannot determine")),
            ]
        ),
        encoding="utf-8",
    )

    rows = read_prediction_jsonl(path)

    assert [row["sample_id"] for row in rows] == ["sample-1", "sample-2"]
    assert [row["normalized_prediction"] for row in rows] == ["positive", "uncertain"]
