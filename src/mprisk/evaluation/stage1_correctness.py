"""Stage 1 structured-judgment correctness."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from mprisk.evaluation.predictions import (
    ABSTAIN_LABELS,
    MAIN_LABELS,
    normalize_prediction_row,
    write_prediction_jsonl,
)


def exact_match(prediction: str, target: str) -> bool:
    return prediction.strip().lower() == target.strip().lower()


def compute_correctness(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize rows and compute Stage 1 correctness flags."""

    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = normalize_prediction_row(row)
        prediction = normalized["normalized_prediction"]
        target = normalized["normalized_target_label"]
        is_evaluated = target in MAIN_LABELS
        is_abstain = prediction in ABSTAIN_LABELS
        is_correct = is_evaluated and not is_abstain and prediction == target

        normalized["is_evaluated"] = is_evaluated
        normalized["is_abstain"] = is_abstain
        normalized["is_correct"] = is_correct
        scored_rows.append(normalized)
    return scored_rows


def summarize_correctness(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Summarize Stage 1 correctness over rows with clear target labels."""

    scored_rows = list(rows)
    if any("is_evaluated" not in row or "is_abstain" not in row for row in scored_rows):
        scored_rows = compute_correctness(scored_rows)
    evaluated_rows = [row for row in scored_rows if row.get("is_evaluated") is True]
    evaluated_count = len(evaluated_rows)
    correct_count = sum(1 for row in evaluated_rows if row.get("is_correct") is True)
    abstain_count = sum(1 for row in evaluated_rows if row.get("is_abstain") is True)
    error_count = evaluated_count - correct_count - abstain_count

    def rate(count: int) -> float:
        if evaluated_count == 0:
            return 0.0
        return count / evaluated_count

    return {
        "total_rows": len(scored_rows),
        "evaluated_count": evaluated_count,
        "correct_count": correct_count,
        "error_count": error_count,
        "abstain_count": abstain_count,
        "accuracy": rate(correct_count),
        "error_rate": rate(error_count),
        "abstain_rate": rate(abstain_count),
    }


def write_stage1_outputs(
    rows: Iterable[Mapping[str, Any]],
    outputs_root: str | Path = "outputs",
) -> tuple[Path, Path]:
    """Write normalized predictions and correctness summary for one model/protocol pair."""

    scored_rows = compute_correctness(rows)
    if not scored_rows:
        raise ValueError("Cannot write Stage 1 outputs for zero rows")

    model_key = str(scored_rows[0]["model_key"])
    protocol = str(scored_rows[0]["protocol"])
    for row in scored_rows:
        if row["model_key"] != model_key or row["protocol"] != protocol:
            raise ValueError("Stage 1 outputs must contain a single model_key/protocol pair")

    output_dir = Path(outputs_root) / "evaluation" / "stage1" / model_key / protocol
    predictions_path = output_dir / "predictions_normalized.jsonl"
    summary_path = output_dir / "correctness_summary.json"

    write_prediction_jsonl(scored_rows, predictions_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summarize_correctness(scored_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return predictions_path, summary_path
