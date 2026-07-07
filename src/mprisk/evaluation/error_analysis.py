"""State-pattern error analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.data.manifests import read_jsonl
from mprisk.utils.io import ensure_parent, write_json


ABSTAIN_POLICY = (
    "Rows are marked abstain when prediction/outcome/status is abstain or "
    "normalized_prediction is uncertain/invalid. Abstain rows are treated as not "
    "correct unless outcome/status explicitly separates abstention from correctness."
)


@dataclass(frozen=True)
class StateToErrorResult:
    json_path: Path
    csv_path: Path
    count: int


def analyze_state_to_error(
    *,
    state_patterns_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> StateToErrorResult:
    state_rows = read_jsonl(state_patterns_path)
    prediction_rows = read_jsonl(predictions_path)

    state_by_id = _index_by_sample_id(state_rows, source_name="state patterns")
    predictions_by_id = _index_by_sample_id(prediction_rows, source_name="predictions")
    merged_rows = _merge_rows(state_rows, predictions_by_id)

    output_root = Path(output_dir)
    overall = _summarize_by_pattern(merged_rows)
    by_sample_type = _summarize_by_sample_type(merged_rows)
    association = _association_test(merged_rows)

    json_path = write_json(
        output_root / "state_to_error.json",
        {
            "metadata": {
                "state_patterns": str(state_patterns_path),
                "predictions": str(predictions_path),
                "total_state_rows": len(state_rows),
                "total_prediction_rows": len(prediction_rows),
                "merged_count": len(merged_rows),
                "missing_prediction_sample_ids": sorted(set(state_by_id) - set(predictions_by_id)),
                "missing_state_sample_ids": sorted(set(predictions_by_id) - set(state_by_id)),
                "abstain_policy": ABSTAIN_POLICY,
            },
            "overall": overall,
            "by_sample_type": by_sample_type,
            "tests": {"association": association},
        },
    )
    csv_path = _write_csv(output_root / "state_to_error.csv", overall, by_sample_type)
    return StateToErrorResult(json_path=json_path, csv_path=csv_path, count=len(merged_rows))


def _index_by_sample_id(rows: list[dict[str, Any]], *, source_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id in {source_name}: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _merge_rows(
    state_rows: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for state_row in state_rows:
        sample_id = str(state_row.get("sample_id", ""))
        prediction_row = predictions_by_id.get(sample_id)
        if prediction_row is None:
            continue

        is_abstain = _is_abstain(prediction_row)
        merged.append(
            {
                "sample_id": sample_id,
                "sample_type": str(state_row.get("sample_type", "")),
                "pattern": str(state_row.get("pattern", "")),
                "is_correct": _effective_correct(prediction_row, is_abstain=is_abstain),
                "is_abstain": is_abstain,
            }
        )
    return merged


def _summarize_by_pattern(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for pattern in sorted({str(row["pattern"]) for row in rows}):
        pattern_rows = [row for row in rows if row["pattern"] == pattern]
        summary[pattern] = _stats(pattern_rows)
    return summary


def _summarize_by_sample_type(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for sample_type in sorted({str(row["sample_type"]) for row in rows}):
        sample_rows = [row for row in rows if row["sample_type"] == sample_type]
        summary[sample_type] = _summarize_by_pattern(sample_rows)
    return summary


def _stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    n = len(rows)
    correct_count = sum(1 for row in rows if row["is_correct"])
    abstain_count = sum(1 for row in rows if row["is_abstain"])
    error_count = n - correct_count
    return {
        "n": n,
        "correct_count": correct_count,
        "error_count": error_count,
        "error_rate": error_count / n if n else 0.0,
        "abstain_count": abstain_count,
        "abstain_rate": abstain_count / n if n else 0.0,
    }


def _association_test(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patterns = sorted({str(row["pattern"]) for row in rows})
    counts = [
        [
            sum(1 for row in rows if row["pattern"] == pattern and row["is_correct"]),
            sum(1 for row in rows if row["pattern"] == pattern and not row["is_correct"]),
        ]
        for pattern in patterns
    ]
    table = {"patterns": patterns, "columns": ["correct", "error"], "counts": counts}
    if len(patterns) < 2 or not _has_both_outcomes(counts):
        return {
            "status": "insufficient_data",
            "method": None,
            "statistic": None,
            "p_value": None,
            "table": table,
        }

    try:
        if len(patterns) == 2:
            from scipy.stats import fisher_exact

            odds_ratio, p_value = fisher_exact(counts)
            return {
                "status": "ok",
                "method": "fisher_exact",
                "statistic": None,
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_value),
                "table": table,
            }

        from scipy.stats import chi2_contingency

        statistic, p_value, degrees_of_freedom, expected = chi2_contingency(counts, correction=False)
        return {
            "status": "ok",
            "method": "chi_square",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": int(degrees_of_freedom),
            "expected": expected.tolist(),
            "table": table,
        }
    except Exception as exc:  # pragma: no cover - deterministic fallback for dependency edge cases.
        return {
            "status": "insufficient_data",
            "method": None,
            "statistic": None,
            "p_value": None,
            "reason": str(exc),
            "table": table,
        }


def _has_both_outcomes(counts: list[list[int]]) -> bool:
    return sum(row[0] for row in counts) > 0 and sum(row[1] for row in counts) > 0


def _write_csv(
    path: str | Path,
    overall: dict[str, dict[str, float | int]],
    by_sample_type: dict[str, dict[str, dict[str, float | int]]],
) -> Path:
    target = ensure_parent(path)
    fieldnames = [
        "scope",
        "sample_type",
        "pattern",
        "n",
        "correct_count",
        "error_count",
        "error_rate",
        "abstain_count",
        "abstain_rate",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pattern, stats in overall.items():
            writer.writerow({"scope": "overall", "sample_type": "", "pattern": pattern, **stats})
        for sample_type, pattern_stats in by_sample_type.items():
            for pattern, stats in pattern_stats.items():
                writer.writerow(
                    {
                        "scope": "sample_type",
                        "sample_type": sample_type,
                        "pattern": pattern,
                        **stats,
                    }
                )
    return target


def _effective_correct(row: dict[str, Any], *, is_abstain: bool) -> bool:
    is_correct = _as_bool(row.get("is_correct"))
    if is_abstain and not _has_separate_outcome_field(row):
        return False
    return is_correct


def _has_separate_outcome_field(row: dict[str, Any]) -> bool:
    return any(_normalized_value(row.get(field)) for field in ("outcome", "status"))


def _is_abstain(row: dict[str, Any]) -> bool:
    if _normalized_value(row.get("normalized_prediction")) in {"abstain", "uncertain", "invalid"}:
        return True
    return any(_normalized_value(row.get(field)) == "abstain" for field in ("prediction", "outcome", "status"))


def _normalized_value(value: Any) -> str:
    return str(value).strip().casefold() if value is not None else ""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return False
