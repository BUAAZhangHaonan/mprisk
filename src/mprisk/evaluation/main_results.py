"""Main-result exports for evaluation analyses."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from mprisk.data.manifests import read_jsonl
from mprisk.utils.io import write_json

GROUPS = ("Conflict", "Aligned")
METRICS = ("S_mean", "D", "abs_R")


@dataclass(frozen=True)
class ConflictVsAlignedResult:
    json_path: Path
    csv_path: Path
    count: int


def compare_conflict_vs_aligned(
    *,
    sdr_scores_path: str | Path,
    state_patterns_path: str | Path | None,
    output_dir: str | Path,
) -> ConflictVsAlignedResult:
    rows = _merged_rows(
        read_jsonl(sdr_scores_path),
        read_jsonl(state_patterns_path) if state_patterns_path is not None else None,
    )
    grouped_rows = {
        group: [row for row in rows if row.get("sample_type") == group] for group in GROUPS
    }
    payload = {
        "inputs": {
            "sdr_scores": str(sdr_scores_path),
            "state_patterns": str(state_patterns_path) if state_patterns_path is not None else None,
        },
        "groups": {group: _group_summary(grouped_rows[group]) for group in GROUPS},
        "p_values": _p_values(grouped_rows),
    }

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_root / "conflict_vs_aligned.json", payload)
    csv_path = _write_compact_csv(output_root / "conflict_vs_aligned.csv", payload["groups"])
    return ConflictVsAlignedResult(
        json_path=json_path,
        csv_path=csv_path,
        count=sum(payload["groups"][group]["n"] for group in GROUPS),
    )


def _merged_rows(
    sdr_rows: list[dict[str, Any]],
    pattern_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if pattern_rows is None:
        return [_with_abs_r(row) for row in sdr_rows]

    patterns_by_id = _rows_by_sample_id(pattern_rows, source_name="state_patterns")
    merged: list[dict[str, Any]] = []
    for row in sdr_rows:
        sample_id = row.get("sample_id")
        merged_row = dict(row)
        pattern_row = patterns_by_id.get(sample_id)
        if pattern_row is not None:
            if "sample_type" in pattern_row:
                merged_row["sample_type"] = pattern_row["sample_type"]
            if "pattern" in pattern_row:
                merged_row["pattern"] = pattern_row["pattern"]
        merged.append(_with_abs_r(merged_row))
    return merged


def _rows_by_sample_id(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[Any, dict[str, Any]]:
    by_id: dict[Any, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id in by_id:
            raise ValueError(f"duplicate sample_id in {source_name}: {sample_id}")
        by_id[sample_id] = row
    return by_id


def _with_abs_r(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    if "R" in output:
        output["abs_R"] = abs(float(output["R"]))
    return output


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patterns = Counter(
        str(row.get("pattern", "")) for row in rows if row.get("pattern") is not None
    )
    total = len(rows)
    return {
        "n": total,
        "metrics": {metric: _numeric_summary(_numeric_values(rows, metric)) for metric in METRICS},
        "pattern_counts": dict(patterns),
        "pattern_proportions": {
            pattern: count / total if total else 0.0 for pattern, count in patterns.items()
        },
    }


def _numeric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if metric in row and row[metric] is not None:
            values.append(float(row[metric]))
    return values


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": float(median(values)),
        "std": _sample_std(values),
    }


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _p_values(grouped_rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    conflict = grouped_rows["Conflict"]
    aligned = grouped_rows["Aligned"]
    tests = {
        metric: _mann_whitney_test(
            _numeric_values(conflict, metric),
            _numeric_values(aligned, metric),
        )
        for metric in METRICS
    }
    tests["pattern_distribution"] = _pattern_distribution_test(conflict, aligned)
    return tests


def _mann_whitney_test(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
    if not x_values or not y_values:
        return {"p_value": None, "method": "not_enough_data", "statistic": None}
    try:
        from scipy import stats

        result = stats.mannwhitneyu(x_values, y_values, alternative="two-sided", method="auto")
        return {
            "p_value": float(result.pvalue),
            "method": "scipy.stats.mannwhitneyu_two_sided",
            "statistic": float(result.statistic),
        }
    except ImportError:
        return _mann_whitney_rank_sum_fallback(x_values, y_values)


def _mann_whitney_rank_sum_fallback(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
    n_x = len(x_values)
    n_y = len(y_values)
    ranked = _average_ranks(
        [(value, "x") for value in x_values] + [(value, "y") for value in y_values]
    )
    rank_sum_x = sum(rank for rank, label in ranked if label == "x")
    u_x = rank_sum_x - n_x * (n_x + 1) / 2
    mean_u = n_x * n_y / 2
    variance = _mann_whitney_variance_with_ties(x_values + y_values, n_x, n_y)
    if variance <= 0:
        return {
            "p_value": None,
            "method": "rank_sum_normal_approximation_degenerate",
            "statistic": u_x,
        }
    z_score = (u_x - mean_u) / math.sqrt(variance)
    p_value = math.erfc(abs(z_score) / math.sqrt(2))
    return {
        "p_value": p_value,
        "method": "rank_sum_normal_approximation_no_scipy",
        "statistic": u_x,
    }


def _average_ranks(values: list[tuple[float, str]]) -> list[tuple[float, str]]:
    sorted_values = sorted(values, key=lambda item: item[0])
    ranked: list[tuple[float, str]] = []
    index = 0
    while index < len(sorted_values):
        end = index + 1
        while end < len(sorted_values) and sorted_values[end][0] == sorted_values[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        ranked.extend((average_rank, sorted_values[position][1]) for position in range(index, end))
        index = end
    return ranked


def _mann_whitney_variance_with_ties(values: list[float], n_x: int, n_y: int) -> float:
    total_n = n_x + n_y
    if total_n < 2:
        return 0.0
    tie_counts = Counter(values)
    tie_sum = sum(count**3 - count for count in tie_counts.values())
    return n_x * n_y / 12 * ((total_n + 1) - tie_sum / (total_n * (total_n - 1)))


def _pattern_distribution_test(
    conflict_rows: list[dict[str, Any]],
    aligned_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    conflict_counts = Counter(str(row.get("pattern", "")) for row in conflict_rows)
    aligned_counts = Counter(str(row.get("pattern", "")) for row in aligned_rows)
    patterns = sorted(set(conflict_counts) | set(aligned_counts))
    if not patterns:
        return {"p_value": None, "method": "not_enough_data", "statistic": None, "df": 0}

    table = [
        [conflict_counts.get(pattern, 0) for pattern in patterns],
        [aligned_counts.get(pattern, 0) for pattern in patterns],
    ]
    if len(patterns) < 2:
        return {
            "p_value": None,
            "method": "chi_square_not_applicable_single_pattern",
            "statistic": 0.0,
            "df": 0,
            "patterns": patterns,
        }
    try:
        from scipy import stats

        statistic, p_value, df, _expected = stats.chi2_contingency(table, correction=False)
        return {
            "p_value": float(p_value),
            "method": "scipy.stats.chi2_contingency",
            "statistic": float(statistic),
            "df": int(df),
            "patterns": patterns,
        }
    except ImportError:
        statistic, df = _chi_square_statistic(table)
        return {
            "p_value": None,
            "method": "chi_square_statistic_only_no_scipy",
            "statistic": statistic,
            "df": df,
            "patterns": patterns,
        }


def _chi_square_statistic(table: list[list[int]]) -> tuple[float, int]:
    row_totals = [sum(row) for row in table]
    column_totals = [
        sum(table[row_index][col_index] for row_index in range(len(table)))
        for col_index in range(len(table[0]))
    ]
    total = sum(row_totals)
    statistic = 0.0
    for row_index, row in enumerate(table):
        for col_index, observed in enumerate(row):
            expected = row_totals[row_index] * column_totals[col_index] / total if total else 0.0
            if expected:
                statistic += (observed - expected) ** 2 / expected
    df = (len(table) - 1) * (len(table[0]) - 1)
    return statistic, df


def _write_compact_csv(path: Path, groups: dict[str, Any]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "metric", "mean", "median", "std", "n"],
        )
        writer.writeheader()
        for group in GROUPS:
            summary = groups[group]
            for metric in METRICS:
                metric_summary = summary["metrics"][metric]
                writer.writerow(
                    {
                        "group": group,
                        "metric": metric,
                        "mean": _csv_value(metric_summary["mean"]),
                        "median": _csv_value(metric_summary["median"]),
                        "std": _csv_value(metric_summary["std"]),
                        "n": metric_summary["n"],
                    }
                )
            for pattern, count in summary["pattern_counts"].items():
                writer.writerow(
                    {
                        "group": group,
                        "metric": f"pattern:{pattern}",
                        "mean": summary["pattern_proportions"][pattern],
                        "median": "",
                        "std": "",
                        "n": count,
                    }
                )
    return path


def _csv_value(value: object) -> object:
    return "" if value is None else value
