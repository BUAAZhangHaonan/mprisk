"""Timing and cost metrics."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from mprisk.data.manifests import read_jsonl
from mprisk.utils.io import ensure_parent, write_json

T0_METHOD = "t0_state"
POSTHOC_METHOD = "posthoc_full_response"
TIMING_FIELDS = (
    "prefill_seconds",
    "decode_seconds",
    "total_seconds",
    "num_generated_tokens",
    "prompt_count",
)


@dataclass(frozen=True)
class EfficiencyComparisonResult:
    json_path: Path
    csv_path: Path
    count: int


def speedup_seconds(posthoc_seconds: float, t0_seconds: float) -> float:
    return posthoc_seconds - t0_seconds


def read_timing_log(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    return [_normalize_timing_row(row) for row in rows]


def compare_t0_posthoc_efficiency(
    *,
    timing_log_path: str | Path,
    output_dir: str | Path,
) -> EfficiencyComparisonResult:
    rows = read_timing_log(timing_log_path)
    methods = _summarize_methods(rows)
    comparison = _compare_methods(methods)
    payload = {
        "inputs": {"timing_log": str(timing_log_path)},
        "methods": methods,
        "comparison": comparison,
    }

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_root / "efficiency_comparison.json", payload)
    csv_path = _write_csv(output_root / "efficiency_comparison.csv", methods, comparison)
    return EfficiencyComparisonResult(json_path=json_path, csv_path=csv_path, count=len(rows))


def _normalize_timing_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["method"] = _normalize_method(row.get("method"))
    for field in TIMING_FIELDS:
        normalized[field] = _numeric_value(row, field)
    return normalized


def _normalize_method(value: Any) -> str:
    method = str(value).strip()
    method_key = re.sub(r"[^a-z0-9]+", "_", method.casefold()).strip("_")
    aliases = {
        "t0": T0_METHOD,
        "t_0": T0_METHOD,
        "t0_state": T0_METHOD,
        "state_t0": T0_METHOD,
        "posthoc": POSTHOC_METHOD,
        "post_hoc": POSTHOC_METHOD,
        "posthoc_full": POSTHOC_METHOD,
        "post_hoc_full": POSTHOC_METHOD,
        "posthoc_full_response": POSTHOC_METHOD,
        "post_hoc_full_response": POSTHOC_METHOD,
        "full_response_posthoc": POSTHOC_METHOD,
    }
    return aliases.get(method_key, method_key)


def _numeric_value(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"Timing row is missing {field}")
    return float(value)


def _summarize_methods(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary = _method_summary(method_rows)
        summaries[method] = summary
    return summaries


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    total_seconds_mean = _mean([float(row["total_seconds"]) for row in rows])
    prompt_count_mean = _mean([float(row["prompt_count"]) for row in rows])
    return {
        "n": len(rows),
        "total_seconds_mean": total_seconds_mean,
        "total_seconds_median": float(median(float(row["total_seconds"]) for row in rows)),
        "prefill_seconds_mean": _mean([float(row["prefill_seconds"]) for row in rows]),
        "decode_seconds_mean": _mean([float(row["decode_seconds"]) for row in rows]),
        "num_generated_tokens_mean": _mean(
            [float(row["num_generated_tokens"]) for row in rows]
        ),
        "prompt_count_mean": prompt_count_mean,
        "prompt_count_adjusted_cost": total_seconds_mean * prompt_count_mean,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _compare_methods(
    methods: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    missing_methods = [method for method in (T0_METHOD, POSTHOC_METHOD) if method not in methods]
    costs = {
        method: methods[method]["prompt_count_adjusted_cost"]
        for method in (T0_METHOD, POSTHOC_METHOD)
        if method in methods
    }
    if missing_methods:
        return {
            "status": "missing_method",
            "t0_method": T0_METHOD,
            "posthoc_method": POSTHOC_METHOD,
            "missing_methods": missing_methods,
            "speedup_ratio": None,
            "saved_seconds": None,
            "prompt_count_adjusted_costs": costs,
        }

    t0_mean = float(methods[T0_METHOD]["total_seconds_mean"])
    posthoc_mean = float(methods[POSTHOC_METHOD]["total_seconds_mean"])
    speedup_ratio = posthoc_mean / t0_mean if t0_mean else None
    status = "ok" if speedup_ratio is not None else "undefined_speedup"
    return {
        "status": status,
        "t0_method": T0_METHOD,
        "posthoc_method": POSTHOC_METHOD,
        "missing_methods": [],
        "speedup_ratio": speedup_ratio,
        "saved_seconds": speedup_seconds(posthoc_mean, t0_mean),
        "prompt_count_adjusted_costs": costs,
    }


def _write_csv(
    path: str | Path,
    methods: dict[str, dict[str, float | int]],
    comparison: dict[str, Any],
) -> Path:
    target = ensure_parent(path)
    fieldnames = [
        "row_type",
        "method",
        "n",
        "total_seconds_mean",
        "total_seconds_median",
        "prefill_seconds_mean",
        "decode_seconds_mean",
        "num_generated_tokens_mean",
        "prompt_count_mean",
        "prompt_count_adjusted_cost",
        "speedup_ratio",
        "saved_seconds",
        "status",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, stats in methods.items():
            writer.writerow(
                {
                    "row_type": "method",
                    "method": method,
                    **stats,
                    "speedup_ratio": "",
                    "saved_seconds": "",
                    "status": "",
                }
            )
        writer.writerow(
            {
                "row_type": "comparison",
                "method": f"{T0_METHOD}_vs_{POSTHOC_METHOD}",
                "n": "",
                "total_seconds_mean": "",
                "total_seconds_median": "",
                "prefill_seconds_mean": "",
                "decode_seconds_mean": "",
                "num_generated_tokens_mean": "",
                "prompt_count_mean": "",
                "prompt_count_adjusted_cost": "",
                "speedup_ratio": _csv_value(comparison["speedup_ratio"]),
                "saved_seconds": _csv_value(comparison["saved_seconds"]),
                "status": comparison["status"],
            }
        )
    return target


def _csv_value(value: object) -> object:
    return "" if value is None else value
