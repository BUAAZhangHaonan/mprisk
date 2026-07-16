"""Representation-level comparison exports for raw and trained embeddings."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev
from typing import Any

from mprisk.data.manifests import read_jsonl
from mprisk.utils.io import ensure_parent, write_json

REQUIRED_INPUTS = ("sdr_scores", "state_patterns", "state_to_error", "conflict_vs_aligned")
CSV_FIELDS = [
    "repr_key",
    "status",
    "missing_reason",
    "loaded_rows",
    "conflict_aligned_d_effect_size",
    "d_mean_difference",
    "d_effect_method",
    "state_error_p_value",
    "state_error_statistic",
    "state_error_method",
    "error_rate_spread",
    "pattern_entropy_bits",
    "pattern_count",
    "dominant_pattern_share",
    "missing_rate",
    "runtime_seconds",
]


@dataclass(frozen=True)
class ReprComparisonResult:
    json_path: Path
    csv_path: Path
    count: int


def compare_representations(
    *,
    repr_results: Mapping[str, Mapping[str, str | Path | None]],
    output_dir: str | Path,
) -> ReprComparisonResult:
    """Compare per-representation evaluation outputs and write JSON/CSV summaries."""
    rows = [
        _comparison_row(repr_key, pathspec)
        for repr_key, pathspec in repr_results.items()
    ]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "repr_count": len(rows),
            "required_inputs": list(REQUIRED_INPUTS),
        },
        "rows": rows,
    }
    json_path = write_json(output_root / "repr_comparison.json", payload)
    csv_path = _write_csv(output_root / "repr_comparison.csv", rows)
    return ReprComparisonResult(json_path=json_path, csv_path=csv_path, count=len(rows))


def _comparison_row(
    repr_key: str,
    pathspec: Mapping[str, str | Path | None],
) -> dict[str, Any]:
    paths = {name: Path(value) for name, value in pathspec.items() if value is not None}
    missing = _missing_inputs(paths)
    base = _empty_row(repr_key, paths)
    if missing:
        base["status"] = "missing"
        base["missing_reason"] = "; ".join(missing)
        return base

    sdr_rows = read_jsonl(paths["sdr_scores"])
    state_rows = read_jsonl(paths["state_patterns"])
    state_to_error = _read_json(paths["state_to_error"])
    conflict_vs_aligned = _read_json(paths["conflict_vs_aligned"])

    d_effect = _d_effect(conflict_vs_aligned, sdr_rows)
    association = _association(state_to_error)
    pattern_summary = _pattern_summary(state_rows, conflict_vs_aligned)

    base.update(
        {
            "status": "ok",
            "loaded_rows": len(sdr_rows),
            "conflict_aligned_d_effect_size": d_effect["effect_size"],
            "d_mean_difference": d_effect["mean_difference"],
            "d_effect_method": d_effect["method"],
            "state_error_p_value": association["p_value"],
            "state_error_statistic": association["statistic"],
            "state_error_method": association["method"],
            "error_rate_spread": _error_rate_spread(state_to_error),
            "pattern_entropy_bits": pattern_summary["entropy_bits"],
            "pattern_count": pattern_summary["pattern_count"],
            "dominant_pattern_share": pattern_summary["dominant_pattern_share"],
            "missing_rate": _missing_rate(
                [conflict_vs_aligned, state_to_error],
                loaded_rows=len(sdr_rows),
            ),
            "runtime_seconds": _runtime_seconds([conflict_vs_aligned, state_to_error]),
        }
    )
    return base


def _empty_row(repr_key: str, paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "repr_key": repr_key,
        "status": "ok",
        "missing_reason": "",
        "loaded_rows": None,
        "conflict_aligned_d_effect_size": None,
        "d_mean_difference": None,
        "d_effect_method": None,
        "state_error_p_value": None,
        "state_error_statistic": None,
        "state_error_method": None,
        "error_rate_spread": None,
        "pattern_entropy_bits": None,
        "pattern_count": None,
        "dominant_pattern_share": None,
        "missing_rate": None,
        "runtime_seconds": None,
        "inputs": {name: str(path) for name, path in paths.items()},
    }


def _missing_inputs(paths: Mapping[str, Path]) -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_INPUTS:
        path = paths.get(name)
        if path is None:
            missing.append(f"{name}: not provided")
        elif not path.exists():
            missing.append(f"{name}: file not found: {path}")
    return missing


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _d_effect(
    conflict_vs_aligned: Mapping[str, Any],
    sdr_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from_summary = _d_effect_from_group_summary(conflict_vs_aligned)
    if from_summary["mean_difference"] is not None:
        return from_summary
    return _d_effect_from_rows(sdr_rows)


def _d_effect_from_group_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    conflict = _metric_summary(payload, "Conflict", "D")
    aligned = _metric_summary(payload, "Aligned", "D")
    if conflict["mean"] is None or aligned["mean"] is None:
        return {"effect_size": None, "mean_difference": None, "method": None}

    mean_difference = conflict["mean"] - aligned["mean"]
    pooled = _pooled_std(
        conflict["std"],
        aligned["std"],
        conflict["n"],
        aligned["n"],
    )
    if pooled is not None and pooled > 0:
        return {
            "effect_size": mean_difference / pooled,
            "mean_difference": mean_difference,
            "method": "pooled_std",
        }
    return {
        "effect_size": mean_difference,
        "mean_difference": mean_difference,
        "method": "raw_difference",
    }


def _metric_summary(payload: Mapping[str, Any], group: str, metric: str) -> dict[str, Any]:
    summary = (
        payload.get("groups", {})
        .get(group, {})
        .get("metrics", {})
        .get(metric, {})
    )
    return {
        "mean": _number_or_none(summary.get("mean")),
        "std": _number_or_none(summary.get("std")),
        "n": int(summary.get("n") or 0),
    }


def _d_effect_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group = {
        "Conflict": _numeric_values(rows, group="Conflict", metric="D"),
        "Aligned": _numeric_values(rows, group="Aligned", metric="D"),
    }
    if not by_group["Conflict"] or not by_group["Aligned"]:
        return {"effect_size": None, "mean_difference": None, "method": None}
    conflict_mean = sum(by_group["Conflict"]) / len(by_group["Conflict"])
    aligned_mean = sum(by_group["Aligned"]) / len(by_group["Aligned"])
    mean_difference = conflict_mean - aligned_mean
    conflict_std = stdev(by_group["Conflict"]) if len(by_group["Conflict"]) > 1 else 0.0
    aligned_std = stdev(by_group["Aligned"]) if len(by_group["Aligned"]) > 1 else 0.0
    pooled = _pooled_std(
        conflict_std,
        aligned_std,
        len(by_group["Conflict"]),
        len(by_group["Aligned"]),
    )
    if pooled is not None and pooled > 0:
        return {
            "effect_size": mean_difference / pooled,
            "mean_difference": mean_difference,
            "method": "pooled_std",
        }
    return {
        "effect_size": mean_difference,
        "mean_difference": mean_difference,
        "method": "raw_difference",
    }


def _numeric_values(rows: list[dict[str, Any]], *, group: str, metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("sample_type") == group and row.get(metric) is not None:
            values.append(float(row[metric]))
    return values


def _pooled_std(
    conflict_std: float | None,
    aligned_std: float | None,
    conflict_n: int,
    aligned_n: int,
) -> float | None:
    if conflict_std is None or aligned_std is None:
        return None
    denominator = conflict_n + aligned_n - 2
    if denominator <= 0:
        return None
    numerator = (conflict_n - 1) * conflict_std**2 + (aligned_n - 1) * aligned_std**2
    if numerator < 0:
        return None
    return math.sqrt(numerator / denominator)


def _association(payload: Mapping[str, Any]) -> dict[str, Any]:
    association = payload.get("tests", {}).get("association", {})
    statistic = association.get("statistic")
    if statistic is None:
        statistic = association.get("odds_ratio")
    return {
        "p_value": _number_or_none(association.get("p_value")),
        "statistic": _number_or_none(statistic),
        "method": association.get("method"),
    }


def _error_rate_spread(payload: Mapping[str, Any]) -> float | None:
    rates = [
        float(stats["error_rate"])
        for stats in payload.get("overall", {}).values()
        if isinstance(stats, dict) and stats.get("error_rate") is not None
    ]
    if not rates:
        return None
    return max(rates) - min(rates)


def _pattern_summary(
    state_rows: list[dict[str, Any]],
    conflict_vs_aligned: Mapping[str, Any],
) -> dict[str, Any]:
    counts = Counter(
        str(row["pattern"])
        for row in state_rows
        if row.get("pattern") is not None
    )
    if not counts:
        counts = _pattern_counts_from_conflict_vs_aligned(conflict_vs_aligned)
    total = sum(counts.values())
    if total == 0:
        return {"entropy_bits": None, "pattern_count": 0, "dominant_pattern_share": None}
    proportions = [count / total for count in counts.values()]
    entropy = -sum(proportion * math.log2(proportion) for proportion in proportions)
    return {
        "entropy_bits": entropy,
        "pattern_count": len(counts),
        "dominant_pattern_share": max(proportions),
    }


def _pattern_counts_from_conflict_vs_aligned(payload: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for group_summary in payload.get("groups", {}).values():
        for pattern, count in group_summary.get("pattern_counts", {}).items():
            counts[str(pattern)] += int(count)
    return counts


def _missing_rate(payloads: list[Mapping[str, Any]], *, loaded_rows: int) -> float:
    del loaded_rows
    for payload in payloads:
        value = _find_number(payload, ("missing_rate",))
        if value is not None:
            return value
    return 0.0


def _runtime_seconds(payloads: list[Mapping[str, Any]]) -> float | None:
    for payload in payloads:
        value = _find_number(
            payload,
            ("runtime_seconds", "runtime", "elapsed_seconds", "duration_seconds"),
        )
        if value is not None:
            return value
    return None


def _find_number(payload: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _number_or_none(payload.get(key))
        if value is not None:
            return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = _number_or_none(metadata.get(key))
            if value is not None:
                return value
    return None


def _number_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})
    return target


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value
