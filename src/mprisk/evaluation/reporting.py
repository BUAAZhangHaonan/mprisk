"""Markdown reports that summarize evaluation outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.utils.io import ensure_parent

CONFLICT_VS_ALIGNED_FILE = "conflict_vs_aligned.json"
STATE_TO_ERROR_FILE = "state_to_error.json"
REPR_COMPARISON_FILE = "repr_comparison.json"
EFFICIENCY_COMPARISON_FILE = "efficiency_comparison.json"


@dataclass(frozen=True)
class MainResultSummaryResult:
    summary_path: Path
    missing_inputs: list[str]


def build_main_result_summary(
    *,
    conflict_vs_aligned_path: str | Path | None = None,
    state_to_error_path: str | Path | None = None,
    repr_comparison_path: str | Path | None = None,
    efficiency_comparison_path: str | Path | None = None,
    output_dir: str | Path,
) -> MainResultSummaryResult:
    """Build a compact Markdown summary from main-result JSON outputs."""

    conflict_data, conflict_missing = _read_optional_json(
        conflict_vs_aligned_path,
        default_filename=CONFLICT_VS_ALIGNED_FILE,
    )
    state_data, state_missing = _read_optional_json(
        state_to_error_path,
        default_filename=STATE_TO_ERROR_FILE,
    )
    repr_data, repr_missing = _read_optional_json(
        repr_comparison_path,
        default_filename=REPR_COMPARISON_FILE,
    )
    efficiency_data, efficiency_missing = _read_optional_json(
        efficiency_comparison_path,
        default_filename=EFFICIENCY_COMPARISON_FILE,
    )
    missing_inputs = [
        missing
        for missing in (conflict_missing, state_missing, repr_missing, efficiency_missing)
        if missing is not None
    ]

    lines = [
        "# Main Result Summary",
        "",
        "## Data Scale",
        *_data_scale_lines(conflict_data, state_data, repr_data, efficiency_data),
        "",
        "## Conflict vs Aligned Main Differences",
        *_conflict_vs_aligned_lines(conflict_data, conflict_missing),
        "",
        "## Four-State/Error-Rate Relation",
        *_state_to_error_lines(state_data, state_missing),
        "",
        "## Raw vs TME Comparison",
        *_repr_comparison_lines(repr_data, repr_missing),
        "",
        "## T0 vs Posthoc Efficiency",
        *_efficiency_comparison_lines(efficiency_data, efficiency_missing),
        "",
        "## Missing Result Reminders",
        *_missing_lines(missing_inputs),
        "",
    ]
    summary_path = ensure_parent(Path(output_dir) / "MAIN_RESULT_SUMMARY.md")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return MainResultSummaryResult(summary_path=summary_path, missing_inputs=missing_inputs)


def _read_optional_json(
    path: str | Path | None,
    *,
    default_filename: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, default_filename
    json_path = Path(path)
    if not json_path.exists():
        return None, str(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{json_path} must contain a JSON object")
    return payload, None


def _data_scale_lines(
    conflict_data: dict[str, Any] | None,
    state_data: dict[str, Any] | None,
    repr_data: dict[str, Any] | None,
    efficiency_data: dict[str, Any] | None,
) -> list[str]:
    lines: list[str] = []
    if conflict_data is not None:
        lines.append(f"- Conflict vs Aligned: {_conflict_total(conflict_data)} samples")
    else:
        lines.append("- Conflict vs Aligned: missing")

    if state_data is not None:
        lines.append(f"- Four-state/error relation: {_state_total(state_data)} samples")
    else:
        lines.append("- Four-state/error relation: missing")

    if repr_data is not None:
        comparison_count = _comparison_total(_named_entries(repr_data))
        lines.append(f"- Raw vs TME comparison: {comparison_count} rows")
    else:
        lines.append("- Raw vs TME comparison: missing")

    if efficiency_data is not None:
        lines.append(
            f"- T0 vs posthoc efficiency: {_comparison_total(_named_entries(efficiency_data))} rows"
        )
    else:
        lines.append("- T0 vs posthoc efficiency: missing")
    return lines


def _conflict_vs_aligned_lines(
    data: dict[str, Any] | None,
    missing: str | None,
) -> list[str]:
    if data is None:
        return [f"- Conflict vs Aligned input missing: {missing}"]

    groups = _dict(data.get("groups"))
    conflict = _dict(groups.get("Conflict"))
    aligned = _dict(groups.get("Aligned"))
    lines = [
        "- Conflict "
        f"n={_format_count(conflict.get('n'))}; "
        f"Aligned n={_format_count(aligned.get('n'))}"
    ]

    conflict_metrics = _dict(conflict.get("metrics"))
    aligned_metrics = _dict(aligned.get("metrics"))
    metric_names = _ordered_keys(("S_mean", "D", "abs_R"), conflict_metrics, aligned_metrics)
    p_values = _dict(data.get("p_values"))
    for metric in metric_names:
        conflict_mean = _dict(conflict_metrics.get(metric)).get("mean")
        aligned_mean = _dict(aligned_metrics.get(metric)).get("mean")
        p_value = _dict(p_values.get(metric)).get("p_value")
        p_text = f", p={_format_decimal(p_value)}" if p_value is not None else ""
        lines.append(
            f"- {metric}: Conflict {_format_decimal(conflict_mean)} vs "
            f"Aligned {_format_decimal(aligned_mean)}{p_text}"
        )

    for group_name, group_data in (("Conflict", conflict), ("Aligned", aligned)):
        patterns = _dict(group_data.get("pattern_counts"))
        if patterns:
            lines.append(f"- {group_name} patterns: {_format_mapping(patterns)}")
    return lines


def _state_to_error_lines(data: dict[str, Any] | None, missing: str | None) -> list[str]:
    if data is None:
        return [f"- State-to-error input missing: {missing}"]

    overall = _dict(data.get("overall"))
    if not overall:
        return ["- No state/error rows found."]

    lines: list[str] = []
    sorted_items = sorted(
        overall.items(),
        key=lambda item: (
            -_float_or_zero(_dict(item[1]).get("error_rate")),
            str(item[0]),
        ),
    )
    for pattern, stats_value in sorted_items:
        stats = _dict(stats_value)
        lines.append(
            f"- {pattern}: n={_format_count(stats.get('n'))}, "
            f"error_rate={_format_decimal(stats.get('error_rate'))}, "
            f"abstain_rate={_format_decimal(stats.get('abstain_rate'))}"
        )
    return lines


def _repr_comparison_lines(data: dict[str, Any] | None, missing: str | None) -> list[str]:
    if data is None:
        return [f"- Raw vs TME input missing: {missing}"]
    entries = _named_entries(data)
    if not entries:
        return ["- No raw/TME comparison rows found."]
    return [f"- {name}: {_format_entry(entry)}" for name, entry in entries]


def _efficiency_comparison_lines(data: dict[str, Any] | None, missing: str | None) -> list[str]:
    if data is None:
        return [f"- Efficiency input missing: {missing}"]
    entries = _named_entries(data)
    if not entries:
        return ["- No efficiency comparison rows found."]
    return [f"- {name}: {_format_entry(entry)}" for name, entry in entries]


def _missing_lines(missing_inputs: list[str]) -> list[str]:
    if not missing_inputs:
        return ["- None."]
    return [f"- {missing_input}" for missing_input in missing_inputs]


def _conflict_total(data: dict[str, Any]) -> int:
    groups = _dict(data.get("groups"))
    return sum(int(_dict(group).get("n") or 0) for group in groups.values())


def _state_total(data: dict[str, Any]) -> int:
    metadata = _dict(data.get("metadata"))
    if metadata.get("merged_count") is not None:
        return int(metadata["merged_count"])
    overall = _dict(data.get("overall"))
    return sum(int(_dict(stats).get("n") or 0) for stats in overall.values())


def _comparison_total(entries: list[tuple[str, dict[str, Any]]]) -> int:
    counts = [entry.get("n") for _name, entry in entries if entry.get("n") is not None]
    if counts:
        return sum(int(count) for count in counts)
    return len(entries)


def _named_entries(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    for key in ("representations", "methods", "groups", "results", "comparisons"):
        value = data.get(key)
        if isinstance(value, dict):
            return [(str(name), _dict(entry)) for name, entry in value.items()]
        if isinstance(value, list):
            return [_entry_from_list_item(item, index) for index, item in enumerate(value)]

    candidate_items = [
        (key, value)
        for key, value in data.items()
        if isinstance(value, dict) and key not in {"metadata", "inputs", "tests", "p_values"}
    ]
    return [(str(name), _dict(entry)) for name, entry in candidate_items]


def _entry_from_list_item(item: Any, index: int) -> tuple[str, dict[str, Any]]:
    entry = _dict(item)
    name = (
        entry.get("name")
        or entry.get("method")
        or entry.get("repr_key")
        or entry.get("representation")
        or f"row_{index + 1}"
    )
    return str(name), entry


def _format_entry(entry: dict[str, Any]) -> str:
    flattened = {key: value for key, value in entry.items() if key != "metrics"}
    flattened.update(_dict(entry.get("metrics")))
    preferred = (
        "n",
        "error_auc",
        "auc",
        "accuracy",
        "error_rate",
        "mean_seconds",
        "seconds",
        "mean_cost",
        "cost",
        "speedup_seconds",
        "speedup",
    )
    keys = _ordered_keys(preferred, flattened)
    if not keys:
        return "no numeric summary fields"
    return ", ".join(f"{key}={_format_value(flattened[key])}" for key in keys)


def _ordered_keys(preferred: tuple[str, ...], *mappings: dict[str, Any]) -> list[str]:
    available = {key for mapping in mappings for key in mapping}
    ordered = [key for key in preferred if key in available]
    ordered.extend(sorted(key for key in available if key not in set(preferred)))
    return ordered


def _format_mapping(mapping: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_format_value(value)}" for key, value in sorted(mapping.items()))


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_decimal(value)
    if value is None:
        return "n/a"
    return str(value)


def _format_count(value: Any) -> str:
    return str(int(value)) if value is not None else "n/a"


def _format_decimal(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return str(value)


def _float_or_zero(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
