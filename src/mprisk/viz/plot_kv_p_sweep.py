"""Latency-stability plots for a Qwen-VL KV-cache prompt sweep.

The sweep runner owns measurement. This module only normalizes recorded
measurements and renders them. Missing metrics remain None and are shown as
Pending rather than being filled with illustrative values.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


SCHEMA = "mprisk_qwen_vl_kv_p_sweep_curve_v1"
EXPECTED_P_VALUES = (1, 2, 4, 8, 16, 32, 64)
CSV_FIELDS = (
    "prompt_count",
    "latency_median_seconds",
    "latency_p95_seconds",
    "latency_mean_seconds",
    "latency_n",
    "pattern_agreement",
    "metric_convergence",
    "metric_convergence_s",
    "metric_convergence_d",
    "metric_convergence_r",
    "state_index_error_mean",
    "stability_n",
    "status",
)


def load_sweep(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and normalize one runner JSON file."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        metadata: dict[str, Any] = {}
        raw_rows = payload
    elif isinstance(payload, Mapping):
        metadata = {
            str(key): value
            for key, value in payload.items()
            if key not in {"rows", "points", "results", "summary"}
        }
        raw_rows = next(
            (
                payload[key]
                for key in ("rows", "points", "results", "summary")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if raw_rows is None:
            raise ValueError("sweep JSON must contain a rows, points, or results list")
    else:
        raise ValueError("sweep JSON root must be an object or list")
    return metadata, normalize_rows(raw_rows)


def normalize_rows(raw_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one deterministic flat row per prompt count."""
    normalized: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("each sweep row must be an object")
        prompt_count = _number(raw, "prompt_count", "P", "p")
        if prompt_count is None or prompt_count <= 0 or not float(prompt_count).is_integer():
            raise ValueError(f"invalid prompt_count in sweep row: {raw!r}")
        latency = raw.get("latency", raw.get("timing"))
        if not isinstance(latency, Mapping):
            latency = raw
        stability = raw.get("stability")
        if not isinstance(stability, Mapping):
            stability = raw
        normalized.append(
            {
                "prompt_count": int(prompt_count),
                "latency_median_seconds": _scaled_number(
                    latency,
                    ("median_seconds", "latency_median_seconds", "median"),
                    ("total_median_ms",),
                ),
                "latency_p95_seconds": _scaled_number(
                    latency,
                    ("p95_seconds", "latency_p95_seconds", "p95"),
                    ("total_p95_ms",),
                ),
                "latency_mean_seconds": _scaled_number(
                    latency,
                    ("mean_seconds", "latency_mean_seconds", "mean"),
                    ("total_mean_ms",),
                ),
                "latency_n": _integer_or_none(
                    latency.get("n", latency.get("measured_runs", raw.get("latency_n")))
                ),
                "pattern_agreement": _number(
                    stability, "pattern_agreement", "state_pattern_agreement"
                ),
                "metric_convergence": _number(stability, "metric_convergence"),
                "metric_convergence_s": _number(
                    stability, "metric_convergence_s", "s_convergence"
                ),
                "metric_convergence_d": _number(
                    stability, "metric_convergence_d", "d_convergence"
                ),
                "metric_convergence_r": _number(
                    stability, "metric_convergence_r", "r_convergence"
                ),
                "state_index_error_mean": _number(
                    stability, "state_index_error_mean", "state_index_mae_mean"
                ),
                "stability_n": _integer_or_none(
                    stability.get("n", stability.get("shared_sample_count", raw.get("stability_n")))
                ),
                "status": str(raw.get("status", "ok")),
            }
        )
    normalized.sort(key=lambda row: row["prompt_count"])
    counts = [row["prompt_count"] for row in normalized]
    if len(counts) != len(set(counts)):
        raise ValueError("sweep contains duplicate prompt_count values")
    return normalized


def export_curve(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    expected_p_values: Iterable[int] = EXPECTED_P_VALUES,
) -> dict[str, str | int | list[int]]:
    """Export normalized data and a two-panel vector PDF/PNG curve."""
    metadata, rows = load_sweep(input_path)
    expected = tuple(int(value) for value in expected_p_values)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    present = {row["prompt_count"] for row in rows}
    missing = [value for value in expected if value not in present]
    payload = {
        "schema": SCHEMA,
        "input": str(Path(input_path)),
        "input_sha256": _sha256(Path(input_path)),
        "expected_prompt_counts": list(expected),
        "missing_prompt_counts": missing,
        "metadata": metadata,
        "rows": rows,
    }
    json_path = output_root / "kv_p_sweep_curve.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = output_root / "kv_p_sweep_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    pdf_path = output_root / "kv_p_sweep_latency_stability.pdf"
    png_path = output_root / "kv_p_sweep_latency_stability.png"
    _render(rows, expected, pdf_path, png_path)
    return {
        "schema": SCHEMA,
        "rows": len(rows),
        "missing_prompt_counts": missing,
        "json": str(json_path),
        "csv": str(csv_path),
        "pdf": str(pdf_path),
        "png": str(png_path),
    }


def _render(
    rows: list[dict[str, Any]],
    expected: tuple[int, ...],
    pdf_path: Path,
    png_path: Path,
) -> None:
    x = [row["prompt_count"] for row in rows]
    has_latency = any(row["latency_median_seconds"] is not None for row in rows)
    has_stability = any(
        row["pattern_agreement"] is not None
        or row["metric_convergence"] is not None
        or row["state_index_error_mean"] is not None
        for row in rows
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.82, wspace=0.18)
    ax_latency, ax_stability = axes
    _setup_log_x(ax_latency, expected)
    _setup_log_x(ax_stability, expected)
    if has_latency:
        median_values = [row["latency_median_seconds"] for row in rows]
        p95_values = [row["latency_p95_seconds"] for row in rows]
        valid_median = [
            (p, value)
            for p, value in zip(x, median_values, strict=True)
            if value is not None
        ]
        valid_p95 = [
            (p, value) for p, value in zip(x, p95_values, strict=True) if value is not None
        ]
        if valid_median:
            ax_latency.plot(
                *zip(*valid_median, strict=True),
                marker="o",
                color="#1f4e79",
                label="Median",
            )
        if valid_p95:
            ax_latency.plot(
                *zip(*valid_p95, strict=True),
                marker="s",
                linestyle="--",
                color="#b04a4a",
                label="P95",
            )
        ax_latency.legend(frameon=False)
    else:
        _pending(ax_latency)
    ax_latency.set_title("KV-cache latency")
    ax_latency.set_xlabel("Equivalent prompts (P)")
    ax_latency.set_ylabel("Wall time (s)")
    if has_stability:
        pattern = [
            (p, row["pattern_agreement"])
            for p, row in zip(x, rows, strict=True)
            if row["pattern_agreement"] is not None
        ]
        convergence = [
            (p, row["metric_convergence"])
            for p, row in zip(x, rows, strict=True)
            if row["metric_convergence"] is not None
        ]
        error = [
            (p, row["state_index_error_mean"])
            for p, row in zip(x, rows, strict=True)
            if row["state_index_error_mean"] is not None
        ]
        if pattern:
            ax_stability.plot(
                *zip(*pattern, strict=True),
                marker="o",
                color="#2c7a4b",
                label="Pattern agreement",
            )
        if convergence:
            ax_stability.plot(
                *zip(*convergence, strict=True),
                marker="s",
                color="#7b4b9a",
                label="Metric convergence",
            )
        if error:
            ax_stability.plot(
                *zip(*error, strict=True),
                marker="^",
                color="#a05a2c",
                label="State-index error vs P=64",
            )
        if pattern or convergence:
            ax_stability.set_ylim(0.0, 1.05)
        if pattern or convergence or error:
            ax_stability.legend(frameon=False)
        else:
            _pending(ax_stability)
    else:
        _pending(ax_stability)
    ax_stability.set_title("State stability")
    ax_stability.set_xlabel("Equivalent prompts (P)")
    ax_stability.set_ylabel("Agreement / error")
    fig.suptitle("Qwen3-VL-8B KV-cache prompt sweep", fontsize=13)
    fig.text(
        0.01,
        0.035,
        "Only recorded measurements are shown; missing metrics are Pending.",
        fontsize=8,
    )
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=220)
    plt.close(fig)


def _setup_log_x(axis: Any, expected: tuple[int, ...]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(expected)
    axis.set_xticklabels([str(value) for value in expected])
    axis.grid(True, which="both", alpha=0.25)


def _pending(axis: Any) -> None:
    axis.text(
        0.5,
        0.5,
        "Pending",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=13,
    )


def _number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None or value == "":
            continue
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite sweep value for {key}: {value!r}")
        return result
    return None


def _scaled_number(
    mapping: Mapping[str, Any], second_keys: tuple[str, ...], millisecond_keys: tuple[str, ...]
) -> float | None:
    value = _number(mapping, *second_keys)
    if value is not None:
        return value
    value = _number(mapping, *millisecond_keys)
    return None if value is None else value / 1000.0


def _integer_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    result = int(value)
    if result < 0:
        raise ValueError(f"negative sample count: {value!r}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
