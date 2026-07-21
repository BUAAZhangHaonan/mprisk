"""Clean, artifact-backed visualizations for the Qwen3-VL KV timing sweep.

This module consumes the recorded condition-level timing CSV and the recorded
P-sweep stability CSV. It never fills missing values or manufactures points.
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


EXPECTED_P = (1, 2, 4, 8, 16, 32, 64)
SCHEMA = "mprisk_qwen3_vl_kv_latency_stability_tables_v1"
TIMING_FIELDS = (
    "prompt_count",
    "full_prefill_seconds",
    "prompt_kv_prefill_seconds",
    "prefill_speedup",
    "generation_seconds",
    "full_end_to_end_seconds",
    "prompt_kv_end_to_end_seconds",
    "end_to_end_speedup",
    "condition_count",
)
STABILITY_FIELDS = (
    "prompt_count",
    "state_index_mae",
    "state_index_mae_s",
    "state_index_mae_d",
    "state_index_mae_r",
    "sample_count",
    "measured_runs",
    "reference_prompt_count",
)


def export_tables(
    timing_csv: str | Path,
    stability_csv: str | Path,
    output_dir: str | Path,
    *,
    stability_tolerance: float = 0.10,
    expected_p: Iterable[int] = EXPECTED_P,
) -> dict[str, Any]:
    """Aggregate recorded rows, export tables, and render three clean PDFs."""
    timing_path = Path(timing_csv)
    stability_path = Path(stability_csv)
    expected = tuple(int(p) for p in expected_p)
    if not expected or any(p <= 0 for p in expected):
        raise ValueError("expected_p must contain positive prompt counts")
    if stability_tolerance < 0 or not math.isfinite(stability_tolerance):
        raise ValueError("stability_tolerance must be finite and non-negative")

    timing_rows = aggregate_timing(timing_path)
    stability_rows = load_stability(stability_path)
    timing_by_p = {int(row["prompt_count"]): row for row in timing_rows}
    stability_by_p = {int(row["prompt_count"]): row for row in stability_rows}
    missing_timing = [p for p in expected if p not in timing_by_p]
    missing_stability = [p for p in expected if p not in stability_by_p]

    pareto_rows = _pareto_rows(timing_rows, stability_rows)
    feasible = [
        row
        for row in pareto_rows
        if row["state_index_mae"] is not None
        and row["state_index_mae"] <= stability_tolerance
    ]
    best = min(feasible, key=lambda row: row["prompt_kv_end_to_end_seconds"]) if feasible else None
    knee = _knee_point(pareto_rows)
    payload = {
        "schema": SCHEMA,
        "timing_input": str(timing_path),
        "timing_input_sha256": _sha256(timing_path),
        "stability_input": str(stability_path),
        "stability_input_sha256": _sha256(stability_path),
        "expected_prompt_counts": list(expected),
        "missing_timing_prompt_counts": missing_timing,
        "missing_stability_prompt_counts": missing_stability,
        "stability_definition": "mean absolute State Index error relative to the nested P=64 reference on shared samples",
        "stability_tolerance": stability_tolerance,
        "selection_rule": "among Pareto points with State Index MAE <= stability_tolerance, choose the minimum Prompt-KV plus generation latency",
        "timing_rows": timing_rows,
        "stability_rows": stability_rows,
        "pareto_frontier_prompt_counts": [row["prompt_count"] for row in pareto_rows],
        "constrained_choice": best,
        "knee_choice": knee,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "kv_latency_stability_tables.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "kv_total_latency_table.csv", timing_rows, TIMING_FIELDS)
    _write_csv(output / "kv_stability_table.csv", stability_rows, STABILITY_FIELDS)
    _write_csv(
        output / "kv_pareto_table.csv",
        [
            {
                "prompt_count": row["prompt_count"],
                "prompt_kv_end_to_end_seconds": row["prompt_kv_end_to_end_seconds"],
                "state_index_mae": row["state_index_mae"],
                "pareto": True,
                "within_tolerance": (
                    row["state_index_mae"] is not None
                    and row["state_index_mae"] <= stability_tolerance
                ),
                "knee": bool(knee and row["prompt_count"] == knee["prompt_count"]),
            }
            for row in pareto_rows
        ],
        (
            "prompt_count",
            "prompt_kv_end_to_end_seconds",
            "state_index_mae",
            "pareto",
            "within_tolerance",
            "knee",
        ),
    )
    _render_latency(timing_rows, expected, output / "kv_total_latency.pdf", output / "kv_total_latency.png")
    _render_stability(stability_rows, expected, output / "kv_stability.pdf", output / "kv_stability.png")
    _render_pareto(
        pareto_rows,
        stability_tolerance,
        best,
        output / "kv_latency_stability_pareto.pdf",
        output / "kv_latency_stability_pareto.png",
    )
    return {
        "schema": SCHEMA,
        "timing_rows": len(timing_rows),
        "stability_rows": len(stability_rows),
        "pareto_prompt_counts": [row["prompt_count"] for row in pareto_rows],
        "constrained_choice": best,
        "knee_choice": knee,
        "output_dir": str(output),
    }


def aggregate_timing(path: str | Path) -> list[dict[str, Any]]:
    """Sum the three recorded conditions for each P (M1, M2, and M12)."""
    required = {
        "p",
        "condition",
        "full_prefill_mean_ms",
        "prompt_kv_prefill_mean_ms",
        "generation_call_mean_ms",
    }
    groups: dict[int, list[dict[str, str]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"timing CSV missing required fields: {missing}")
        for row in reader:
            p = int(row["p"])
            groups.setdefault(p, []).append(row)
    result: list[dict[str, Any]] = []
    for p in sorted(groups):
        rows = groups[p]
        if not rows:
            raise ValueError(f"no timing conditions recorded for P={p}")
        full = sum(_finite(row["full_prefill_mean_ms"]) for row in rows) / 1000.0
        kv = sum(_finite(row["prompt_kv_prefill_mean_ms"]) for row in rows) / 1000.0
        generation = sum(_finite(row["generation_call_mean_ms"]) for row in rows) / 1000.0
        full_e2e = full + generation
        kv_e2e = kv + generation
        result.append(
            {
                "prompt_count": p,
                "full_prefill_seconds": full,
                "prompt_kv_prefill_seconds": kv,
                "prefill_speedup": None if kv == 0 else full / kv,
                "generation_seconds": generation,
                "full_end_to_end_seconds": full_e2e,
                "prompt_kv_end_to_end_seconds": kv_e2e,
                "end_to_end_speedup": None if kv_e2e == 0 else full_e2e / kv_e2e,
                "condition_count": len(rows),
            }
        )
    return result


def load_stability(path: str | Path) -> list[dict[str, Any]]:
    required = {
        "p",
        "state_index_mae_mean",
        "state_index_mae_S_mean",
        "state_index_mae_D",
        "state_index_mae_R",
        "sample_count",
        "measured_runs",
        "reference_p",
    }
    result: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"stability CSV missing required fields: {missing}")
        for row in reader:
            result.append(
                {
                    "prompt_count": int(row["p"]),
                    "state_index_mae": _nullable_finite(row["state_index_mae_mean"]),
                    "state_index_mae_s": _nullable_finite(row["state_index_mae_S_mean"]),
                    "state_index_mae_d": _nullable_finite(row["state_index_mae_D"]),
                    "state_index_mae_r": _nullable_finite(row["state_index_mae_R"]),
                    "sample_count": int(row["sample_count"]),
                    "measured_runs": int(row["measured_runs"]),
                    "reference_prompt_count": int(row["reference_p"]),
                }
            )
    result.sort(key=lambda row: row["prompt_count"])
    return result


def _pareto_rows(timing_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stability = {int(row["prompt_count"]): row for row in stability_rows}
    joined = []
    for row in timing_rows:
        p = int(row["prompt_count"])
        error = stability.get(p, {}).get("state_index_mae")
        if p != 1 and error is not None:
            joined.append({**row, "state_index_mae": error})
    frontier = []
    for row in joined:
        dominated = any(
            other["prompt_kv_end_to_end_seconds"] <= row["prompt_kv_end_to_end_seconds"]
            and other["state_index_mae"] <= row["state_index_mae"]
            and (
                other["prompt_kv_end_to_end_seconds"] < row["prompt_kv_end_to_end_seconds"]
                or other["state_index_mae"] < row["state_index_mae"]
            )
            for other in joined
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: row["prompt_count"])


def _knee_point(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(rows) < 3:
        return None
    x = [float(row["prompt_kv_end_to_end_seconds"]) for row in rows]
    y = [float(row["state_index_mae"]) for row in rows]
    xn = [(value - min(x)) / (max(x) - min(x)) for value in x]
    yn = [(value - min(y)) / (max(y) - min(y)) for value in y]
    x0, y0, x1, y1 = xn[0], yn[0], xn[-1], yn[-1]
    denom = math.hypot(x1 - x0, y1 - y0)
    distances = (
        []
        if denom == 0
        else [
            abs((x1 - x0) * (y0 - yi) - (x0 - xi) * (y1 - y0)) / denom
            for xi, yi in zip(xn[1:-1], yn[1:-1], strict=True)
        ]
    )
    index = 1 + max(range(len(distances)), key=lambda item: distances[item]) if distances else 0
    return {
        **rows[index],
        "normalized_distance_to_endpoint_chord": distances[index - 1] if distances else 0.0,
    }


def _render_latency(rows: list[dict[str, Any]], expected: tuple[int, ...], pdf: Path, png: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = [row["prompt_count"] for row in rows]
    fig, (ax_prefill, ax_generation) = plt.subplots(
        2, 1, figsize=(7.0, 6.0), sharex=True, constrained_layout=True
    )
    width = 0.34
    full = [row["full_prefill_seconds"] for row in rows]
    kv = [row["prompt_kv_prefill_seconds"] for row in rows]
    ax_prefill.bar([p / 1.18 for p in x], full, width=width, color="#577590", label="Full prefill")
    ax_prefill.bar([p * 1.18 for p in x], kv, width=width, color="#f3722c", label="Prompt KV")
    ax_prefill.set_ylabel("Seconds")
    ax_prefill.set_xscale("log", base=2)
    ax_prefill.set_xticks(expected)
    ax_prefill.set_xticklabels([str(value) for value in expected])
    ax_prefill.grid(axis="y", alpha=0.22)
    ax_prefill.legend(frameon=False, ncol=2, loc="upper left")
    generation = [row["generation_seconds"] for row in rows]
    ax_generation.plot(x, generation, color="#43aa8b", marker="o", linewidth=2)
    ax_generation.set_xlabel("Equivalent prompts, P")
    ax_generation.set_ylabel("Generation seconds")
    ax_generation.set_xscale("log", base=2)
    ax_generation.set_xticks(expected)
    ax_generation.set_xticklabels([str(value) for value in expected])
    ax_generation.grid(axis="y", alpha=0.22)
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=220)
    plt.close(fig)


def _render_stability(rows: list[dict[str, Any]], expected: tuple[int, ...], pdf: Path, png: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    valid = [row for row in rows if row["state_index_mae"] is not None]
    x = [row["prompt_count"] for row in valid]
    ax.plot(x, [row["state_index_mae"] for row in valid], color="#1d3557", marker="o", linewidth=2, label="Overall")
    ax.plot(x, [row["state_index_mae_s"] for row in valid], color="#457b9d", marker="o", linewidth=1.5, label="S")
    ax.plot(x, [row["state_index_mae_d"] for row in valid], color="#e76f51", marker="o", linewidth=1.5, label="D")
    ax.plot(x, [row["state_index_mae_r"] for row in valid], color="#2a9d8f", marker="o", linewidth=1.5, label="R")
    ax.set_xscale("log", base=2)
    ax.set_xticks(expected)
    ax.set_xticklabels([str(value) for value in expected])
    ax.set_xlabel("Equivalent prompts, P")
    ax.set_ylabel("State-index MAE")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=220)
    plt.close(fig)


def _render_pareto(rows: list[dict[str, Any]], tolerance: float, choice: dict[str, Any] | None, pdf: Path, png: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    if rows:
        xs = [row["prompt_kv_end_to_end_seconds"] for row in rows]
        ys = [row["state_index_mae"] for row in rows]
        ax.plot(xs, ys, color="#adb5bd", linewidth=1.2, linestyle="--", zorder=1)
        ax.scatter(xs, ys, s=55, color="#457b9d", edgecolor="white", linewidth=0.8, zorder=2)
        for row in rows:
            ax.annotate(
                f"P={row['prompt_count']}",
                (row["prompt_kv_end_to_end_seconds"], row["state_index_mae"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.axhline(tolerance, color="#999999", linewidth=1, linestyle=":")
    if choice is not None:
        ax.scatter(
            [choice["prompt_kv_end_to_end_seconds"]],
            [choice["state_index_mae"]],
            s=150,
            facecolor="none",
            edgecolor="#d95f02",
            linewidth=2,
            zorder=3,
        )
    ax.set_xlabel("Prompt-KV + generation seconds")
    ax.set_ylabel("State-index MAE")
    ax.grid(alpha=0.22)
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, dpi=220, format="png")
    plt.close(fig)


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite timing value: {value!r}")
    return number


def _nullable_finite(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return _finite(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
