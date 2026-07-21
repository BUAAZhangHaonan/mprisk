"""Export per-condition Full/Prompt-KV/generation timing tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("M1", "M2", "M12")


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_ms": None, "std_ms": None, "median_ms": None, "p95_ms": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def export(path: Path, output: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("timing_runs.jsonl is empty")
    p_values = sorted({int(row["p"]) for row in rows})
    result: dict[str, object] = {
        "schema": "mprisk_qwen_vl_prefill_generation_condition_report_v1",
        "source": str(path.resolve()),
        "p_values": p_values,
        "generation_scope": "full_input_only",
        "rows": [],
    }
    csv_rows: list[dict[str, object]] = []
    for p in p_values:
        selected = [row for row in rows if int(row["p"]) == p]
        for condition in CONDITIONS:
            full = _stats([float(row["full_condition_ms"][condition]) for row in selected])
            kv = _stats([float(row["kv_condition_ms"][condition]) for row in selected])
            generation = _stats([float(row["generation_condition_ms"][condition]) for row in selected])
            speedup = None if p == 1 else full["mean_ms"] / kv["mean_ms"]
            item = {
                "p": p,
                "condition": condition,
                "sample_repeat_rows": len(selected),
                "full_prefill": full,
                "prompt_kv_prefill": kv,
                "prefill_speedup": speedup,
                "generation_call": generation,
                "generation_scope": "full_input_only",
            }
            result["rows"].append(item)
            csv_rows.append({
                "p": p,
                "condition": condition,
                "sample_repeat_rows": len(selected),
                "full_prefill_mean_ms": full["mean_ms"],
                "full_prefill_std_ms": full["std_ms"],
                "full_prefill_median_ms": full["median_ms"],
                "full_prefill_p95_ms": full["p95_ms"],
                "prompt_kv_prefill_mean_ms": kv["mean_ms"],
                "prompt_kv_prefill_std_ms": kv["std_ms"],
                "prompt_kv_prefill_median_ms": kv["median_ms"],
                "prompt_kv_prefill_p95_ms": kv["p95_ms"],
                "prefill_speedup": speedup,
                "generation_call_mean_ms": generation["mean_ms"],
                "generation_call_std_ms": generation["std_ms"],
                "generation_call_median_ms": generation["median_ms"],
                "generation_call_p95_ms": generation["p95_ms"],
                "generation_scope": "full_input_only",
            })
    output.mkdir(parents=True, exist_ok=True)
    (output / "condition_timing_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "condition_timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    export(args.input, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
