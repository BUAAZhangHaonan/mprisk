from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from mprisk.evaluation.efficiency import (
    compare_t0_posthoc_efficiency,
    read_timing_log,
    speedup_seconds,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _timing_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "s1",
            "model_key": "qwen",
            "protocol": "t0_vs_posthoc",
            "method": "t0",
            "prefill_seconds": 1.0,
            "decode_seconds": 3.0,
            "total_seconds": 4.0,
            "num_generated_tokens": 20,
            "prompt_count": 1,
        },
        {
            "sample_id": "s2",
            "model_key": "qwen",
            "protocol": "t0_vs_posthoc",
            "method": "t0_state",
            "prefill_seconds": 2.0,
            "decode_seconds": 4.0,
            "total_seconds": 6.0,
            "num_generated_tokens": 30,
            "prompt_count": 1,
        },
        {
            "sample_id": "s1",
            "model_key": "qwen",
            "protocol": "t0_vs_posthoc",
            "method": "posthoc_full",
            "prefill_seconds": 2.0,
            "decode_seconds": 8.0,
            "total_seconds": 10.0,
            "num_generated_tokens": 80,
            "prompt_count": 4,
        },
        {
            "sample_id": "s2",
            "model_key": "qwen",
            "protocol": "t0_vs_posthoc",
            "method": "posthoc_full_response",
            "prefill_seconds": 4.0,
            "decode_seconds": 10.0,
            "total_seconds": 14.0,
            "num_generated_tokens": 100,
            "prompt_count": 4,
        },
    ]


def test_speedup_seconds() -> None:
    assert speedup_seconds(10.0, 4.0) == 6.0


def test_read_timing_log_normalizes_method_variants(tmp_path: Path) -> None:
    timing_log = tmp_path / "timing.jsonl"
    _write_jsonl(timing_log, _timing_rows())

    rows = read_timing_log(timing_log)

    assert [row["method"] for row in rows] == [
        "t0_state",
        "t0_state",
        "posthoc_full_response",
        "posthoc_full_response",
    ]
    assert rows[0]["total_seconds"] == 4.0


def test_compare_t0_posthoc_efficiency_writes_summary_json_and_csv(tmp_path: Path) -> None:
    timing_log = tmp_path / "timing.jsonl"
    output_dir = tmp_path / "efficiency"
    _write_jsonl(timing_log, _timing_rows())

    result = compare_t0_posthoc_efficiency(timing_log_path=timing_log, output_dir=output_dir)

    assert result.count == 4
    assert result.json_path == output_dir / "efficiency_comparison.json"
    assert result.csv_path == output_dir / "efficiency_comparison.csv"
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    t0_stats = payload["methods"]["t0_state"]
    assert t0_stats["n"] == 2
    assert t0_stats["total_seconds_mean"] == 5.0
    assert t0_stats["total_seconds_median"] == 5.0
    assert t0_stats["prefill_seconds_mean"] == 1.5
    assert t0_stats["decode_seconds_mean"] == 3.5
    assert t0_stats["num_generated_tokens_mean"] == 25.0
    assert t0_stats["prompt_count_mean"] == 1.0
    assert t0_stats["prompt_count_adjusted_cost"] == 5.0

    posthoc_stats = payload["methods"]["posthoc_full_response"]
    assert posthoc_stats["total_seconds_mean"] == 12.0
    assert posthoc_stats["prompt_count_mean"] == 4.0
    assert posthoc_stats["prompt_count_adjusted_cost"] == 48.0

    comparison = payload["comparison"]
    assert comparison["status"] == "ok"
    assert comparison["speedup_ratio"] == 2.4
    assert comparison["saved_seconds"] == 7.0
    assert comparison["prompt_count_adjusted_costs"] == {
        "t0_state": 5.0,
        "posthoc_full_response": 48.0,
    }

    with result.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {
        "row_type": "method",
        "method": "t0_state",
        "n": "2",
        "total_seconds_mean": "5.0",
        "total_seconds_median": "5.0",
        "prefill_seconds_mean": "1.5",
        "decode_seconds_mean": "3.5",
        "num_generated_tokens_mean": "25.0",
        "prompt_count_mean": "1.0",
        "prompt_count_adjusted_cost": "5.0",
        "speedup_ratio": "",
        "saved_seconds": "",
        "status": "",
    } in rows
    assert {
        "row_type": "comparison",
        "method": "t0_state_vs_posthoc_full_response",
        "n": "",
        "total_seconds_mean": "",
        "total_seconds_median": "",
        "prefill_seconds_mean": "",
        "decode_seconds_mean": "",
        "num_generated_tokens_mean": "",
        "prompt_count_mean": "",
        "prompt_count_adjusted_cost": "",
        "speedup_ratio": "2.4",
        "saved_seconds": "7.0",
        "status": "ok",
    } in rows


def test_compare_t0_posthoc_efficiency_handles_missing_comparison_method(tmp_path: Path) -> None:
    timing_log = tmp_path / "timing.jsonl"
    output_dir = tmp_path / "efficiency"
    _write_jsonl(timing_log, _timing_rows()[:2])

    result = compare_t0_posthoc_efficiency(timing_log_path=timing_log, output_dir=output_dir)

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["methods"]["t0_state"]["n"] == 2
    assert payload["comparison"] == {
        "status": "missing_method",
        "t0_method": "t0_state",
        "posthoc_method": "posthoc_full_response",
        "missing_methods": ["posthoc_full_response"],
        "speedup_ratio": None,
        "saved_seconds": None,
        "prompt_count_adjusted_costs": {"t0_state": 5.0},
    }


def test_efficiency_comparison_cli_accepts_timing_log_and_output_dir(tmp_path: Path) -> None:
    timing_log = tmp_path / "timing.jsonl"
    output_dir = tmp_path / "cli-efficiency"
    _write_jsonl(timing_log, _timing_rows())

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_efficiency_comparison.py",
            "--timing-log",
            str(timing_log),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "efficiency_comparison_json=" in completed.stdout
    assert "efficiency_comparison_csv=" in completed.stdout
    assert (output_dir / "efficiency_comparison.json").exists()
    assert (output_dir / "efficiency_comparison.csv").exists()
