from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mprisk.evaluation.main_results import compare_conflict_vs_aligned


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_conflict_vs_aligned_uses_state_patterns_for_grouping_and_patterns(tmp_path: Path) -> None:
    sdr_path = tmp_path / "sdr_scores.jsonl"
    patterns_path = tmp_path / "state_patterns.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(
        sdr_path,
        [
            {"sample_id": "c1", "sample_type": "Aligned", "S_mean": 0.2, "D": 0.8, "R": -0.5},
            {"sample_id": "c2", "sample_type": "Conflict", "S_mean": 0.4, "D": 1.0, "R": 0.1},
            {"sample_id": "a1", "sample_type": "Conflict", "S_mean": 0.8, "D": 0.2, "R": -0.2},
            {"sample_id": "a2", "sample_type": "Aligned", "S_mean": 1.0, "D": 0.4, "R": 0.4},
        ],
    )
    _write_jsonl(
        patterns_path,
        [
            {"sample_id": "c1", "sample_type": "Conflict", "pattern": "S-low/D-high/R-negative"},
            {"sample_id": "c2", "sample_type": "Conflict", "pattern": "S-low/D-high/R-positive"},
            {"sample_id": "a1", "sample_type": "Aligned", "pattern": "S-high/D-low/R-negative"},
            {"sample_id": "a2", "sample_type": "Aligned", "pattern": "S-high/D-low/R-negative"},
        ],
    )

    result = compare_conflict_vs_aligned(
        sdr_scores_path=sdr_path,
        state_patterns_path=patterns_path,
        output_dir=output_dir,
    )

    assert result.json_path == output_dir / "conflict_vs_aligned.json"
    assert result.csv_path == output_dir / "conflict_vs_aligned.csv"
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    conflict = payload["groups"]["Conflict"]
    aligned = payload["groups"]["Aligned"]
    assert conflict["n"] == 2
    assert aligned["n"] == 2
    assert conflict["metrics"]["S_mean"]["mean"] == pytest.approx(0.3)
    assert conflict["metrics"]["S_mean"]["median"] == pytest.approx(0.3)
    assert conflict["metrics"]["S_mean"]["std"] == pytest.approx(0.1414213562)
    assert aligned["metrics"]["D"]["mean"] == pytest.approx(0.3)
    assert conflict["metrics"]["abs_R"]["mean"] == pytest.approx(0.3)
    assert conflict["metrics"]["abs_R"]["std"] == pytest.approx(0.2828427125)

    assert conflict["pattern_counts"] == {
        "S-low/D-high/R-negative": 1,
        "S-low/D-high/R-positive": 1,
    }
    assert conflict["pattern_proportions"] == {
        "S-low/D-high/R-negative": 0.5,
        "S-low/D-high/R-positive": 0.5,
    }
    assert aligned["pattern_counts"] == {"S-high/D-low/R-negative": 2}
    assert aligned["pattern_proportions"] == {"S-high/D-low/R-negative": 1.0}


def test_conflict_vs_aligned_writes_p_value_metadata_and_compact_csv(tmp_path: Path) -> None:
    sdr_path = tmp_path / "sdr_scores.jsonl"
    patterns_path = tmp_path / "state_patterns.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(
        sdr_path,
        [
            {"sample_id": "c1", "sample_type": "Conflict", "S_mean": 0.1, "D": 0.9, "R": -0.9},
            {"sample_id": "c2", "sample_type": "Conflict", "S_mean": 0.2, "D": 0.8, "R": -0.8},
            {"sample_id": "a1", "sample_type": "Aligned", "S_mean": 0.9, "D": 0.1, "R": 0.1},
            {"sample_id": "a2", "sample_type": "Aligned", "S_mean": 0.8, "D": 0.2, "R": 0.2},
        ],
    )
    _write_jsonl(
        patterns_path,
        [
            {"sample_id": "c1", "sample_type": "Conflict", "pattern": "conflict-pattern"},
            {"sample_id": "c2", "sample_type": "Conflict", "pattern": "conflict-pattern"},
            {"sample_id": "a1", "sample_type": "Aligned", "pattern": "aligned-pattern"},
            {"sample_id": "a2", "sample_type": "Aligned", "pattern": "aligned-pattern"},
        ],
    )

    result = compare_conflict_vs_aligned(
        sdr_scores_path=sdr_path,
        state_patterns_path=patterns_path,
        output_dir=output_dir,
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    for metric in ("S_mean", "D", "abs_R"):
        assert set(payload["p_values"][metric]) >= {"p_value", "method", "statistic"}
        assert payload["p_values"][metric]["method"]
    assert set(payload["p_values"]["pattern_distribution"]) >= {"p_value", "method", "statistic"}
    assert payload["p_values"]["pattern_distribution"]["method"]

    with result.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {"group", "metric", "mean", "median", "std", "n"} <= set(rows[0])
    assert any(row["group"] == "Conflict" and row["metric"] == "abs_R" for row in rows)
    conflict_pattern = next(
        row
        for row in rows
        if row["group"] == "Conflict" and row["metric"] == "pattern:conflict-pattern"
    )
    assert conflict_pattern["mean"] == "1.0"
    assert conflict_pattern["n"] == "2"
