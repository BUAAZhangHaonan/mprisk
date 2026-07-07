from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mprisk.evaluation.error_analysis import analyze_state_to_error


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sample_inputs(tmp_path: Path) -> tuple[Path, Path]:
    state_path = tmp_path / "state_patterns.jsonl"
    prediction_path = tmp_path / "predictions_normalized.jsonl"
    _write_jsonl(
        state_path,
        [
            {"sample_id": "s1", "sample_type": "Conflict", "pattern": "Confusion"},
            {"sample_id": "s2", "sample_type": "Conflict", "pattern": "Confusion"},
            {"sample_id": "s3", "sample_type": "Aligned", "pattern": "Consensus"},
            {"sample_id": "s4", "sample_type": "Aligned", "pattern": "Consensus"},
            {"sample_id": "s5", "sample_type": "Conflict", "pattern": "Dominant"},
        ],
    )
    _write_jsonl(
        prediction_path,
        [
            {
                "sample_id": "s1",
                "normalized_prediction": "joy",
                "target_label": "joy",
                "is_correct": True,
            },
            {
                "sample_id": "s2",
                "prediction": "anger",
                "target_label": "joy",
                "is_correct": False,
            },
            {
                "sample_id": "s3",
                "normalized_prediction": "uncertain",
                "target_label": "sadness",
                "is_correct": True,
            },
            {
                "sample_id": "s4",
                "prediction": "abstain",
                "target_label": "sadness",
                "is_correct": False,
                "outcome": "abstain",
            },
            {
                "sample_id": "s5",
                "normalized_prediction": "joy",
                "target_label": "anger",
                "is_correct": False,
                "status": "complete",
            },
        ],
    )
    return state_path, prediction_path


def test_analyze_state_to_error_merges_rows_and_writes_json_and_csv(tmp_path: Path) -> None:
    state_path, prediction_path = _sample_inputs(tmp_path)
    output_dir = tmp_path / "analysis"

    result = analyze_state_to_error(
        state_patterns_path=state_path,
        predictions_path=prediction_path,
        output_dir=output_dir,
    )

    assert result.count == 5
    assert result.json_path == output_dir / "state_to_error.json"
    assert result.csv_path == output_dir / "state_to_error.csv"
    assert result.json_path.exists()
    assert result.csv_path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["merged_count"] == 5
    assert payload["metadata"]["missing_prediction_sample_ids"] == []
    assert payload["metadata"]["missing_state_sample_ids"] == []


def test_state_to_error_overall_stats_and_abstain_handling(tmp_path: Path) -> None:
    state_path, prediction_path = _sample_inputs(tmp_path)

    result = analyze_state_to_error(
        state_patterns_path=state_path,
        predictions_path=prediction_path,
        output_dir=tmp_path / "analysis",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["overall"]["Confusion"] == {
        "n": 2,
        "correct_count": 1,
        "error_count": 1,
        "error_rate": 0.5,
        "abstain_count": 0,
        "abstain_rate": 0.0,
    }
    assert payload["overall"]["Consensus"] == {
        "n": 2,
        "correct_count": 0,
        "error_count": 2,
        "error_rate": 1.0,
        "abstain_count": 2,
        "abstain_rate": 1.0,
    }
    assert payload["metadata"]["abstain_policy"] == (
        "Rows are marked abstain when prediction/outcome/status is abstain or "
        "normalized_prediction is uncertain/invalid. Abstain rows are treated as not "
        "correct unless outcome/status explicitly separates abstention from correctness."
    )


def test_state_to_error_by_sample_type_stats(tmp_path: Path) -> None:
    state_path, prediction_path = _sample_inputs(tmp_path)

    result = analyze_state_to_error(
        state_patterns_path=state_path,
        predictions_path=prediction_path,
        output_dir=tmp_path / "analysis",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["by_sample_type"]["Conflict"]["Confusion"]["n"] == 2
    assert payload["by_sample_type"]["Conflict"]["Confusion"]["error_rate"] == 0.5
    assert payload["by_sample_type"]["Conflict"]["Dominant"] == {
        "n": 1,
        "correct_count": 0,
        "error_count": 1,
        "error_rate": 1.0,
        "abstain_count": 0,
        "abstain_rate": 0.0,
    }
    assert payload["by_sample_type"]["Aligned"]["Consensus"]["abstain_count"] == 2


def test_state_to_error_json_contains_association_metadata(tmp_path: Path) -> None:
    state_path, prediction_path = _sample_inputs(tmp_path)

    result = analyze_state_to_error(
        state_patterns_path=state_path,
        predictions_path=prediction_path,
        output_dir=tmp_path / "analysis",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    association = payload["tests"]["association"]
    assert association["table"]["columns"] == ["correct", "error"]
    assert association["table"]["patterns"] == ["Confusion", "Consensus", "Dominant"]
    assert association["table"]["counts"] == [[1, 1], [0, 2], [0, 1]]
    assert association["method"] in {"chi_square", "fisher_exact", None}
    assert association["status"] in {"ok", "insufficient_data"}


def test_state_to_error_csv_contains_overall_and_sample_type_rows(tmp_path: Path) -> None:
    state_path, prediction_path = _sample_inputs(tmp_path)

    result = analyze_state_to_error(
        state_patterns_path=state_path,
        predictions_path=prediction_path,
        output_dir=tmp_path / "analysis",
    )

    with result.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {
        "scope": "overall",
        "sample_type": "",
        "pattern": "Confusion",
        "n": "2",
        "correct_count": "1",
        "error_count": "1",
        "error_rate": "0.5",
        "abstain_count": "0",
        "abstain_rate": "0.0",
    } in rows
    assert {
        "scope": "sample_type",
        "sample_type": "Aligned",
        "pattern": "Consensus",
        "n": "2",
        "correct_count": "0",
        "error_count": "2",
        "error_rate": "1.0",
        "abstain_count": "2",
        "abstain_rate": "1.0",
    } in rows


def test_state_to_error_cli_accepts_required_arguments(tmp_path: Path) -> None:
    state_path, prediction_path = _sample_inputs(tmp_path)
    output_dir = tmp_path / "cli-analysis"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_state_to_error.py",
            "--state-patterns",
            str(state_path),
            "--predictions",
            str(prediction_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "state_to_error_json=" in completed.stdout
    assert (output_dir / "state_to_error.json").exists()
    assert (output_dir / "state_to_error.csv").exists()


def test_state_to_error_requires_unique_sample_ids(tmp_path: Path) -> None:
    state_path = tmp_path / "state_patterns.jsonl"
    prediction_path = tmp_path / "predictions_normalized.jsonl"
    _write_jsonl(
        state_path,
        [
            {"sample_id": "s1", "sample_type": "Conflict", "pattern": "Confusion"},
            {"sample_id": "s1", "sample_type": "Conflict", "pattern": "Consensus"},
        ],
    )
    _write_jsonl(
        prediction_path,
        [
            {
                "sample_id": "s1",
                "normalized_prediction": "joy",
                "target_label": "joy",
                "is_correct": True,
            }
        ],
    )

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        analyze_state_to_error(
            state_patterns_path=state_path,
            predictions_path=prediction_path,
            output_dir=tmp_path / "analysis",
        )
