from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mprisk.evaluation.repr_comparison import compare_representations


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _repr_paths(tmp_path: Path, repr_key: str, *, d_offset: float) -> dict[str, str]:
    root = tmp_path / repr_key
    sdr_scores = root / "sdr_scores.jsonl"
    state_patterns = root / "state_patterns.jsonl"
    state_to_error = root / "state_to_error.json"
    conflict_vs_aligned = root / "conflict_vs_aligned.json"

    _write_jsonl(
        sdr_scores,
        [
            {"sample_id": "c1", "sample_type": "Conflict", "D": 0.8 + d_offset},
            {"sample_id": "c2", "sample_type": "Conflict", "D": 1.0 + d_offset},
            {"sample_id": "a1", "sample_type": "Aligned", "D": 0.2},
            {"sample_id": "a2", "sample_type": "Aligned", "D": 0.4},
        ],
    )
    _write_jsonl(
        state_patterns,
        [
            {"sample_id": "c1", "sample_type": "Conflict", "pattern": "risk"},
            {"sample_id": "c2", "sample_type": "Conflict", "pattern": "risk"},
            {"sample_id": "a1", "sample_type": "Aligned", "pattern": "safe"},
            {"sample_id": "a2", "sample_type": "Aligned", "pattern": "mixed"},
        ],
    )
    _write_json(
        state_to_error,
        {
            "metadata": {"merged_count": 4, "missing_rate": 0.25, "runtime_seconds": 3.5},
            "overall": {
                "risk": {"n": 2, "error_rate": 1.0},
                "safe": {"n": 1, "error_rate": 0.0},
                "mixed": {"n": 1, "error_rate": 0.5},
            },
            "tests": {
                "association": {
                    "status": "ok",
                    "method": "chi_square",
                    "statistic": 6.0 + d_offset,
                    "p_value": 0.01,
                }
            },
        },
    )
    _write_json(
        conflict_vs_aligned,
        {
            "metadata": {"runtime_seconds": 7.0},
            "groups": {
                "Conflict": {
                    "n": 2,
                    "metrics": {
                        "D": {"n": 2, "mean": 0.9 + d_offset, "median": 0.9, "std": 0.1414213562}
                    },
                    "pattern_counts": {"risk": 2},
                },
                "Aligned": {
                    "n": 2,
                    "metrics": {
                        "D": {"n": 2, "mean": 0.3, "median": 0.3, "std": 0.1414213562}
                    },
                    "pattern_counts": {"safe": 1, "mixed": 1},
                },
            },
        },
    )
    return {
        "sdr_scores": str(sdr_scores),
        "state_patterns": str(state_patterns),
        "state_to_error": str(state_to_error),
        "conflict_vs_aligned": str(conflict_vs_aligned),
    }


def test_compare_representations_writes_json_and_csv_with_tme_and_missing_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs/evaluation/main/qwen3_vl_8b/VT"
    repr_results = {
        "raw_layernorm_mean": _repr_paths(tmp_path, "raw_layernorm_mean", d_offset=0.0),
        "tme_supcon_v1": _repr_paths(tmp_path, "tme_supcon_v1", d_offset=0.2),
        "raw_layernorm_flat": {
            "sdr_scores": str(tmp_path / "missing/raw/sdr_scores.jsonl"),
            "state_patterns": str(tmp_path / "missing/raw/state_patterns.jsonl"),
            "state_to_error": str(tmp_path / "missing/raw/state_to_error.json"),
            "conflict_vs_aligned": str(tmp_path / "missing/raw/conflict_vs_aligned.json"),
        },
    }

    result = compare_representations(repr_results=repr_results, output_dir=output_dir)

    assert result.json_path == output_dir / "repr_comparison.json"
    assert result.csv_path == output_dir / "repr_comparison.csv"
    assert result.count == 3

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    rows = {row["repr_key"]: row for row in payload["rows"]}
    assert set(rows) == {"raw_layernorm_mean", "tme_supcon_v1", "raw_layernorm_flat"}

    raw = rows["raw_layernorm_mean"]
    assert raw["status"] == "ok"
    assert raw["conflict_aligned_d_effect_size"] == pytest.approx(4.2426406884)
    assert raw["d_effect_method"] == "pooled_std"
    assert raw["state_error_p_value"] == pytest.approx(0.01)
    assert raw["state_error_statistic"] == pytest.approx(6.0)
    assert raw["state_error_method"] == "chi_square"
    assert raw["error_rate_spread"] == pytest.approx(1.0)
    assert raw["pattern_entropy_bits"] == pytest.approx(1.5)
    assert raw["dominant_pattern_share"] == pytest.approx(0.5)
    assert raw["missing_rate"] == pytest.approx(0.25)
    assert raw["runtime_seconds"] == pytest.approx(7.0)

    tme = rows["tme_supcon_v1"]
    assert tme["status"] == "ok"
    assert tme["conflict_aligned_d_effect_size"] > raw["conflict_aligned_d_effect_size"]

    missing = rows["raw_layernorm_flat"]
    assert missing["status"] == "missing"
    assert "sdr_scores" in missing["missing_reason"]

    with result.csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["repr_key"] for row in csv_rows] == [
        "raw_layernorm_mean",
        "tme_supcon_v1",
        "raw_layernorm_flat",
    ]
    assert csv_rows[1]["repr_key"] == "tme_supcon_v1"
    assert csv_rows[2]["status"] == "missing"


def test_repr_comparison_cli_accepts_config_and_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs/evaluation/main/qwen3_vl_8b/VT"
    config_path = tmp_path / "comparison_config.json"
    _write_json(
        config_path,
        {
            "representations": {
                "raw_layernorm_mean": _repr_paths(tmp_path, "raw_layernorm_mean", d_offset=0.0),
                "tme_supcon_v1": _repr_paths(tmp_path, "tme_supcon_v1", d_offset=0.1),
                "raw_layernorm_flat": {
                    "sdr_scores": str(tmp_path / "missing/sdr_scores.jsonl"),
                    "state_patterns": str(tmp_path / "missing/state_patterns.jsonl"),
                    "state_to_error": str(tmp_path / "missing/state_to_error.json"),
                    "conflict_vs_aligned": str(tmp_path / "missing/conflict_vs_aligned.json"),
                },
            }
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_repr_comparison.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "repr_comparison_json=" in completed.stdout
    assert "repr_comparison_csv=" in completed.stdout
    assert (output_dir / "repr_comparison.json").exists()
    assert (output_dir / "repr_comparison.csv").exists()
