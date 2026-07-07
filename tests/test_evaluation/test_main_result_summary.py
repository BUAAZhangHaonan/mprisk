from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mprisk.evaluation.reporting import build_main_result_summary


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_main_result_summary_writes_all_sections_from_present_inputs(tmp_path: Path) -> None:
    conflict_path = _write_json(
        tmp_path / "conflict_vs_aligned.json",
        {
            "groups": {
                "Conflict": {
                    "n": 3,
                    "metrics": {
                        "S_mean": {"mean": 0.2},
                        "D": {"mean": 0.8},
                        "abs_R": {"mean": 0.6},
                    },
                    "pattern_counts": {"lowS-highD": 2, "mixed": 1},
                },
                "Aligned": {
                    "n": 2,
                    "metrics": {
                        "S_mean": {"mean": 0.7},
                        "D": {"mean": 0.3},
                        "abs_R": {"mean": 0.2},
                    },
                    "pattern_counts": {"highS-lowD": 2},
                },
            },
            "p_values": {"S_mean": {"p_value": 0.04}},
        },
    )
    state_path = _write_json(
        tmp_path / "state_to_error.json",
        {
            "metadata": {"merged_count": 5},
            "overall": {
                "lowS-highD": {"n": 3, "error_rate": 0.667, "abstain_rate": 0.333},
                "highS-lowD": {"n": 2, "error_rate": 0.0, "abstain_rate": 0.0},
            },
        },
    )
    repr_path = _write_json(
        tmp_path / "repr_comparison.json",
        {
            "representations": {
                "raw_layernorm_mean": {"n": 5, "metrics": {"error_auc": 0.61}},
                "tme_supcon_v1": {"n": 5, "metrics": {"error_auc": 0.78}},
            }
        },
    )
    efficiency_path = _write_json(
        tmp_path / "efficiency_comparison.json",
        {
            "methods": {
                "t0": {"n": 5, "mean_seconds": 0.8, "mean_cost": 0.01},
                "posthoc": {"n": 5, "mean_seconds": 2.4, "mean_cost": 0.03},
            }
        },
    )

    result = build_main_result_summary(
        conflict_vs_aligned_path=conflict_path,
        state_to_error_path=state_path,
        repr_comparison_path=repr_path,
        efficiency_comparison_path=efficiency_path,
        output_dir=tmp_path / "out",
    )

    assert result.summary_path == tmp_path / "out" / "MAIN_RESULT_SUMMARY.md"
    text = result.summary_path.read_text(encoding="utf-8")
    assert "# Main Result Summary" in text
    assert "## Data Scale" in text
    assert "- Conflict vs Aligned: 5 samples" in text
    assert "## Conflict vs Aligned Main Differences" in text
    assert "S_mean: Conflict 0.2000 vs Aligned 0.7000" in text
    assert "p=0.0400" in text
    assert "## Four-State/Error-Rate Relation" in text
    assert "lowS-highD: n=3, error_rate=0.6670" in text
    assert "## Raw vs TME Comparison" in text
    assert "raw_layernorm_mean" in text
    assert "tme_supcon_v1" in text
    assert "## T0 vs Posthoc Efficiency" in text
    assert "mean_seconds=0.8000" in text
    assert "## Missing Result Reminders" in text
    assert "- None." in text


def test_build_main_result_summary_lists_missing_optional_inputs(tmp_path: Path) -> None:
    result = build_main_result_summary(output_dir=tmp_path / "out")

    text = result.summary_path.read_text(encoding="utf-8")
    assert result.missing_inputs == [
        "conflict_vs_aligned.json",
        "state_to_error.json",
        "repr_comparison.json",
        "efficiency_comparison.json",
    ]
    assert "Conflict vs Aligned input missing: conflict_vs_aligned.json" in text
    assert "State-to-error input missing: state_to_error.json" in text
    assert "Raw vs TME input missing: repr_comparison.json" in text
    assert "Efficiency input missing: efficiency_comparison.json" in text
    assert "- conflict_vs_aligned.json" in text
    assert "- state_to_error.json" in text
    assert "- repr_comparison.json" in text
    assert "- efficiency_comparison.json" in text


def test_build_main_result_summary_cli_accepts_optional_inputs(tmp_path: Path) -> None:
    conflict_path = _write_json(
        tmp_path / "conflict_vs_aligned.json",
        {
            "groups": {
                "Conflict": {"n": 1, "metrics": {"S_mean": {"mean": 0.1}}},
                "Aligned": {"n": 1, "metrics": {"S_mean": {"mean": 0.9}}},
            }
        },
    )
    output_dir = tmp_path / "cli_out"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_main_result_summary.py",
            "--conflict-vs-aligned",
            str(conflict_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "main_result_summary=" in completed.stdout
    text = (output_dir / "MAIN_RESULT_SUMMARY.md").read_text(encoding="utf-8")
    assert "Conflict vs Aligned: 2 samples" in text
    assert "- state_to_error.json" in text
