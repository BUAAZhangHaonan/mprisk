from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mprisk.utils.io import write_json, write_jsonl
from mprisk.viz.bundle_figures import export_bundle_figures
from mprisk.viz.figure_inputs import build_state_figure_inputs


def _scores() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "a-stable-consensus",
            "sample_type": "Aligned",
            "model_key": "qwen3_vl_8b",
            "S_mean": 0.1,
            "D": 0.2,
            "R": 0.1,
        },
        {
            "sample_id": "c-stable-directional",
            "sample_type": "Conflict",
            "model_key": "qwen3_vl_8b",
            "S_mean": 0.2,
            "D": 0.8,
            "R": -0.4,
        },
        {
            "sample_id": "c-unstable",
            "sample_type": "Conflict",
            "model_key": "qwen3_vl_8b",
            "S_mean": 0.9,
            "D": 0.7,
            "R": 0.7,
        },
    ]


def _patterns() -> list[dict[str, object]]:
    patterns = ["Consensus", "Dominant", "Confusion"]
    return [dict(row, pattern=pattern) for row, pattern in zip(_scores(), patterns, strict=True)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_state_figure_inputs_record_hashes_commands_and_exact_masks(tmp_path) -> None:
    scores_path = write_jsonl(tmp_path / "sdr.jsonl", _scores())
    patterns_path = write_jsonl(tmp_path / "patterns.jsonl", _patterns())
    thresholds_path = write_json(
        tmp_path / "thresholds.json",
        {"schema": "mprisk_spherical_calibration_v1", "kappa": 0.5, "tau": 0.3},
    )

    result = build_state_figure_inputs(
        sdr_scores_path=scores_path,
        state_patterns_path=patterns_path,
        thresholds_path=thresholds_path,
        output_dir=tmp_path / "inputs",
        generated_command=["python", "scripts/build_figure_inputs.py"],
    )

    fig4 = _read_csv(result.fig04_path)
    fig5 = _read_csv(result.fig05_path)
    fig6 = _read_csv(result.fig06_path)
    fig4_provenance = json.loads(result.fig04_provenance_path.read_text())
    assert [row["metric"] for row in fig4].count("S") == 3
    assert [row["metric"] for row in fig4].count("D") == 2
    assert [row["metric"] for row in fig4].count("abs_R") == 1
    assert sum(int(row["count"]) for row in fig5) == 3
    assert {row["sample_id"] for row in fig6} == {
        "a-stable-consensus",
        "c-stable-directional",
    }
    assert {row["direction_emphasized"] for row in fig6} == {"false", "true"}
    assert fig4_provenance["generated_command"] == [
        "python",
        "scripts/build_figure_inputs.py",
    ]
    assert len(fig4_provenance["sources"]) == 2
    assert all(len(source["sha256"]) == 64 for source in fig4_provenance["sources"])
    assert fig4_provenance["sample_masks"] == {
        "S": "all_samples",
        "D": "S<=kappa",
        "abs_R": "S<=kappa and D>tau",
    }


def test_fig6_rejects_rows_that_violate_stable_or_direction_mask(tmp_path) -> None:
    scores_path = write_jsonl(tmp_path / "sdr.jsonl", _scores())
    patterns_path = write_jsonl(tmp_path / "patterns.jsonl", _patterns())
    thresholds_path = write_json(tmp_path / "thresholds.json", {"kappa": 0.5, "tau": 0.3})
    result = build_state_figure_inputs(
        sdr_scores_path=scores_path,
        state_patterns_path=patterns_path,
        thresholds_path=thresholds_path,
        output_dir=tmp_path / "inputs",
        generated_command=["pytest"],
    )
    rows = _read_csv(result.fig06_path)
    rows[0]["S"] = "0.8"
    with result.fig06_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "schema": "mprisk_bundle_figure_map_v1",
        "figures": {
            **{
                f"fig{index:02d}_pending": {
                    "title": f"pending {index}",
                    "input": str(tmp_path / f"missing-{index}.json"),
                    "output": str(tmp_path / f"pending-{index}.pdf"),
                }
                for index in range(1, 11)
                if index != 6
            },
            "fig06_stable_d_signed_r": {
                "title": "stable",
                "input": str(result.fig06_path),
                "output": str(tmp_path / "fig06.pdf"),
            },
        },
    }
    config_path = tmp_path / "figures.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Fig. 6 stable mask"):
        export_bundle_figures(config_path)
