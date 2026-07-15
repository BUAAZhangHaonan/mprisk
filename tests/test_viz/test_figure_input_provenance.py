from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mprisk.utils.io import write_json, write_jsonl
from mprisk.viz.bundle_figures import _validate_fig06_masks
from mprisk.viz.figure_inputs import build_state_figure_inputs


def _scores() -> list[dict[str, object]]:
    identity = {
        "protocol": "vt",
        "prompt_set_key": "p8-seed",
        "prompt_set_artifact_sha256": "1" * 64,
        "repr_key": "tme",
        "encoder_checkpoint_sha256": "2" * 64,
        "split_assignment_sha256": "3" * 64,
        "embedding_manifest_sha256": "4" * 64,
    }
    return [
        {
            "sample_id": "a-stable-consensus",
            "sample_type": "Aligned",
            "model_key": "qwen3_vl_8b",
            "representation_split": "official_test",
            **identity,
            "sdr_schema": "mprisk_spherical_sdr_v2",
            "distance_metric": "geodesic_acos_v1",
            "S_mean": 0.1,
            "D": 0.2,
            "R": 0.1,
            "delta_i": 0.1,
        },
        {
            "sample_id": "c-stable-directional",
            "sample_type": "Conflict",
            "model_key": "qwen3_vl_8b",
            "representation_split": "official_test",
            **identity,
            "sdr_schema": "mprisk_spherical_sdr_v2",
            "distance_metric": "geodesic_acos_v1",
            "S_mean": 0.2,
            "D": 0.8,
            "R": -0.4,
            "delta_i": 0.1,
        },
        {
            "sample_id": "c-unstable",
            "sample_type": "Conflict",
            "model_key": "qwen3_vl_8b",
            "representation_split": "official_test",
            **identity,
            "sdr_schema": "mprisk_spherical_sdr_v2",
            "distance_metric": "geodesic_acos_v1",
            "S_mean": 0.9,
            "D": 0.7,
            "R": 0.7,
            "delta_i": 0.1,
        },
    ]


def _patterns() -> list[dict[str, object]]:
    patterns = ["Consensus", "Dominant", "Confusion"]
    return [dict(row, pattern=pattern) for row, pattern in zip(_scores(), patterns, strict=True)]


def _thresholds() -> dict[str, object]:
    return {
        "schema": "mprisk_spherical_calibration_v2",
        "sdr_schema": "mprisk_spherical_sdr_v2",
        "distance_metric": "geodesic_acos_v1",
        "sample_type": "Aligned",
        "calibration_split": "aligned_calibration",
        "selection_rule": "representation_split=aligned_calibration then sample_type=Aligned",
        "model_key": "qwen3_vl_8b",
        "protocol": "vt",
        "prompt_set_key": "p8-seed",
        "prompt_set_artifact_sha256": "1" * 64,
        "repr_key": "tme",
        "encoder_checkpoint_sha256": "2" * 64,
        "split_assignment_sha256": "3" * 64,
        "embedding_manifest_sha256": "4" * 64,
        "kappa": 0.5,
        "tau": 0.3,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_state_figure_inputs_record_hashes_commands_and_exact_masks(tmp_path) -> None:
    scores_path = write_jsonl(tmp_path / "sdr.jsonl", _scores())
    patterns_path = write_jsonl(tmp_path / "patterns.jsonl", _patterns())
    thresholds_path = write_json(
        tmp_path / "thresholds.json",
        {
            "schema": "mprisk_spherical_calibration_v2",
            "sdr_schema": "mprisk_spherical_sdr_v2",
            "distance_metric": "geodesic_acos_v1",
            "sample_type": "Aligned",
            "calibration_split": "aligned_calibration",
            "selection_rule": "representation_split=aligned_calibration then sample_type=Aligned",
            "model_key": "qwen3_vl_8b",
            "protocol": "vt",
            "prompt_set_key": "p8-seed",
            "prompt_set_artifact_sha256": "1" * 64,
            "repr_key": "tme",
            "encoder_checkpoint_sha256": "2" * 64,
            "split_assignment_sha256": "3" * 64,
            "embedding_manifest_sha256": "4" * 64,
            "kappa": 0.5,
            "tau": 0.3,
        },
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
        "S": "representation_split=official_test",
        "D": "representation_split=official_test and S<=kappa",
        "abs_R": "representation_split=official_test and S<=kappa and D>tau",
    }
    assert fig4_provenance["sdr_schema"] == "mprisk_spherical_sdr_v2"
    assert fig4_provenance["distance_metric"] == "geodesic_acos_v1"
    assert fig4_provenance["representation_split"] == "official_test"
    assert fig4_provenance["source_representation_split_counts"] == {"official_test": 3}
    assert fig4_provenance["official_test_sample_count"] == 3
    assert fig4_provenance["excluded_non_official_test_count"] == 0


def test_fig6_rejects_rows_that_violate_stable_or_direction_mask(tmp_path) -> None:
    scores_path = write_jsonl(tmp_path / "sdr.jsonl", _scores())
    patterns_path = write_jsonl(tmp_path / "patterns.jsonl", _patterns())
    thresholds_path = write_json(
        tmp_path / "thresholds.json",
        {
            "schema": "mprisk_spherical_calibration_v2",
            "sdr_schema": "mprisk_spherical_sdr_v2",
            "distance_metric": "geodesic_acos_v1",
            "sample_type": "Aligned",
            "calibration_split": "aligned_calibration",
            "selection_rule": "representation_split=aligned_calibration then sample_type=Aligned",
            "model_key": "qwen3_vl_8b",
            "protocol": "vt",
            "prompt_set_key": "p8-seed",
            "prompt_set_artifact_sha256": "1" * 64,
            "repr_key": "tme",
            "encoder_checkpoint_sha256": "2" * 64,
            "split_assignment_sha256": "3" * 64,
            "embedding_manifest_sha256": "4" * 64,
            "kappa": 0.5,
            "tau": 0.3,
        },
    )
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

    with pytest.raises(ValueError, match="Fig. 6 stable mask"):
        _validate_fig06_masks(rows, json.loads(result.fig06_provenance_path.read_text()))


def test_state_figure_inputs_exclude_non_official_rows(tmp_path) -> None:
    scores = _scores()
    train_row = dict(scores[0], sample_id="train-a", representation_split="relation_train")
    patterns = _patterns()
    patterns.append(dict(train_row, pattern="Consensus"))
    scores.append(train_row)
    scores_path = write_jsonl(tmp_path / "sdr.jsonl", scores)
    patterns_path = write_jsonl(tmp_path / "patterns.jsonl", patterns)
    thresholds_path = write_json(
        tmp_path / "thresholds.json",
        {
            "schema": "mprisk_spherical_calibration_v2",
            "sdr_schema": "mprisk_spherical_sdr_v2",
            "distance_metric": "geodesic_acos_v1",
            "sample_type": "Aligned",
            "calibration_split": "aligned_calibration",
            "selection_rule": "representation_split=aligned_calibration then sample_type=Aligned",
            "model_key": "qwen3_vl_8b",
            "protocol": "vt",
            "prompt_set_key": "p8-seed",
            "prompt_set_artifact_sha256": "1" * 64,
            "repr_key": "tme",
            "encoder_checkpoint_sha256": "2" * 64,
            "split_assignment_sha256": "3" * 64,
            "embedding_manifest_sha256": "4" * 64,
            "kappa": 0.5,
            "tau": 0.3,
        },
    )
    result = build_state_figure_inputs(
        sdr_scores_path=scores_path,
        state_patterns_path=patterns_path,
        thresholds_path=thresholds_path,
        output_dir=tmp_path / "inputs",
        generated_command=["pytest"],
    )
    provenance = json.loads(result.fig04_provenance_path.read_text())
    assert provenance["source_sample_count"] == 4
    assert provenance["official_test_sample_count"] == 3
    assert provenance["excluded_non_official_test_count"] == 1
    assert {row["sample_id"] for row in _read_csv(result.fig04_path)} <= {
        "a-stable-consensus",
        "c-stable-directional",
        "c-unstable",
    }


def test_state_figure_inputs_reject_tampered_pattern_assignment(tmp_path) -> None:
    patterns = _patterns()
    patterns[1]["pattern"] = "Balanced"
    with pytest.raises(ValueError, match="hierarchical S/D/R assignment"):
        build_state_figure_inputs(
            sdr_scores_path=write_jsonl(tmp_path / "sdr.jsonl", _scores()),
            state_patterns_path=write_jsonl(tmp_path / "patterns.jsonl", patterns),
            thresholds_path=write_json(tmp_path / "thresholds.json", _thresholds()),
            output_dir=tmp_path / "inputs",
            generated_command=["pytest"],
        )
