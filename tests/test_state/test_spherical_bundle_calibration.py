from __future__ import annotations

import json

import pytest

from mprisk.state.patterns import StatePattern, StateThresholds, assign_state
from mprisk.state.spherical import compute_spherical_state
from mprisk.state.thresholds import (
    calibrate_aligned_thresholds,
    calibrate_registered_aligned_thresholds,
)
from mprisk.utils.io import write_jsonl
from scripts.compute_sdr_scores import compute_sdr_scores


def _identity() -> dict[str, str]:
    return {
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "prompt_set_key": "p8",
        "prompt_set_artifact_sha256": "c" * 64,
        "repr_key": "tme_proxy_anchor_v1",
        "encoder_checkpoint_sha256": "d" * 64,
        "split_assignment_sha256": "a" * 64,
        "embedding_manifest_sha256": "e" * 64,
    }


def _bundle(sample_id: str, *, sample_type: str = "Aligned") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "calibration_split": "aligned_calibration",
        "embeddings": {
            "M1": {"p1": [1.0, 0.0], "p2": [1.0, 0.0]},
            "M2": {"p1": [0.0, 1.0], "p2": [0.0, 1.0]},
            "M12": {
                "p1": [0.9797958971132712, 0.2],
                "p2": [0.9, 0.4358898943540673],
            },
        },
    }


def test_spherical_bundle_uses_signed_r_and_synchronous_prompt_delta() -> None:
    state = compute_spherical_state(_bundle("a1"))
    assert state["R"] > 0.0, "positive R must mean the joint condition leans vision/M1"
    assert state["lean"] == "V"
    assert state["delta_i"] == pytest.approx(1.96 * state["R_bootstrap_se"])
    assert state["delta_method"] == "synchronous_prompt_bootstrap_1.96se_v1"
    assert state["bootstrap_replicates"] == 2000
    assert state["prompt_ids"] == ["p1", "p2"]


def test_spherical_bundle_rejects_nonunit_or_unsynchronized_embeddings() -> None:
    nonunit = _bundle("a1")
    nonunit["embeddings"]["M1"]["p1"] = [2.0, 0.0]
    with pytest.raises(ValueError, match="unit hypersphere"):
        compute_spherical_state(nonunit)

    unsynchronized = _bundle("a2")
    del unsynchronized["embeddings"]["M12"]["p2"]
    with pytest.raises(ValueError, match="synchronized prompt IDs"):
        compute_spherical_state(unsynchronized)


def test_aligned_only_calibration_uses_q95_and_stable_aligned_for_tau() -> None:
    rows = [_bundle(f"a{index}") for index in range(1, 6)]
    states = [compute_spherical_state(row) for row in rows]
    calibration = calibrate_aligned_thresholds(states, quantile_level=0.95)
    assert calibration["schema"] == "mprisk_spherical_calibration_v2"
    assert calibration["sdr_schema"] == "mprisk_spherical_sdr_v2"
    assert calibration["distance_metric"] == "geodesic_acos_v1"
    assert calibration["sample_type"] == "Aligned"
    assert calibration["kappa"] == max(row["S_mean"] for row in states)
    assert calibration["tau"] == max(row["D"] for row in states)
    assert calibration["stable_aligned_count"] == 5

    conflict = dict(states[0], sample_type="Conflict")
    with pytest.raises(ValueError, match="Aligned calibration"):
        calibrate_aligned_thresholds([conflict])

    stale = dict(states[0], sdr_schema="mprisk_spherical_sdr_v1")
    with pytest.raises(ValueError, match="exact spherical SDR v2"):
        calibrate_aligned_thresholds([stale])


def test_pattern_hierarchy_uses_sample_delta_after_confusion_and_consensus() -> None:
    thresholds = StateThresholds(kappa=0.5, tau=0.2)
    assert assign_state(0.6, 1.0, 0.0, thresholds, delta_i=0.5) is StatePattern.CONFUSION
    assert assign_state(0.1, 0.2, 0.9, thresholds, delta_i=0.1) is StatePattern.CONSENSUS
    assert assign_state(0.1, 0.8, 0.1, thresholds, delta_i=0.1) is StatePattern.BALANCED
    assert assign_state(0.1, 0.8, -0.2, thresholds, delta_i=0.1) is StatePattern.DOMINANT


def test_calibration_filters_registered_split_before_aligned_label(tmp_path) -> None:
    calibration = compute_spherical_state(_bundle("calibration"))
    calibration.update(
        representation_split="aligned_calibration",
        **_identity(),
    )
    relation_val = dict(
        compute_spherical_state(_bundle("relation-val")),
        representation_split="relation_val",
        **_identity(),
    )
    official_test = dict(
        compute_spherical_state(_bundle("official-test")),
        representation_split="official_test",
        **_identity(),
    )

    result = calibrate_registered_aligned_thresholds(
        [relation_val, calibration, official_test]
    )

    assert result["aligned_count"] == 1
    assert result["registered_calibration_count"] == 1
    assert result["input_count"] == 3
    assert result["split_assignment_sha256"] == "a" * 64


def test_sdr_score_export_preserves_registered_split_for_calibration(tmp_path) -> None:
    bundle = _bundle("calibration-export")
    bundle.update(
        model_key="qwen3_vl_8b",
        protocol="VT",
        prompt_set_key="p8",
        repr_key="tme_proxy_anchor_v1",
        representation_split="aligned_calibration",
        split_group_id="group-calibration",
        split_assignment_sha256="b" * 64,
        prompt_set_artifact_sha256="c" * 64,
        encoder_checkpoint_sha256="d" * 64,
    )
    source = write_jsonl(tmp_path / "embeddings.jsonl", [bundle])

    result = compute_sdr_scores(embedding_manifest_path=source, output_dir=tmp_path / "scores")
    row = json.loads(result.scores_path.read_text().strip())

    assert row["representation_split"] == "aligned_calibration"
    assert row["split_group_id"] == "group-calibration"
    assert row["split_assignment_sha256"] == "b" * 64
