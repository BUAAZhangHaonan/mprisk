from __future__ import annotations

from typing import Any

import pytest

from curation.scripts.export_final_manifests import build_manifest_row


def _valid_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "sample_id": "s1",
        "source_dataset": "ch_sims_v2",
        "source_id": "clip-1",
        "protocol": "VT",
        "sample_type": "Conflict",
        "m1_modality": "vision",
        "m2_modality": "text",
        "m1_label": "positive",
        "m2_label": "negative",
        "joint_label": "negative",
        "m1_specific_affect": "smile",
        "m2_specific_affect": "complaint",
        "joint_specific_affect": "sarcasm",
        "m1_is_clear": True,
        "m2_is_clear": True,
        "joint_is_clear": True,
        "dominant_modality": "M2",
        "annotator_agreement": 0.86,
        "annotation_count": 2,
        "quality_flags": [],
        "source_is_generated": False,
    }
    row.update(overrides)
    return row


def test_manifest_export_preserves_three_views_and_split_group() -> None:
    row = build_manifest_row(_valid_row())

    assert row["split_group_id"] == "ch_sims_v2:clip-1"
    assert row["views"]["M1"]["modality"] == "vision"
    assert row["views"]["M2"]["label"] == "negative"
    assert row["views"]["M12"]["is_clear"] is True
    assert row["annotation_count"] == 2


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"annotation_count": 1}, "single annotation"),
        ({"annotator_agreement": 0.66}, "low agreement"),
        ({"m1_is_clear": False}, "unclear m1"),
        ({"m2_is_clear": False}, "unclear m2"),
        ({"joint_is_clear": False}, "unclear joint"),
        ({"quality_flags": ["missing_text"]}, "blocking quality flag"),
        ({"sample_type": "Ambiguous"}, "non-main sample type"),
        (
            {"sample_type": "Conflict", "m1_label": "positive", "m2_label": "positive"},
            "bad conflict labels",
        ),
        (
            {
                "sample_type": "Aligned",
                "m1_label": "positive",
                "m2_label": "positive",
                "joint_label": "negative",
            },
            "bad aligned labels",
        ),
    ],
)
def test_manifest_use_in_main_rejects_rows_below_strict_thresholds(
    overrides: dict[str, Any], reason: str
) -> None:
    row = build_manifest_row(_valid_row(**overrides))

    assert row["use_in_main"] is False, reason


@pytest.mark.parametrize(
    "source_row",
    [
        _valid_row(
            sample_type="Conflict",
            m1_label="positive",
            m2_label="negative",
            joint_label="negative",
        ),
        _valid_row(
            sample_type="Aligned",
            m1_label="positive",
            m2_label="positive",
            joint_label="positive",
        ),
    ],
)
def test_manifest_use_in_main_accepts_qualified_conflict_and_aligned(
    source_row: dict[str, Any],
) -> None:
    row = build_manifest_row(source_row)

    assert row["use_in_main"] is True
