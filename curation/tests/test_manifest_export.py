from __future__ import annotations

from curation.scripts.export_final_manifests import build_manifest_row


def test_manifest_export_preserves_three_views_and_split_group() -> None:
    row = build_manifest_row(
        {
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
            "source_is_generated": False,
        }
    )

    assert row["split_group_id"] == "ch_sims_v2:clip-1"
    assert row["views"]["M1"]["modality"] == "vision"
    assert row["views"]["M2"]["label"] == "negative"
    assert row["views"]["M12"]["is_clear"] is True
