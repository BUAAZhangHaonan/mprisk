from __future__ import annotations

from curation.scripts.adjudicate_annotations import adjudicate_sample


def test_adjudication_marks_agreement_and_final_type() -> None:
    rows = [
        {
            "sample_id": "s1",
            "sample_type": "Conflict",
            "m1_label": "positive",
            "m2_label": "negative",
            "joint_label": "negative",
            "m1_is_clear": True,
            "m2_is_clear": True,
            "joint_is_clear": True,
            "dominant_modality": "M2",
        },
        {
            "sample_id": "s1",
            "sample_type": "Conflict",
            "m1_label": "positive",
            "m2_label": "negative",
            "joint_label": "negative",
            "m1_is_clear": True,
            "m2_is_clear": True,
            "joint_is_clear": True,
            "dominant_modality": "M2",
        },
    ]

    adjudicated = adjudicate_sample(rows)

    assert adjudicated["sample_type"] == "Conflict"
    assert adjudicated["annotator_agreement"] == 1.0
    assert adjudicated["dominant_modality"] == "M2"
