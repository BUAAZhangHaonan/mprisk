from __future__ import annotations

from curation.scripts.run_llm_screening import MockProvider, screen_candidate


def test_mock_llm_screening_outputs_three_views() -> None:
    candidate = {
        "sample_id": "s1",
        "protocol": "VT",
        "candidate_type": "Conflict",
        "m1_label": "positive",
        "m2_label": "negative",
        "joint_label": "negative",
    }

    output = screen_candidate(candidate, MockProvider())

    assert set(output["view_outputs"]) == {"M1", "M2", "M12"}
    assert output["sample_type_suggestion"] == "Conflict"
    assert output["dominant_modality_suggestion"] in {"M1", "M2", "balanced", "unclear"}
    assert output["needs_human_review"] is True
