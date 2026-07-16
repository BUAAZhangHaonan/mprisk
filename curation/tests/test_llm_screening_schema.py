from __future__ import annotations

import json
from typing import Any

from curation.scripts.run_llm_screening import (
    MockProvider,
    build_prompt,
    build_view_payload,
    screen_candidate,
)


def _serialized(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _candidate_with_leaky_fields() -> dict[str, Any]:
    return {
        "sample_id": "s1",
        "source_dataset": "source-a",
        "source_id": "clip-1",
        "protocol": "VT",
        "candidate_type": "LEAK_CANDIDATE_TYPE",
        "planned_sample_type": "LEAK_PLANNED_TYPE",
        "planned_m1_label": "LEAK_PLANNED_M1",
        "planned_m2_label": "LEAK_PLANNED_M2",
        "planned_joint_label": "LEAK_PLANNED_JOINT",
        "m1_modality": "vision",
        "m2_modality": "text",
        "m1_label": "LEAK_M1_LABEL",
        "m2_label": "LEAK_M2_LABEL",
        "joint_label": "LEAK_JOINT_LABEL",
        "m1_raw_label": "LEAK_M1_RAW",
        "m2_raw_label": "LEAK_M2_RAW",
        "joint_raw_label": "LEAK_JOINT_RAW",
        "raw_joint_suggestions": ["LEAK_RAW_JOINT_SUGGESTION"],
        "joint_suggestions": ["LEAK_JOINT_SUGGESTION"],
        "media_paths": {
            "vision": "media/m1-only.mp4",
            "text": "media/m2-only.txt",
            "audio": "media/unselected-audio.wav",
        },
        "source_is_generated": True,
        "quality_metadata": {"fps": 25},
    }


def _assert_no_label_leaks(serialized: str) -> None:
    for leaked in (
        "candidate_type",
        "planned_sample_type",
        "planned_m1_label",
        "planned_m2_label",
        "planned_joint_label",
        "m1_label",
        "m2_label",
        "joint_label",
        "m1_raw_label",
        "m2_raw_label",
        "joint_raw_label",
        "raw_joint_suggestions",
        "joint_suggestions",
        "LEAK_",
    ):
        assert leaked not in serialized


def test_build_view_payload_m1_contains_only_m1_media_and_metadata() -> None:
    payload = build_view_payload(_candidate_with_leaky_fields(), "M1")
    serialized = _serialized(payload)

    assert payload["view"] == "M1"
    assert payload["sample_id"] == "s1"
    assert payload["m1_modality"] == "vision"
    assert "media/m1-only.mp4" in serialized
    assert "media/m2-only.txt" not in serialized
    assert "media/unselected-audio.wav" not in serialized
    _assert_no_label_leaks(serialized)


def test_build_view_payload_m2_contains_only_m2_media_and_metadata() -> None:
    payload = build_view_payload(_candidate_with_leaky_fields(), "M2")
    serialized = _serialized(payload)

    assert payload["view"] == "M2"
    assert payload["m2_modality"] == "text"
    assert "media/m2-only.txt" in serialized
    assert "media/m1-only.mp4" not in serialized
    assert "media/unselected-audio.wav" not in serialized
    _assert_no_label_leaks(serialized)


def test_build_view_payload_m12_contains_pair_media_without_planned_or_raw_labels() -> None:
    payload = build_view_payload(_candidate_with_leaky_fields(), "M12")
    serialized = _serialized(payload)

    assert payload["view"] == "M12"
    assert payload["m1_modality"] == "vision"
    assert payload["m2_modality"] == "text"
    assert "media/m1-only.mp4" in serialized
    assert "media/m2-only.txt" in serialized
    assert "media/unselected-audio.wav" not in serialized
    _assert_no_label_leaks(serialized)


def test_build_prompt_uses_view_payload_instead_of_full_candidate() -> None:
    candidate = _candidate_with_leaky_fields()
    prompt = build_prompt(candidate, "M1")
    prompt_payload = json.loads(prompt)
    serialized = _serialized(prompt_payload)

    assert "candidate" not in prompt_payload
    assert prompt_payload["view_payload"] == build_view_payload(candidate, "M1")
    assert "media/m1-only.mp4" in serialized
    assert "media/m2-only.txt" not in serialized
    _assert_no_label_leaks(serialized)


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
