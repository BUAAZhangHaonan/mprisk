from __future__ import annotations

from curation.scripts.initial_screen_ch_sims_v2 import classify_relation, make_candidate


def test_classify_relation_conflict_aligned_ambiguous() -> None:
    assert classify_relation(0.8, -0.7, -0.6, clear_abs_threshold=0.4, conflict_gap_threshold=0.8) == "Conflict"
    assert classify_relation(0.7, 0.8, 0.6, clear_abs_threshold=0.4, conflict_gap_threshold=0.8) == "Aligned"
    assert classify_relation(0.1, 0.9, 0.1, clear_abs_threshold=0.4, conflict_gap_threshold=0.8) == "Ambiguous"


def test_candidate_record_contains_relation_fields() -> None:
    candidate = make_candidate(
        sample_id="s1",
        source_dataset="ch_sims_v2",
        source_id="clip-1",
        protocol="VT",
        m1_modality="vision",
        m2_modality="text",
        m1_raw=0.8,
        m2_raw=-0.7,
        joint_raw=-0.6,
        media_paths={"vision": "v.mp4", "text": "hello"},
    )

    assert candidate["candidate_type"] == "Conflict"
    assert candidate["m1_label"] == "positive"
    assert candidate["m2_label"] == "negative"
    assert candidate["joint_label"] == "negative"
    assert candidate["needs_llm_screening"] is True
