from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from mprisk.ground_truth.annotation_inputs import (
    GT_ANNOTATION_INPUT_SCHEMA,
    GT_INPUT_SCHEMA_VERSION,
    PILOT_SAMPLE_IDS,
    SCENARIO_CONTEXT_SOURCE_PRIORITY,
    GTAnnotationInput,
    build_gt_annotation_input_pilot,
    resolve_scenario_context,
)

ROOT = Path(__file__).resolve().parents[2]


def test_scenario_context_priority_is_explicit() -> None:
    assert SCENARIO_CONTEXT_SOURCE_PRIORITY == ("setting", "trigger", "source_prompt")
    assert resolve_scenario_context(
        {"setting": " recorded setting ", "trigger": "natural trigger", "ltx2_prompt": "raw"}
    ) == ("recorded setting", "setting")
    assert resolve_scenario_context(
        {"setting": None, "trigger": " natural trigger ", "ltx2_prompt": "raw"}
    ) == ("natural trigger", "trigger")
    assert resolve_scenario_context(
        {"setting": None, "trigger": "T3", "ltx2_prompt": " raw prompt "}
    ) == ("raw prompt", "source_prompt")
    with pytest.raises(ValueError, match="requires"):
        resolve_scenario_context({"setting": None, "trigger": "T2", "ltx2_prompt": None})


def test_pilot_has_new_strict_schema_and_canonical_sample_types() -> None:
    rows, provenance = build_gt_annotation_input_pilot(ROOT)

    assert [row["sample_id"] for row in rows] == list(PILOT_SAMPLE_IDS)
    assert Counter((row["sample_type"], row["protocol"]) for row in rows) == {
        ("Conflict", "VT"): 2,
        ("Conflict", "VA"): 2,
        ("Aligned", "VT"): 2,
        ("Aligned", "VA"): 2,
    }
    assert provenance["scenario_context_source_counts"] == {
        "setting": 126,
        "trigger": 36,
        "source_prompt": 490,
    }
    assert provenance["gt_input_schema_version"] == GT_INPUT_SCHEMA_VERSION
    for row in rows:
        validated = GTAnnotationInput.model_validate(row)
        assert validated.schema_name == GT_ANNOTATION_INPUT_SCHEMA
        assert validated.gt_input_schema_version == GT_INPUT_SCHEMA_VERSION
        assert validated.scenario_context_source == "source_prompt"
        assert validated.source_provenance.source_class_code in {"A", "C"}
        assert "data_type" not in row
        assert "protocol_version" not in row
        assert "context_text" not in row
        assert "context_source" not in row
        assert "ltx2_prompt" not in json.dumps(row, ensure_ascii=False)


def test_annotation_input_rejects_legacy_semantic_fields() -> None:
    rows, _ = build_gt_annotation_input_pilot(ROOT)
    payload = dict(rows[0])
    payload["context_text"] = payload.pop("scenario_context")
    with pytest.raises(ValidationError):
        GTAnnotationInput.model_validate(payload)


def test_media_and_source_assignment_hashes_are_preserved() -> None:
    rows, _ = build_gt_annotation_input_pilot(ROOT)
    for row in rows:
        media_path = Path(row["media"]["path"])
        assert media_path.is_file()
        source = row["source_provenance"]
        assert source["source_assignment"]["source_row_sha256"] == source["source_row_sha256"]


def test_frozen_new_pilot_matches_the_builder_without_legacy_input() -> None:
    expected, _ = build_gt_annotation_input_pilot(ROOT)
    path = (
        ROOT
        / "data/frozen/generated_round1_v1/ground_truth_inputs/"
        "gt_annotation_input_v1/pilot.jsonl"
    )
    actual = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert actual == expected
