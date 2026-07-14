from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from mprisk.data.archetype_canonical_meanings import (
    _build_dictionary_rows,
    normalize_source_description,
)

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data/frozen/generated_round1_v1"
DICTIONARY = FROZEN / "archetype_canonical_meanings_v1.jsonl"
ASSIGNMENTS = FROZEN / "archetype_semantic_assignments_v1.jsonl"
REVIEW_QUEUE = FROZEN / "archetype_canonical_review_queue_v1.jsonl"
PROVENANCE = FROZEN / "archetype_canonical_meanings_v1.provenance.json"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_canonical_dictionary_has_exact_frozen_coverage() -> None:
    dictionary = _read_jsonl(DICTIONARY)
    assignments = _read_jsonl(ASSIGNMENTS)
    review = _read_jsonl(REVIEW_QUEUE)

    assert len(dictionary) == 24
    assert len(assignments) == 652
    assert review == []
    semantic_ids = {row["archetype_semantic_id"] for row in dictionary}
    assert semantic_ids == {
        *(f"A:{value:03d}" for value in range(1, 15)),
        *(f"C:{value:03d}" for value in range(101, 111)),
    }
    assert len({row["sample_id"] for row in assignments}) == 652
    assert {row["archetype_semantic_id"] for row in assignments} == semantic_ids
    assert sum(bool(row["gt_eligible"]) for row in assignments) == 162
    assert Counter(row["data_type"] for row in assignments) == {"A": 423, "C": 229}


def test_per_semantic_snapshot_and_eligible_counts_are_exact() -> None:
    assignments = _read_jsonl(ASSIGNMENTS)
    snapshot = Counter(row["archetype_semantic_id"] for row in assignments)
    eligible = Counter(row["archetype_semantic_id"] for row in assignments if row["gt_eligible"])
    assert snapshot == {
        "A:001": 15,
        "A:002": 26,
        "A:003": 14,
        "A:004": 18,
        "A:005": 41,
        "A:006": 50,
        "A:007": 50,
        "A:008": 21,
        "A:009": 29,
        "A:010": 6,
        "A:011": 30,
        "A:012": 76,
        "A:013": 2,
        "A:014": 45,
        "C:101": 26,
        "C:102": 26,
        "C:103": 42,
        "C:104": 11,
        "C:105": 11,
        "C:106": 7,
        "C:107": 7,
        "C:108": 39,
        "C:109": 27,
        "C:110": 33,
    }
    assert eligible == {
        "A:005": 2,
        "A:006": 6,
        "A:009": 1,
        "A:010": 1,
        "A:011": 15,
        "A:012": 39,
        "A:014": 8,
        "C:101": 9,
        "C:102": 8,
        "C:103": 14,
        "C:105": 11,
        "C:106": 7,
        "C:107": 7,
        "C:108": 15,
        "C:109": 5,
        "C:110": 14,
    }


def test_meanings_are_one_sentence_and_keep_a_c_semantics_separate() -> None:
    dictionary = _read_jsonl(DICTIONARY)
    for row in dictionary:
        meaning = str(row["canonical_meaning"])
        assert len(re.findall(r"[.!?](?=\s|$)", meaning)) == 1
        assert row["status"] == "source_defined"
        assert len(str(row["input_hash"])) == 64
        assert len(str(row["source_sha256"])) == 64
        if row["data_type"] == "A":
            assert row["surface_emotion"] is not None
            assert row["source_kind"] == "ARCHETYPES_GLM.source_defined"
            assert meaning.startswith("Person ")
        else:
            assert row["surface_emotion"] is None
            assert row["source_kind"] == "EMOTION_VARIANTS.exact_tuple_emotion"
            assert meaning == f"The modalities consistently express {row['true_emotion']}."


def test_dictionary_provenance_binds_freeze_and_all_artifacts() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["status"] == "complete"
    assert provenance["counts"] == {
        "assignments": 652,
        "dictionary": 24,
        "dictionary_by_data_type": {"A": 14, "C": 10},
        "gt_eligible_assignments": 162,
        "review_queue": 0,
        "snapshot_by_data_type": {"A": 423, "C": 229},
    }
    assert (
        provenance["freeze_binding"]["sha256"]
        == hashlib.sha256((FROZEN / "provenance.json").read_bytes()).hexdigest()
    )
    for artifact in provenance["artifacts"].values():
        path = ROOT / artifact["path"]
        content = path.read_bytes()
        assert len(content) == artifact["bytes"]
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]


def test_source_description_normalization_is_strict() -> None:
    assert normalize_source_description(" Person hides sadness behind a smile ", max_words=10) == (
        "Person hides sadness behind a smile."
    )
    with pytest.raises(ValueError, match="exactly one sentence"):
        normalize_source_description("Person smiles. Person cries.", max_words=10)
    with pytest.raises(ValueError, match="scene-independent"):
        normalize_source_description("At a desk the person smiles", max_words=10)


def test_invalid_official_description_creates_review_item_without_fallback(
    tmp_path: Path,
) -> None:
    assignments = [
        {
            "sample_id": "sample",
            "data_type": "A",
            "true_emotion": "sadness",
            "source_row_sha256": "a" * 64,
            "gt_eligible": True,
        }
    ]
    rows, review = _build_dictionary_rows(
        grouped={"A:001": assignments},
        eligible_by_id={"sample": {}},
        official_archetypes={
            1: {
                "name": "forced_smile",
                "type": "A",
                "gt": "sadness",
                "surface": "warmth",
                "desc": "not a source-defined Person statement",
            }
        },
        official_c_templates=[],
        archetype_source=(tmp_path / "glm_client.py", "b" * 64),
        c_template_source=(tmp_path / "c_type_batch.py", "c" * 64),
        freeze_provenance_sha="d" * 64,
        dictionary_id="archetype_canonical_meanings_v1",
        max_words=30,
    )
    assert rows == []
    assert len(review) == 1
    assert review[0]["status"] == "needs_review"
    assert "scene-independent" in review[0]["reason"]
