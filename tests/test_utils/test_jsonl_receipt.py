from __future__ import annotations

import json
from pathlib import Path

import pytest

from mprisk.utils.jsonl_receipt import (
    publish_jsonl_receipt,
    read_validated_jsonl,
    receipt_path_for,
    validate_jsonl,
    write_atomic_jsonl,
)


FIELDS = ("sample_id", "model_key", "embeddings")


def _rows() -> list[dict[str, object]]:
    return [
        {"sample_id": "a", "model_key": "model", "embeddings": {"M1": [1.0]}},
        {"sample_id": "b", "model_key": "model", "embeddings": {"M1": [1.0]}},
    ]


def test_atomic_jsonl_receipt_round_trip_and_ascii_bytes(tmp_path: Path) -> None:
    path = write_atomic_jsonl(
        tmp_path / "spherical_embedding_manifest.jsonl",
        [{**_rows()[0], "note": "情绪"}, _rows()[1]],
    )
    receipt = publish_jsonl_receipt(
        path,
        required_fields=FIELDS,
        identity_fields=("sample_id",),
        expected_rows=2,
        bindings={"prompt_sha256": "a" * 64, "checkpoint_sha256": "b" * 64},
    )

    assert receipt == receipt_path_for(path)
    assert all(byte < 128 for byte in path.read_bytes())
    assert read_validated_jsonl(
        path,
        required_fields=FIELDS,
        identity_fields=("sample_id",),
        expected_bindings={"prompt_sha256": "a" * 64, "checkpoint_sha256": "b" * 64},
    ) == [{**_rows()[0], "note": "情绪"}, _rows()[1]]


def test_reader_reports_exact_line_and_byte_for_broken_json(tmp_path: Path) -> None:
    path = tmp_path / "spherical_embedding_manifest.jsonl"
    first = json.dumps(_rows()[0], ensure_ascii=True) + "\n"
    path.write_bytes(first.encode("utf-8") + b'{"sample_id":"b","embeddings":"unterminated}\n')

    with pytest.raises(
        ValueError,
        match=rf"{path.resolve()}.*line=2:byte=\d+",
    ):
        validate_jsonl(
            path,
            required_fields=FIELDS,
            identity_fields=("sample_id",),
        )


def test_receipt_rejects_duplicate_identity_and_wrong_row_count(tmp_path: Path) -> None:
    path = write_atomic_jsonl(
        tmp_path / "spherical_embedding_manifest.jsonl",
        [_rows()[0], _rows()[0]],
    )
    with pytest.raises(ValueError, match="identity fields are not unique"):
        publish_jsonl_receipt(
            path,
            required_fields=FIELDS,
            identity_fields=("sample_id",),
            expected_rows=2,
            bindings={"checkpoint_sha256": "b" * 64},
        )
    path = write_atomic_jsonl(path, _rows())
    with pytest.raises(ValueError, match="row count mismatch"):
        publish_jsonl_receipt(
            path,
            required_fields=FIELDS,
            identity_fields=("sample_id",),
            expected_rows=3,
            bindings={"checkpoint_sha256": "b" * 64},
        )


def test_resume_reader_rejects_mutated_artifact_or_binding(tmp_path: Path) -> None:
    path = write_atomic_jsonl(tmp_path / "spherical_embedding_manifest.jsonl", _rows())
    publish_jsonl_receipt(
        path,
        required_fields=FIELDS,
        identity_fields=("sample_id",),
        expected_rows=2,
        bindings={"checkpoint_sha256": "b" * 64},
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        read_validated_jsonl(
            path,
            required_fields=FIELDS,
            identity_fields=("sample_id",),
            expected_bindings={"checkpoint_sha256": "c" * 64},
        )
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(_rows()[0]) + "\n")
    with pytest.raises(ValueError, match="row count mismatch"):
        read_validated_jsonl(
            path,
            required_fields=FIELDS,
            identity_fields=("sample_id",),
        )
