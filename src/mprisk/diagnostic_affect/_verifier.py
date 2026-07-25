"""Diagnostic Affect Description output validation and verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mprisk.models.base_wrapper import GenerationRequest, GenerationResult
from mprisk.utils.io import (
    read_json_object as _read_json,
    read_jsonl as _read_jsonl,
    sha256_file as _sha256,
)

from mprisk.diagnostic_affect._plan import (
    CANONICAL_DIAGNOSTIC_AFFECT_PROMPT,
    OUTPUT_SCHEMA,
    PROVENANCE_SCHEMA,
    SIGNATURE_SCHEMA,
    _SENTENCE_END_RE,
)


def validate_diagnostic_affect_description(result: GenerationResult) -> None:
    """Reject invalid model output; never rewrite, truncate, or replace it."""
    text = result.text
    if result.request.condition != "M12":
        raise ValueError("Diagnostic descriptions require the M12 condition")
    if not text or text != text.strip() or "\n" in text:
        raise ValueError("Generated description must be non-empty")
    endings = _SENTENCE_END_RE.findall(text)
    if len(endings) != 1 or text[-1] not in ".!?":
        raise ValueError("Generated description must contain exactly one sentence")


def verify_diagnostic_affect_descriptions(
    *,
    manifest_path: Path,
    output_root: Path,
    subject_model_key: str,
    run_id: str,
    protocol: str,
    condition: str,
    dataset: str,
    split: str,
    strict_full: bool = True,
) -> dict[str, Any]:
    records = _read_jsonl(output_root / "manifest.jsonl")
    protocol = protocol.upper()
    condition = condition.upper()
    selected_rows = [
        row
        for row in _read_jsonl(manifest_path)
        if str(row.get("source_dataset", "")) == dataset
        and str(row.get("split", "")) == split
        and str(row.get("protocol", "")).upper() == protocol
    ]
    expected = {str(row["sample_id"]): protocol for row in selected_rows}
    observed = {str(row.get("sample_id")): str(row.get("protocol")).upper() for row in records}
    if len(observed) != len(records):
        raise ValueError("Description manifest contains duplicate sample IDs")
    if strict_full and set(observed) != set(expected):
        raise ValueError("Description manifest does not match selected manifest sample IDs")
    if not strict_full and not set(observed).issubset(expected):
        raise ValueError("Description smoke manifest contains unknown sample IDs")
    expected_fields = {
        "schema_name", "run_id", "sample_id", "subject_model_key", "protocol", "condition",
        "dataset", "split",
        "DIAGNOSTIC_AFFECT_DESCRIPTION", "token_ids", "eos_token_ids", "finish_reason",
        "input_token_count", "input_sha256", "media_sha256", "prompt_sha256", "provenance",
    }
    for record in records:
        if set(record) != expected_fields:
            raise ValueError("Description manifest fields are not strict")
        if (
            record.get("schema_name") != OUTPUT_SCHEMA
            or record.get("run_id") != run_id
            or record.get("subject_model_key") != subject_model_key
            or record.get("protocol") != protocol
            or record.get("condition") != condition
            or record.get("dataset") != dataset
            or record.get("split") != split
        ):
            raise ValueError("Description manifest schema or condition mismatch")
        if observed[str(record["sample_id"])] != expected[str(record["sample_id"])]:
            raise ValueError("Description protocol does not match frozen eligibility input")
        request = GenerationRequest(
            sample_id=str(record["sample_id"]),
            model_key=subject_model_key,
            protocol=str(record["protocol"]),
            condition=condition,
            messages=(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CANONICAL_DIAGNOSTIC_AFFECT_PROMPT}
                    ],
                },
            ),
            media_paths={},
            use_audio_in_video=str(record["protocol"]).upper() == "VA",
            generation_kwargs={"do_sample": False, "num_beams": 1, "max_new_tokens": 1},
        )
        validate_diagnostic_affect_description(
            GenerationResult(
                request=request,
                text=str(record["DIAGNOSTIC_AFFECT_DESCRIPTION"]),
                token_ids=record["token_ids"],
                eos_token_ids=record["eos_token_ids"],
                finish_reason=str(record["finish_reason"]),
                input_token_count=int(record["input_token_count"]),
            )
        )
    summary = _read_json(output_root / "summary.json")
    if (
        summary.get("failed") != 0
        or summary.get("pending") != 0
        or summary.get("running") != 0
        or summary.get("completed") != len(records)
    ):
        raise ValueError("Description ledger summary is incomplete or contains failures")
    counts = {protocol: len(records)}
    provenance = _read_json(output_root / "provenance.json")
    if (
        provenance.get("schema_name") != PROVENANCE_SCHEMA
        or provenance.get("run_id") != run_id
    ):
        raise ValueError("Description provenance schema mismatch")
    if provenance.get("canonical_prompt") != CANONICAL_DIAGNOSTIC_AFFECT_PROMPT:
        raise ValueError("Description provenance prompt mismatch")
    signature = provenance.get("signature")
    expected_identity = {
        "schema_name": SIGNATURE_SCHEMA,
        "run_id": run_id,
        "subject_model_key": subject_model_key,
        "protocol": protocol,
        "condition": condition,
        "dataset": dataset,
        "split": split,
    }
    if not isinstance(signature, dict) or any(
        signature.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("Description provenance identity mismatch")
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Description provenance artifacts are missing")
    for name in ("manifest", "failures", "attempts", "summary"):
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError(f"Description artifact metadata is invalid: {name}")
        if artifact.get("sha256") != _sha256(output_root / artifact["path"]):
            raise ValueError(f"Description artifact hash mismatch: {name}")
    return {
        "status": "passed",
        "count": len(records),
        "counts": counts,
        "manifest_sha256": _sha256(output_root / "manifest.jsonl"),
    }
