#!/usr/bin/env python3
"""Prepare and run the extended Conflict-Misread final-direction audit.

Preparation, validation, and self-test modes are offline. The ``run`` command
performs one OpenAI-compatible DeepSeek pass over the frozen manifest. Its
append-only ledger treats every recorded audit_id, including started and failed
requests, as consumed so an interrupted run never retries silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT.parent / "mprisk-v2" / "outputs" / "v2" / "misread"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "analysis" / "misread_final_direction_extended_v1"
PROMPT_PATH = Path(__file__).with_name("misread_final_direction_extended_prompt.txt")
MANIFEST_PATH = OUTPUT_ROOT / "manifest.jsonl"
DRY_RUN_PATH = OUTPUT_ROOT / "dry_run_requests.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
RESULTS_PATH = OUTPUT_ROOT / "results.jsonl"
RUN_SUMMARY_PATH = OUTPUT_ROOT / "run_summary.json"

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_MODEL = "deepseek-v4-pro"
API_KEY_ENV = "DEEPSEEK_API_KEY"
REQUEST_TIMEOUT_SECONDS = 90.0
MAX_CONCURRENCY = 1
MAX_TOKENS = 128

SCHEMA_MANIFEST = "mprisk_misread_final_direction_extended_manifest_v1"
SCHEMA_DRY_RUN = "mprisk_misread_final_direction_extended_dry_run_v1"
SCHEMA_ANNOTATION = "mprisk_misread_final_direction_extended_annotation_v1"
SCHEMA_RESULT = "mprisk_misread_final_direction_extended_result_v1"
EXPECTED_ANNOTATION_KEYS = ("chosen_modality", "reason")

SOURCE_RULE = re.compile(
    r"\b(?:(?:the\s+)?overall|this)\s+judgment\s+follows\s+the\s+"
    r"(video|text|audio)\s+modality\b",
    re.IGNORECASE,
)
SOURCE_WORD_TO_MODALITY = {
    "video": "V",
    "text": "T",
    "audio": "A",
}
PROTOCOL_ALLOWED = {
    "VT": ("V", "T"),
    "VA": ("V", "A"),
}
MODEL_SPECS: dict[str, dict[str, str]] = {
    "qwen3_5_4b": {
        "protocol": "VT",
        "judgment_file": "qwen3_5_4b/judgments.jsonl",
        "description_file": "qwen3_5_4b/descriptions.jsonl",
    },
    "gemma4_12b": {
        "protocol": "VA",
        "judgment_file": "gemma4_12b/judgments.jsonl",
        "description_file": "gemma4_12b/descriptions.jsonl",
    },
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def unique_by(rows: list[dict[str, Any]], field: str, path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}: missing non-empty {field}")
        if value in result:
            raise ValueError(f"{path}: duplicate {field}={value}")
        result[value] = row
    return result


def parse_true_source_modality(gt_describe: Any, protocol: str, sample_id: str) -> str:
    if not isinstance(gt_describe, str) or not gt_describe.strip():
        raise ValueError(f"{sample_id}: gt_describe is empty")
    matches = SOURCE_RULE.findall(gt_describe)
    if len(matches) != 1:
        raise ValueError(
            f"{sample_id}: expected exactly one true-source match, got {len(matches)}"
        )
    modality = SOURCE_WORD_TO_MODALITY[matches[0].lower()]
    if modality not in PROTOCOL_ALLOWED[protocol]:
        raise ValueError(
            f"{sample_id}: true source {modality} violates protocol {protocol}"
        )
    return modality


def parse_modal_descriptions(gt_describe: Any, protocol: str, sample_id: str) -> dict[str, str]:
    if not isinstance(gt_describe, str) or not gt_describe.strip():
        raise ValueError(f"{sample_id}: gt_describe is empty")
    text = gt_describe.strip()
    second_word = "text" if protocol == "VT" else "audio"
    second_modality = "T" if protocol == "VT" else "A"
    video_match = re.search(
        r"\b(?:The|In the|From the) video modality(?: alone)?\b",
        text,
        flags=re.IGNORECASE,
    )
    second_match = re.search(
        rf"\b(?:The|In the|From the) {second_word} modality(?: alone)?\b",
        text,
        flags=re.IGNORECASE,
    )
    combined_match = re.search(
        r"\b(?:When the two modalities are combined|"
        r"The (?:true )?overall emotion when the two modalities are combined)\b",
        text,
        flags=re.IGNORECASE,
    )
    if video_match is None or second_match is None or combined_match is None:
        raise ValueError(f"{sample_id}: gt_describe is missing modality sections")
    if not video_match.end() < second_match.start() < combined_match.start():
        raise ValueError(f"{sample_id}: modality sections are not ordered V/{second_modality}/combined")
    descriptions = {
        "V": text[video_match.start() : second_match.start()].strip(),
        second_modality: text[second_match.start() : combined_match.start()].strip(),
    }
    for modality, description in descriptions.items():
        if not description:
            raise ValueError(f"{sample_id}: empty {modality} modality description")
    return descriptions


def judgment_path(model: str) -> Path:
    return FORMAL_ROOT / MODEL_SPECS[model]["judgment_file"]


def description_path(model: str) -> Path:
    return FORMAL_ROOT / MODEL_SPECS[model]["description_file"]


def build_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, spec in MODEL_SPECS.items():
        protocol = spec["protocol"]
        allowed = PROTOCOL_ALLOWED[protocol]
        judgments = unique_by(load_jsonl(judgment_path(model)), "sample_id", judgment_path(model))
        descriptions = unique_by(
            [
                row
                for row in load_jsonl(description_path(model))
                if row.get("condition") == "M12"
            ],
            "sample_id",
            description_path(model),
        )
        candidate_ids = sorted(
            sample_id
            for sample_id, judgment in judgments.items()
            if judgment.get("final_label") == "MISREAD"
        )
        for sample_id in candidate_ids:
            description = descriptions.get(sample_id)
            if description is None:
                raise ValueError(f"{model}/{sample_id}: missing M12 description")
            if description.get("sample_type") != "Conflict":
                continue
            if description.get("protocol") != protocol:
                raise ValueError(
                    f"{model}/{sample_id}: description protocol {description.get('protocol')!r} "
                    f"does not match {protocol!r}"
                )
            judgment_protocol = judgments[sample_id].get("protocol")
            if judgment_protocol != protocol:
                raise ValueError(
                    f"{model}/{sample_id}: judgment protocol {judgment_protocol!r} "
                    f"does not match {protocol!r}"
                )
            diagnostic = description.get("diagnostic_description")
            if not isinstance(diagnostic, str) or not diagnostic.strip():
                raise ValueError(f"{model}/{sample_id}: diagnostic_description is empty")
            generated = description.get("generated_description")
            if not isinstance(generated, str) or not generated.strip():
                raise ValueError(f"{model}/{sample_id}: generated_description is empty")
            gt_describe = description.get("gt_describe")
            true_source = parse_true_source_modality(gt_describe, protocol, sample_id)
            modal_descriptions = parse_modal_descriptions(gt_describe, protocol, sample_id)
            row = {
                "schema": SCHEMA_MANIFEST,
                "audit_id": f"{model}::{sample_id}",
                "model": model,
                "protocol": protocol,
                "allowed_modalities": list(allowed),
                "sample_id": sample_id,
                "sample_type": "Conflict",
                "source_id": description.get("source_id"),
                "gt_emotion": description.get("gt_emotion"),
                "true_source_modality": true_source,
                "V_modality_description": modal_descriptions["V"],
                f"{allowed[1]}_modality_description": modal_descriptions[allowed[1]],
                "final_diagnostic_description": diagnostic.strip(),
            }
            rows.append(row)
    rows.sort(key=lambda row: (row["model"], row["sample_id"]))
    audit_ids = [row["audit_id"] for row in rows]
    if len(set(audit_ids)) != len(audit_ids):
        duplicates = [key for key, count in Counter(audit_ids).items() if count > 1]
        raise ValueError(f"duplicate audit_id values: {duplicates[:5]}")
    return rows


def render_request(row: dict[str, Any], prompt_text: str) -> tuple[list[dict[str, str]], str]:
    second = row["allowed_modalities"][1]
    user_payload = {
        "audit_id": row["audit_id"],
        "protocol": row["protocol"],
        "allowed_modalities": row["allowed_modalities"],
        "V_modality_description": row["V_modality_description"],
        f"{second}_modality_description": row[f"{second}_modality_description"],
        "final_diagnostic_description": row["final_diagnostic_description"],
        "expected_response_schema": {
            "chosen_modality": "V|T|A|unclear",
            "reason": "brief",
        },
    }
    user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user_content},
    ]
    request_hash = sha256_text(prompt_text + "\n" + user_content)
    return messages, request_hash


def build_dry_run_rows(manifest_rows: list[dict[str, Any]], prompt_text: str) -> list[dict[str, Any]]:
    prompt_hash = sha256_text(prompt_text)
    dry_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        messages, _ = render_request(row, prompt_text)
        dry_rows.append(
            {
                "schema": SCHEMA_DRY_RUN,
                "annotation_schema": SCHEMA_ANNOTATION,
                "audit_id": row["audit_id"],
                "model": row["model"],
                "sample_id": row["sample_id"],
                "protocol": row["protocol"],
                "prompt_sha256": prompt_hash,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_tokens": MAX_TOKENS,
            }
        )
    return dry_rows


def validate_annotation(annotation: Any, protocol: str) -> None:
    if not isinstance(annotation, dict):
        raise ValueError("annotation must be a JSON object")
    keys = set(annotation.keys())
    expected_keys = set(EXPECTED_ANNOTATION_KEYS)
    if keys != expected_keys:
        raise ValueError(
            f"annotation keys must be exactly {sorted(expected_keys)}, got {sorted(keys)}"
        )
    chosen = annotation["chosen_modality"]
    if chosen not in (*PROTOCOL_ALLOWED[protocol], "unclear"):
        raise ValueError(f"chosen_modality {chosen!r} violates protocol {protocol}")
    reason = annotation["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be non-empty text")


def parse_annotation_content(content: str, protocol: str) -> dict[str, str]:
    try:
        annotation = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("response content is not exact JSON") from exc
    validate_annotation(annotation, protocol)
    return {
        "chosen_modality": annotation["chosen_modality"],
        "reason": annotation["reason"].strip(),
    }


def parse_ledger_rows(
    rows: list[dict[str, Any]], source: Path
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    latest: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    for row in rows:
        audit_id = row.get("audit_id")
        if not isinstance(audit_id, str) or not audit_id:
            raise ValueError(f"{source}: ledger row has no non-empty audit_id")
        status = row.get("status")
        if status not in {"started", "success", "failed"}:
            raise ValueError(f"{source}: invalid status {status!r}")
        consumed.add(audit_id)
        latest[audit_id] = row
    return latest, consumed


def load_ledger() -> tuple[dict[str, dict[str, Any]], set[str]]:
    if not RESULTS_PATH.exists():
        return {}, set()
    return parse_ledger_rows(load_jsonl(RESULTS_PATH), RESULTS_PATH)


def model_unavailable(status_code: int, body: str) -> bool:
    lowered = body.lower()
    return status_code in {400, 404} and "model" in lowered and any(
        token in lowered
        for token in ("not found", "does not exist", "unsupported", "unavailable", "invalid")
    )


def summarize(manifest_rows: list[dict[str, Any]], dry_run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for model in MODEL_SPECS:
        model_rows = [row for row in manifest_rows if row["model"] == model]
        source_counts = Counter(row["true_source_modality"] for row in model_rows)
        by_model[model] = {
            "protocol": MODEL_SPECS[model]["protocol"],
            "manifest_count": len(model_rows),
            "true_source_counts": dict(sorted(source_counts.items())),
            "non_empty_final_diagnostic_count": sum(
                1 for row in model_rows if row["final_diagnostic_description"]
            ),
        }
    total_source_counts = Counter(row["true_source_modality"] for row in manifest_rows)
    return {
        "schema": "mprisk_misread_final_direction_extended_summary_v1",
        "repo_root": str(REPO_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "prompt_path": str(PROMPT_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "dry_run_path": str(DRY_RUN_PATH),
        "summary_path": str(SUMMARY_PATH),
        "manifest_count": len(manifest_rows),
        "dry_run_count": len(dry_run_rows),
        "unique_audit_id_count": len({row["audit_id"] for row in manifest_rows}),
        "total_true_source_counts": dict(sorted(total_source_counts.items())),
        "by_model": by_model,
        "prompt_sha256": file_sha256(PROMPT_PATH),
        "manifest_sha256": file_sha256(MANIFEST_PATH) if MANIFEST_PATH.exists() else None,
        "dry_run_sha256": file_sha256(DRY_RUN_PATH) if DRY_RUN_PATH.exists() else None,
    }


def assert_prompt_schema_consistency(prompt_text: str) -> None:
    required = '{"chosen_modality": "V|T|A|unclear", "reason": "brief"}'
    if required not in prompt_text:
        raise ValueError("prompt does not contain the exact expected JSON schema")
    forbidden = ("dominant_direction", "follows_dominant_direction")
    for token in forbidden:
        if token in prompt_text:
            raise ValueError(f"prompt must not mention {token}")


def command_self_test(_: argparse.Namespace) -> int:
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    assert_prompt_schema_consistency(prompt_text)
    vt_text = (
        "The video modality alone shows fear. "
        "The text modality alone conveys calm. "
        "When the two modalities are combined, the true overall emotion is fear. "
        "The overall judgment follows the video modality because the body shows fear."
    )
    va_text = (
        "The video modality alone shows calm. "
        "The audio modality alone conveys anger. "
        "When the two modalities are combined, the true overall emotion is anger. "
        "This judgment follows the audio modality because the voice carries anger."
    )
    if parse_true_source_modality(vt_text, "VT", "selftest-vt") != "V":
        raise AssertionError("VT true-source parse failed")
    if parse_true_source_modality(va_text, "VA", "selftest-va") != "A":
        raise AssertionError("VA true-source parse failed")
    parse_modal_descriptions(vt_text, "VT", "selftest-vt")
    parse_modal_descriptions(va_text, "VA", "selftest-va")
    try:
        parse_true_source_modality(vt_text + " This judgment follows the text modality.", "VT", "multi")
    except ValueError:
        pass
    else:
        raise AssertionError("multiple true-source matches must fail")
    try:
        parse_true_source_modality(va_text, "VT", "protocol")
    except ValueError:
        pass
    else:
        raise AssertionError("protocol mismatch must fail")
    validate_annotation({"chosen_modality": "V", "reason": "closer to visible fear"}, "VT")
    validate_annotation({"chosen_modality": "unclear", "reason": "no usable affect"}, "VA")
    try:
        validate_annotation({"chosen_modality": "A", "reason": "wrong protocol"}, "VT")
    except ValueError:
        pass
    else:
        raise AssertionError("protocol-invalid annotation must fail")
    validate_annotation({"reason": "reversed key order is valid JSON", "chosen_modality": "V"}, "VT")
    try:
        validate_annotation({"chosen_modality": "V"}, "VT")
    except ValueError:
        pass
    else:
        raise AssertionError("missing key must fail")
    parsed = parse_annotation_content(
        '{"reason":"visible expression is closer","chosen_modality":"V"}', "VT"
    )
    if parsed != {"chosen_modality": "V", "reason": "visible expression is closer"}:
        raise AssertionError("strict annotation parsing failed")
    try:
        parse_annotation_content(
            '{"chosen_modality":"V","reason":"clear","extra":true}', "VT"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("extra annotation key must fail")
    ledger_rows = [
        {"audit_id": "a", "status": "started"},
        {"audit_id": "b", "status": "success"},
        {"audit_id": "c", "status": "failed"},
    ]
    _, consumed = parse_ledger_rows(ledger_rows, Path("self-test-ledger.jsonl"))
    if consumed != {"a", "b", "c"}:
        raise AssertionError("all appeared audit_id values must be consumed")
    request_row = {
        "audit_id": "model::sample",
        "protocol": "VT",
        "allowed_modalities": ["V", "T"],
        "true_source_modality": "V",
        "V_modality_description": "visible fear",
        "T_modality_description": "spoken calm",
        "final_diagnostic_description": "the speaker sounds calm",
    }
    messages, _ = render_request(request_row, prompt_text)
    serialized_messages = canonical(messages)
    for forbidden in (
        "true_source_modality",
        "dominant_direction",
        "follows_dominant_direction",
    ):
        if forbidden in serialized_messages:
            raise AssertionError(f"request must not expose {forbidden}")
    if API_MODEL != "deepseek-v4-pro" or MAX_CONCURRENCY != 1 or MAX_TOKENS != 128:
        raise AssertionError("fixed API execution settings changed")
    print(json.dumps({"self_test": "passed"}, ensure_ascii=False, sort_keys=True))
    return 0


def command_dry_run(_: argparse.Namespace) -> int:
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    assert_prompt_schema_consistency(prompt_text)
    manifest_rows = build_manifest_rows()
    dry_run_rows = build_dry_run_rows(manifest_rows, prompt_text)
    write_jsonl(MANIFEST_PATH, manifest_rows)
    write_jsonl(DRY_RUN_PATH, dry_run_rows)
    summary = summarize(manifest_rows, dry_run_rows)
    summary["manifest_sha256"] = file_sha256(MANIFEST_PATH)
    summary["dry_run_sha256"] = file_sha256(DRY_RUN_PATH)
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def command_validate_manifest(_: argparse.Namespace) -> int:
    manifest_rows = load_jsonl(MANIFEST_PATH)
    dry_run_rows = load_jsonl(DRY_RUN_PATH)
    if len(manifest_rows) != len(dry_run_rows):
        raise ValueError("manifest and dry-run counts differ")
    manifest_ids = [row["audit_id"] for row in manifest_rows]
    dry_ids = [row["audit_id"] for row in dry_run_rows]
    if manifest_ids != dry_ids:
        raise ValueError("manifest and dry-run audit_id order differs")
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("manifest contains duplicate audit_id")
    for row in manifest_rows:
        protocol = row["protocol"]
        if row["true_source_modality"] not in PROTOCOL_ALLOWED[protocol]:
            raise ValueError(f"{row['audit_id']}: true_source_modality violates protocol")
        if not row["final_diagnostic_description"]:
            raise ValueError(f"{row['audit_id']}: final_diagnostic_description is empty")
    print(json.dumps({"validate_manifest": "passed", "count": len(manifest_rows)}, sort_keys=True))
    return 0


def summarize_run() -> int:
    manifest_rows = load_jsonl(MANIFEST_PATH)
    latest, consumed = load_ledger()
    manifest_ids = {row["audit_id"] for row in manifest_rows}
    unknown_ids = sorted(consumed - manifest_ids)
    if unknown_ids:
        raise ValueError(f"ledger contains unknown audit_id values: {unknown_ids[:5]}")
    status_counts = Counter(row["status"] for row in latest.values())
    per_model: dict[str, dict[str, Any]] = {}
    for model in MODEL_SPECS:
        model_rows = [row for row in manifest_rows if row["model"] == model]
        model_latest = [latest[row["audit_id"]] for row in model_rows if row["audit_id"] in latest]
        per_model[model] = {
            "manifest_count": len(model_rows),
            "consumed_count": len(model_latest),
            "pending_count": len(model_rows) - len(model_latest),
            "status_counts": dict(sorted(Counter(row["status"] for row in model_latest).items())),
        }
    summary = {
        "schema": "mprisk_misread_final_direction_extended_run_summary_v1",
        "api_model": API_MODEL,
        "max_concurrency": MAX_CONCURRENCY,
        "max_tokens": MAX_TOKENS,
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "manifest_count": len(manifest_rows),
        "results_path": str(RESULTS_PATH),
        "results_sha256": file_sha256(RESULTS_PATH) if RESULTS_PATH.exists() else None,
        "consumed_count": len(consumed),
        "pending_count": len(manifest_rows) - len(consumed),
        "status_counts": dict(sorted(status_counts.items())),
        "per_model": per_model,
    }
    write_json(RUN_SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def command_run(_: argparse.Namespace) -> int:
    if MAX_CONCURRENCY != 1:
        raise RuntimeError("formal run requires fixed sequential concurrency=1")
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(PROMPT_PATH)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"build the manifest first with dry-run: {MANIFEST_PATH}")
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    assert_prompt_schema_consistency(prompt_text)
    stored_manifest = load_jsonl(MANIFEST_PATH)
    fresh_manifest = build_manifest_rows()
    if stored_manifest != fresh_manifest:
        raise RuntimeError("manifest changed since dry-run; refusing to call the API")
    audit_ids = [row["audit_id"] for row in stored_manifest]
    if len(audit_ids) != 417 or len(set(audit_ids)) != 417:
        raise RuntimeError("formal run requires exactly 417 unique audit_id values")

    _, consumed = load_ledger()
    unknown_ids = sorted(consumed - set(audit_ids))
    if unknown_ids:
        raise RuntimeError(f"ledger contains unknown audit_id values: {unknown_ids[:5]}")
    pending = [row for row in stored_manifest if row["audit_id"] not in consumed]
    print(
        f"model={API_MODEL} manifest={len(stored_manifest)} consumed={len(consumed)} "
        f"pending_calls={len(pending)} concurrency={MAX_CONCURRENCY} max_tokens={MAX_TOKENS}"
    )
    if not pending:
        print("no API calls needed; every manifest audit_id already appears in the ledger")
        return summarize_run()

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is required in the process environment")

    fatal_model_error = False
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for index, row in enumerate(pending, 1):
            messages, prompt_sha256 = render_request(row, prompt_text)
            base_record = {
                "schema": SCHEMA_RESULT,
                "audit_id": row["audit_id"],
                "model": row["model"],
                "sample_id": row["sample_id"],
                "prompt_sha256": prompt_sha256,
            }
            append_jsonl(
                RESULTS_PATH,
                {
                    **base_record,
                    "status": "started",
                    "request_id": None,
                },
            )
            try:
                response = client.post(
                    API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": API_MODEL,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                        "thinking": {"type": "disabled"},
                        "temperature": 0,
                        "max_tokens": MAX_TOKENS,
                        "stream": False,
                    },
                )
                body_text = response.text
                if response.status_code >= 400:
                    fatal_model_error = model_unavailable(response.status_code, body_text)
                    raise RuntimeError(f"HTTP {response.status_code}")
                payload = response.json()
                if payload.get("model") != API_MODEL:
                    raise ValueError("API response model does not match deepseek-v4-pro")
                choices = payload.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("API response must contain exactly one choice")
                choice = choices[0]
                message = choice.get("message") if isinstance(choice, dict) else None
                if not isinstance(message, dict):
                    raise ValueError("API response choice has no message")
                if message.get("reasoning_content") not in (None, ""):
                    raise ValueError("API returned reasoning content despite disabled thinking")
                content = message.get("content")
                if not isinstance(content, str) or not content:
                    raise ValueError("API response has empty content")
                result = parse_annotation_content(content, row["protocol"])
                append_jsonl(
                    RESULTS_PATH,
                    {
                        **base_record,
                        "status": "success",
                        "request_id": payload.get("id"),
                        "response_model": payload.get("model"),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": payload.get("usage")
                        if isinstance(payload.get("usage"), dict)
                        else {},
                        "response": result,
                    },
                )
                print(f"completed {index}/{len(pending)} audit_id={row['audit_id']}")
            except Exception as exc:
                append_jsonl(
                    RESULTS_PATH,
                    {
                        **base_record,
                        "status": "failed",
                        "request_id": None,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                print(
                    f"failed {index}/{len(pending)} audit_id={row['audit_id']} "
                    f"error={type(exc).__name__}",
                    file=sys.stderr,
                )
                if fatal_model_error:
                    print("deepseek-v4-pro unavailable; stopping without fallback", file=sys.stderr)
                    break
    return summarize_run()


def command_summarize_run(_: argparse.Namespace) -> int:
    return summarize_run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    self_test = subparsers.add_parser("self-test", help="run offline consistency tests")
    self_test.set_defaults(func=command_self_test)
    dry_run = subparsers.add_parser("dry-run", help="build manifest and dry-run request JSONL")
    dry_run.set_defaults(func=command_dry_run)
    validate_manifest = subparsers.add_parser("validate-manifest", help="validate generated outputs")
    validate_manifest.set_defaults(func=command_validate_manifest)
    run = subparsers.add_parser(
        "run", help="execute one sequential deepseek-v4-pro pass over unconsumed audit_id values"
    )
    run.set_defaults(func=command_run)
    summarize_results = subparsers.add_parser(
        "summarize-results", help="summarize the append-only execution ledger"
    )
    summarize_results.set_defaults(func=command_summarize_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
