#!/usr/bin/env python3
"""One-shot DeepSeek audit of dominant-direction affect descriptions.

This script builds a frozen candidate manifest from the canonical SDR states,
formal Misread judgments, source descriptions, and model M12 descriptions.
The run ledger is append-only and treats every recorded request (including a
started or failed request) as consumed, so an interrupted run never retries
silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_ROOT = REPO_ROOT / "outputs/downstream/delivery_20260716/seed20260717/tme_ablation_v1"
FORMAL_ROOT = REPO_ROOT.parent / "mprisk-v2/outputs/v2/misread"
SOURCE_ROOT = REPO_ROOT / "data/processed/manifests/protocol_manifests_merged"
PROMPT_PATH = Path(__file__).with_name("dominant_misread_direction_prompt.txt")
OUTPUT_ROOT = REPO_ROOT / "outputs/analysis/dominant_misread_direction_v1"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.jsonl"
RESULTS_PATH = OUTPUT_ROOT / "results.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_MODEL = "deepseek-v4-pro"
API_KEY_ENV = "DEEPSEEK_API_KEY"
REQUEST_TIMEOUT_SECONDS = 90.0
MAX_TOKENS = 128

MODEL_SPECS: dict[str, dict[str, str]] = {
    "qwen3_vl_8b": {
        "state_dir": "qwen3_vl_8b",
        "protocol": "vt",
        "source_file": "vt_merged_primary.jsonl",
        "description_file": "qwen3_vl_8b/descriptions.jsonl",
        "judgment_file": "qwen3_vl_8b/judgments.jsonl",
    },
    "internvl3_5_8b": {
        "state_dir": "internvl3_5_8b",
        "protocol": "vt",
        "source_file": "vt_merged_primary.jsonl",
        "description_file": "internvl3_5_8b/descriptions.jsonl",
        "judgment_file": "internvl3_5_8b/judgments.jsonl",
    },
    "qwen2_5_omni_7b": {
        "state_dir": "qwen2_5_omni_7b",
        "protocol": "va",
        "source_file": "va_merged_primary.jsonl",
        "description_file": "qwen2_5_omni_7b/descriptions.jsonl",
        "judgment_file": "qwen2_5_omni_7b/judgments.jsonl",
    },
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(record) + "\n")
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
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
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


def parse_modal_descriptions(text: Any, protocol: str, sample_id: str) -> tuple[str, str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{sample_id}: missing gt_describe")
    second_word = "text" if protocol == "vt" else "audio"
    normalized = text.strip()
    video_match = re.search(
        r"\b(?:The|In the) video modality(?: alone)?\b",
        normalized,
        flags=re.IGNORECASE,
    )
    second_match = re.search(
        rf"\b(?:The|In the) {second_word} modality(?: alone)?\b",
        normalized,
        flags=re.IGNORECASE,
    )
    combined_match = re.search(
        r"\b(?:When the two modalities are combined|"
        r"The (?:true )?overall emotion when the two modalities are combined)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if (
        video_match is None
        or second_match is None
        or combined_match is None
        or video_match.start() != 0
        or not video_match.end() < second_match.start() < combined_match.start()
    ):
        raise ValueError(
            f"{sample_id}: gt_describe does not contain the expected "
            f"video/{second_word}/combined sections"
        )
    vision = normalized[video_match.start() : second_match.start()].strip()
    second = normalized[second_match.start() : combined_match.start()].strip()
    if not vision or not second:
        raise ValueError(f"{sample_id}: empty modality description")
    return vision, second


def state_path(model: str) -> Path:
    spec = MODEL_SPECS[model]
    return (
        STATE_ROOT
        / spec["state_dir"]
        / "tme_pa_dstrong_v2/state_all_registered_splits/state_patterns.jsonl"
    )


def judgment_path(model: str) -> Path:
    return FORMAL_ROOT / MODEL_SPECS[model]["judgment_file"]


def description_path(model: str) -> Path:
    return FORMAL_ROOT / MODEL_SPECS[model]["description_file"]


def build_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for model, spec in MODEL_SPECS.items():
        protocol = spec["protocol"]
        second_modality = "T" if protocol == "vt" else "A"

        state_rows = load_jsonl(state_path(model))
        state_map: dict[str, dict[str, Any]] = {}
        for row in state_rows:
            if row.get("sample_type") != "Conflict" or row.get("pattern") != "Dominant":
                continue
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{state_path(model)}: invalid candidate sample_id")
            if sample_id in state_map:
                raise ValueError(f"{state_path(model)}: duplicate candidate {sample_id}")
            state_map[sample_id] = row

        judgment_map = unique_by(load_jsonl(judgment_path(model)), "sample_id", judgment_path(model))
        candidate_ids = sorted(
            sample_id
            for sample_id, row in state_map.items()
            if judgment_map.get(sample_id, {}).get("final_label") == "MISREAD"
        )

        source_path = SOURCE_ROOT / spec["source_file"]
        source_map = unique_by(load_jsonl(source_path), "sample_id", source_path)

        description_rows = load_jsonl(description_path(model))
        m12_rows = [
            row
            for row in description_rows
            if row.get("condition") == "M12"
        ]
        description_map = unique_by(m12_rows, "sample_id", description_path(model))

        for sample_id in candidate_ids:
            state = state_map[sample_id]
            source = source_map.get(sample_id)
            if source is None:
                raise ValueError(f"{model}/{sample_id}: missing source manifest row")
            description = description_map.get(sample_id)
            if description is None:
                raise ValueError(f"{model}/{sample_id}: missing M12 description")
            final_text = description.get("diagnostic_description")
            if not isinstance(final_text, str) or not final_text.strip():
                raise ValueError(
                    f"{model}/{sample_id}: diagnostic_description is not complete text"
                )
            if state.get("protocol") != protocol:
                raise ValueError(
                    f"{model}/{sample_id}: state protocol {state.get('protocol')!r} "
                    f"does not match expected {protocol!r}"
                )
            value_r = state.get("R")
            if not isinstance(value_r, (int, float)) or not math.isfinite(float(value_r)):
                raise ValueError(f"{model}/{sample_id}: R is not finite")
            if value_r == 0:
                raise ValueError(f"{model}/{sample_id}: R=0 has no unique direction")
            vision_text, second_text = parse_modal_descriptions(
                source.get("gt_describe"), protocol, sample_id
            )
            direction = "V" if value_r > 0 else second_modality
            manifest.append(
                {
                    "schema": "mprisk_dominant_misread_direction_manifest_v1",
                    "model": model,
                    "sample_id": sample_id,
                    "protocol": protocol.upper(),
                    "direction": direction,
                    "direction_basis": "state.protocol + sign(state.R)",
                    "state_R": float(value_r),
                    "state_lean": state.get("lean"),
                    "V_pre_description": vision_text,
                    f"{second_modality}_pre_description": second_text,
                    "DIAGNOSTIC_AFFECT_DESCRIPTION": final_text.strip(),
                }
            )
    manifest.sort(key=lambda row: (row["model"], row["sample_id"]))
    if len({(row["model"], row["sample_id"]) for row in manifest}) != len(manifest):
        raise ValueError("manifest contains duplicate model/sample_id pairs")
    return manifest


def write_manifest(rows: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def manifest_fingerprint(rows: list[dict[str, Any]]) -> str:
    encoded = "".join(canonical(row) + "\n" for row in rows).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_key(row: dict[str, Any]) -> str:
    return f"{row['model']}\t{row['sample_id']}"


def load_ledger() -> tuple[dict[str, dict[str, Any]], set[str]]:
    if not RESULTS_PATH.exists():
        return {}, set()
    latest: dict[str, dict[str, Any]] = {}
    started: set[str] = set()
    for row in load_jsonl(RESULTS_PATH):
        key = request_key(row)
        status = row.get("status")
        if status == "started":
            started.add(key)
        elif status in {"success", "failed"}:
            started.add(key)
        else:
            raise ValueError(f"{RESULTS_PATH}: invalid status {status!r}")
        latest[key] = row
    return latest, started


def render_request(row: dict[str, Any], prompt: str) -> tuple[list[dict[str, str]], str]:
    second_modality = "T" if row["protocol"] == "VT" else "A"
    user_input = {
        "dominant_direction": row["direction"],
        "V_pre_description": row["V_pre_description"],
        f"{second_modality}_pre_description": row[f"{second_modality}_pre_description"],
        "final_description": row["DIAGNOSTIC_AFFECT_DESCRIPTION"],
    }
    system = prompt.strip()
    user = canonical(user_input)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()


def validate_judge_output(content: str, direction: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("response content is not exact JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "follows_dominant_direction",
        "chosen_modality",
        "reason",
    }:
        raise ValueError("response JSON has an invalid key set")
    follows = value["follows_dominant_direction"]
    chosen = value["chosen_modality"]
    reason = value["reason"]
    if type(follows) is not bool:
        raise ValueError("follows_dominant_direction must be boolean")
    if chosen not in {"V", "T", "A", "unclear"}:
        raise ValueError("chosen_modality must be V, T, A, or unclear")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be non-empty text")
    if chosen == "unclear" and follows:
        raise ValueError("unclear cannot be marked as following")
    if follows and chosen != direction:
        raise ValueError("true result must choose the dominant modality")
    return {
        "follows_dominant_direction": follows,
        "chosen_modality": chosen,
        "reason": reason.strip(),
    }


def model_unavailable(status_code: int, body: str) -> bool:
    lowered = body.lower()
    return status_code in {400, 404} and "model" in lowered and any(
        token in lowered
        for token in ("not found", "does not exist", "unsupported", "unavailable", "invalid")
    )


def run_once() -> int:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(PROMPT_PATH)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"prepare manifest first: {MANIFEST_PATH}")
    stored_manifest = load_jsonl(MANIFEST_PATH)
    fresh_manifest = build_manifest()
    if stored_manifest != fresh_manifest:
        raise RuntimeError("manifest changed since dry-run; refusing to call the API")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is required in the process environment")

    latest, consumed = load_ledger()
    pending = [row for row in stored_manifest if request_key(row) not in consumed]
    print(f"manifest={len(stored_manifest)} pending_calls={len(pending)}")
    if not pending:
        print("no API calls needed; all request keys are already recorded")
        return summarize()

    fatal_model_error = False
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for index, row in enumerate(pending, 1):
            key = request_key(row)
            messages, prompt_sha256 = render_request(row, prompt)
            append_jsonl(
                RESULTS_PATH,
                {
                    "schema": "mprisk_dominant_misread_direction_result_v1",
                    "status": "started",
                    "model": row["model"],
                    "sample_id": row["sample_id"],
                    "request_id": None,
                    "prompt_sha256": prompt_sha256,
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
                result = validate_judge_output(content, row["direction"])
                record = {
                    "schema": "mprisk_dominant_misread_direction_result_v1",
                    "status": "success",
                    "model": row["model"],
                    "sample_id": row["sample_id"],
                    "request_id": payload.get("id"),
                    "response_model": payload.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
                    "response": result,
                    "prompt_sha256": prompt_sha256,
                }
                append_jsonl(RESULTS_PATH, record)
                print(f"completed {index}/{len(pending)}")
            except Exception as exc:
                append_jsonl(
                    RESULTS_PATH,
                    {
                        "schema": "mprisk_dominant_misread_direction_result_v1",
                        "status": "failed",
                        "model": row["model"],
                        "sample_id": row["sample_id"],
                        "request_id": None,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "prompt_sha256": prompt_sha256,
                    },
                )
                print(
                    f"failed {index}/{len(pending)} model={row['model']} "
                    f"sample_id={row['sample_id']} error={type(exc).__name__}",
                    file=sys.stderr,
                )
                if fatal_model_error:
                    print("deepseek-v4-pro unavailable; stopping without fallback", file=sys.stderr)
                    break
    return summarize()


def summarize() -> int:
    rows = load_jsonl(MANIFEST_PATH) if MANIFEST_PATH.exists() else []
    latest, consumed = load_ledger()
    per_model: dict[str, dict[str, Any]] = {}
    for model in MODEL_SPECS:
        model_rows = [row for row in rows if row["model"] == model]
        model_keys = {request_key(row) for row in model_rows}
        records = [latest[key] for key in model_keys if key in latest]
        successes = [row for row in records if row.get("status") == "success"]
        failed = [row for row in records if row.get("status") == "failed"]
        started_unresolved = [row for row in records if row.get("status") == "started"]
        true_count = sum(
            row.get("response", {}).get("follows_dominant_direction") is True
            for row in successes
        )
        unclear_count = sum(
            row.get("response", {}).get("chosen_modality") == "unclear"
            for row in successes
        )
        false_count = len(successes) - true_count - unclear_count
        direction_counts = Counter(row["direction"] for row in model_rows)
        per_model[model] = {
            "candidate_count": len(model_rows),
            "unique_sample_id_count": len({row["sample_id"] for row in model_rows}),
            "call_count": sum(key in consumed for key in model_keys),
            "success_count": len(successes),
            "failed_count": len(failed),
            "unresolved_started_count": len(started_unresolved),
            "true_count": true_count,
            "false_count": false_count,
            "unclear_count": unclear_count,
            "true_over_candidate": true_count / len(model_rows) if model_rows else None,
            "true_over_success": true_count / len(successes) if successes else None,
            "direction_distribution": dict(sorted(direction_counts.items())),
        }
    summary = {
        "schema": "mprisk_dominant_misread_direction_summary_v1",
        "api_model": API_MODEL,
        "manifest_sha256": manifest_fingerprint(rows) if rows else None,
        "per_model": per_model,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(canonical(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("prepare", "run", "summarize", "self-test"), required=True
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        rows = build_manifest()
        write_manifest(rows)
        print(f"manifest_path={MANIFEST_PATH}")
        print(f"manifest_sha256={manifest_fingerprint(rows)}")
        for model in MODEL_SPECS:
            model_rows = [row for row in rows if row["model"] == model]
            print(
                f"{model}: candidates={len(model_rows)} "
                f"unique_sample_ids={len({row['sample_id'] for row in model_rows})} "
                f"directions={dict(Counter(row['direction'] for row in model_rows))}"
            )
        print(f"expected_api_calls={len(rows)}")
        return 0
    if args.mode == "run":
        return run_once()
    if args.mode == "summarize":
        return summarize()
    validate_judge_output(
        '{"follows_dominant_direction": true, "chosen_modality": "V", "reason": "clear"}',
        "V",
    )
    validate_judge_output(
        '{"follows_dominant_direction": false, "chosen_modality": "unclear", "reason": "ambiguous"}',
        "V",
    )
    print("self-test=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
