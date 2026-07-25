"""GT Description generation export and materialization of output artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mprisk.data.generated_archive_freeze import _canonical_json, _sha256
from mprisk.utils.io import atomic_write_bytes as _atomic_write

from mprisk.ground_truth._ledger import GTDescriptionGenerationLedger
from mprisk.ground_truth._plan import (
    OUTPUT_SCHEMA,
    PROVENANCE_SCHEMA,
    GTDescriptionGenerationConfig,
)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode()


def _export(
    output_root: Path,
    ledger: GTDescriptionGenerationLedger,
    config: GTDescriptionGenerationConfig,
    config_file: Path,
) -> None:
    rows = ledger.rows()
    manifest: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    ledger_export: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        ledger_export.append(
            {
                key: record[key]
                for key in record
                if key not in {"request_json", "annotation_input_json", "result_json"}
            }
        )
        if row["status"] == "completed":
            annotation_input = json.loads(row["annotation_input_json"])
            result = json.loads(row["result_json"])
            manifest.append(
                {
                    **annotation_input,
                    "schema_name": OUTPUT_SCHEMA,
                    "run_id": config.run_id,
                    "GT_DESCRIPTION": result["GT_DESCRIPTION"],
                }
            )
            raw.append(
                {
                    "sample_id": row["sample_id"],
                    "input_hash": row["input_hash"],
                    "prompt_hash": row["prompt_hash"],
                    "request": json.loads(row["request_json"]),
                    "response": result,
                }
            )
            sidecar.append(
                {
                    "sample_id": row["sample_id"],
                    "annotation_status": "preliminary_ai_draft",
                    "human_review_status": "pending_human",
                    "gt_description_sha256": hashlib.sha256(
                        result["GT_DESCRIPTION"].encode()
                    ).hexdigest(),
                }
            )
        elif row["status"] == "failed":
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "error_type": row["error_type"],
                    "error_message": row["error_message"],
                }
            )
    payloads = {
        "gt_manifest.jsonl": _jsonl(manifest),
        "raw_responses.jsonl": _jsonl(raw),
        "review_status.jsonl": _jsonl(sidecar),
        "failures.jsonl": _jsonl(failures),
        "ledger.jsonl": _jsonl(ledger_export),
        "attempts.jsonl": _jsonl([dict(row) for row in ledger.attempt_rows()]),
    }
    artifacts = {
        name.removesuffix(".jsonl"): {
            "path": (config.output_root / name).as_posix(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in payloads.items()
    }
    provenance = {
        "schema_name": PROVENANCE_SCHEMA,
        "run_id": config.run_id,
        "provider_key": config.provider_key,
        "gt_generator_model": config.gt_generator_model,
        "gt_description_schema_name": OUTPUT_SCHEMA,
        "gt_input_schema_version": config.gt_input_schema_version,
        "provider_settings_sha256": hashlib.sha256(
            _canonical_json(config.provider_settings).encode()
        ).hexdigest(),
        "config_sha256": _sha256(config_file),
        "input_manifest": config.input_manifest.as_posix(),
        "input_manifest_sha256": config.input_manifest_sha256,
        "expected_count": config.expected_count,
        "counts": dict(Counter(row["status"] for row in rows)),
        "artifacts": artifacts,
    }
    payloads["provenance.json"] = (
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    for name, content in payloads.items():
        _atomic_write(output_root / name, content)
