"""Diagnostic Affect Description artifact materialization and config loading."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from mprisk.utils.io import (
    atomic_write_text as _atomic_text,
    canonical_json as _canonical_json,
    sha256_file as _sha256,
)

from mprisk.diagnostic_affect._ledger import DiagnosticAffectDescriptionLedger
from mprisk.diagnostic_affect._plan import (
    CANONICAL_DIAGNOSTIC_AFFECT_PROMPT,
    CONFIG_SCHEMA,
    PROVENANCE_SCHEMA,
)


def export_diagnostic_affect_descriptions(
    records: Sequence[dict[str, Any]], destination: Path
) -> None:
    """Write a deterministic manifest only after complete records have been validated."""
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Description manifest contains duplicate sample_id values")
    for record in records:
        text = str(record.get("DIAGNOSTIC_AFFECT_DESCRIPTION", ""))
        if not text.strip():
            raise ValueError(f"empty DIAGNOSTIC_AFFECT_DESCRIPTION for sample {record['sample_id']}")
    _atomic_text(destination, "".join(_canonical_json(record) + "\n" for record in records))


def _materialize(
    ledger: DiagnosticAffectDescriptionLedger,
    output_root: Path,
    signature: dict[str, Any],
) -> None:
    records = ledger.completed_records()
    export_diagnostic_affect_descriptions(records, output_root / "manifest.jsonl")
    _atomic_text(
        output_root / "failures.jsonl",
        "".join(_canonical_json(row) + "\n" for row in ledger.failures()),
    )
    _atomic_text(
        output_root / "attempts.jsonl",
        "".join(_canonical_json(row) + "\n" for row in ledger.attempt_records()),
    )
    _atomic_text(
        output_root / "summary.json",
        json.dumps(ledger.summary(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        output_root / "provenance.json",
        json.dumps(
            {
                "schema_name": PROVENANCE_SCHEMA,
                "run_id": signature["run_id"],
                "canonical_prompt": CANONICAL_DIAGNOSTIC_AFFECT_PROMPT,
                "signature": signature,
                "artifacts": {
                    name: {"path": filename, "sha256": _sha256(output_root / filename)}
                    for name, filename in {
                        "manifest": "manifest.jsonl",
                        "failures": "failures.jsonl",
                        "attempts": "attempts.jsonl",
                        "summary": "summary.json",
                    }.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_name") != CONFIG_SCHEMA:
        raise ValueError(f"Unsupported diagnostic-description config: {path}")
    required = {
        "schema_name",
        "run_id",
        "asset_config",
        "manifest_path",
        "output_root",
        "subject_model_key",
        "model_path",
        "protocol",
        "condition",
        "dataset",
        "split",
        "device",
        "dtype",
        "max_new_tokens",
        "video_fps",
        "attn_implementation",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Diagnostic-description config is missing: {sorted(missing)}")
    if set(value) != required:
        raise ValueError("Diagnostic-description config contains unsupported fields")
    return value
