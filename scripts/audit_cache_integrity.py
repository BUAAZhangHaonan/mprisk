#!/usr/bin/env python3
"""Audit active cache artifacts from an immutable staging checkout."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from mprisk.assets.registry import index_assets, load_model_assets
from mprisk.cache.cache_matrix_queue import (
    _expected_batch_signature,
    _ledger_status,
    _validate_accepted_bundle,
    build_asset_signature,
    load_matrix_config,
)
from mprisk.cache.integrity import (
    CacheIntegrityError,
    audit_completed_cache,
    build_checkpoint_digest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("all", "source", "target"), default="all")
    parser.add_argument("--model-key", action="append", default=[])
    parser.add_argument("--write-receipts", action="store_true")
    parser.add_argument("--report-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_matrix_config(args.config)
    selected_models = set(args.model_key)
    unknown = selected_models - {model.model_key for model in config.models}
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    assets = index_assets(load_model_assets(config.asset_config))
    signatures: dict[str, dict[str, Any]] = {}
    checkpoint_receipts: dict[str, dict[str, Any]] = {}
    for model in config.models:
        if selected_models and model.model_key not in selected_models:
            continue
        asset = assets[model.model_key]
        receipt_path = (
            config.output_root
            / "receipts"
            / "checkpoints"
            / f"{model.model_key}.json"
        )
        checkpoint_receipts[model.model_key] = build_checkpoint_digest(
            asset.local_model_path,
            receipt_path=receipt_path,
            write_receipt=args.write_receipts,
        )
        signatures[model.model_key] = build_asset_signature(config, model)

    records = []
    for job in config.jobs:
        if args.stage != "all" and job.domain.domain != args.stage:
            continue
        if selected_models and job.model.model_key not in selected_models:
            continue
        ledger = _ledger_status(job.output_root, job.domain.expected_tasks)
        record: dict[str, Any] = {
            "job_id": job.job_id,
            "ledger": ledger,
            "checkpoint_sha256": checkpoint_receipts[job.model.model_key][
                "checkpoint_sha256"
            ],
        }
        accepted = job.model.accepted_bundle_domains.get(job.domain.domain)
        if accepted:
            try:
                _validate_accepted_bundle(
                    config,
                    job,
                    accepted,
                    asset_signature=signatures[job.model.model_key],
                )
            except (CacheIntegrityError, FileNotFoundError, KeyError, ValueError) as exc:
                record.update(
                    status="incompatible_accepted_bundle",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    required_action=(
                        "Supply a field-exact signed equivalence waiver after independent "
                        "review, or re-extract every expected task."
                    ),
                    tasks_requiring_reextraction=job.domain.expected_tasks,
                )
            else:
                record["status"] = "accepted_bundle"
        elif ledger["status"] == "complete":
            expected_signature = _expected_batch_signature(config, job)
            try:
                completion = audit_completed_cache(
                    job.output_root,
                    expected_signature=expected_signature,
                    expected_tasks=job.domain.expected_tasks,
                    write_receipt=args.write_receipts,
                )
            except (CacheIntegrityError, OSError, sqlite3.Error, ValueError) as exc:
                record.update(
                    status="incompatible_completed_cache",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    required_action="Re-extract every task not proven by the artifact audit.",
                    tasks_requiring_reextraction=job.domain.expected_tasks,
                )
            else:
                record["status"] = "complete"
                record["completion_receipt"] = completion
        else:
            record["status"] = ledger["status"]
        records.append(record)
    payload = {
        "schema": "mprisk_cache_integrity_audit_v1",
        "write_receipts": args.write_receipts,
        "summary": {
            "jobs": len(records),
            "incompatible_jobs": sum(
                str(record["status"]).startswith("incompatible")
                for record in records
            ),
            "tasks_requiring_reextraction": sum(
                int(record.get("tasks_requiring_reextraction", 0))
                for record in records
            ),
        },
        "records": records,
    }
    if args.report_path is not None:
        report_path = args.report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f".{report_path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(report_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(
        record["status"] in {"complete", "accepted_bundle", "absent", "incomplete"}
        for record in records
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
