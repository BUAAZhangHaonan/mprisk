"""Prepare the frozen two-stage Diagnostic Description -> Misread matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from mprisk.assets.registry import index_assets, load_model_assets

MATRIX_SCHEMA = "mprisk_cross_domain_misread_matrix_v1"
PLAN_SCHEMA = "mprisk_cross_domain_misread_plan_v1"
GT_SCHEMA = "mprisk_gt_description_v1"
GT_INPUT_SCHEMA = "gt_annotation_input_v1"
DIAGNOSTIC_SCHEMA = "mprisk_diagnostic_affect_description_config_v2"
ENSEMBLE_SCHEMA = "mprisk_ensemble_misread_judgment_config_v1"
REQUEST_PLAN_SCHEMA = "mprisk_misread_request_plan_v1"
SOURCE_LABEL_SCHEMA = "mprisk_v2_misread_label_v1"
GT_COVERAGE_SCHEMA = "mprisk_target_gt_coverage_v1"


class TargetGTCoverageBlocked(RuntimeError):
    """Raised after recording that canonical target GT is incomplete."""


def prepare_matrix(config_path: Path, *, destination: Path) -> dict[str, Any]:
    """Materialize immutable manifests/configs without loading models or calling an API."""
    config = _load_yaml(config_path)
    if config.get("schema_name") != MATRIX_SCHEMA:
        raise ValueError(f"Unsupported matrix schema: {config.get('schema_name')!r}")
    required = {
        "schema_name",
        "run_id",
        "asset_config",
        "cache_matrix_config",
        "frame_plan_validation",
        "bundle_root",
        "existing_source_judgment_root",
        "output_root",
        "api_url",
        "temperature",
        "confidence_threshold",
        "flash_model",
        "pro_model",
        "flash_replicates",
        "request_timeout_seconds",
        "max_concurrency",
        "pricing",
        "jobs",
    }
    if set(config) != required:
        raise ValueError(f"Matrix config fields differ: {sorted(set(config) ^ required)}")
    if config["temperature"] != 0 or config["flash_replicates"] != 3:
        raise ValueError("The fixed protocol requires temperature=0 and three Flash calls")
    assets = index_assets(
        load_model_assets(Path(config["asset_config"]), require_local_paths=False)
    )
    asset_config_path = Path(config["asset_config"]).expanduser().resolve()
    cache_matrix_path = Path(config["cache_matrix_config"]).expanduser().resolve()
    frame_validation_path = Path(config["frame_plan_validation"]).expanduser().resolve()
    bundle_root = Path(config["bundle_root"]).expanduser().resolve()
    source_judgment_root = Path(config["existing_source_judgment_root"]).expanduser().resolve()
    output_root = Path(config["output_root"]).expanduser().resolve()
    destination = destination.expanduser().resolve()
    formal_contract = _validate_formal_cache_contract(
        cache_matrix_path=cache_matrix_path,
        asset_config_path=asset_config_path,
        frame_validation_path=frame_validation_path,
    )
    canonical_models = formal_contract["models"]
    canonical_domains = formal_contract["domains"]
    jobs = config["jobs"]
    if not isinstance(jobs, list) or len(jobs) != 17:
        raise ValueError("The matrix must contain 16 target jobs plus the Phi-4 source job")
    if len({job.get("job_id") for job in jobs}) != len(jobs):
        raise ValueError("job_id values must be unique")
    target_model_keys = {
        _required_text(job, "model_key") for job in jobs if job.get("domain") == "target"
    }
    if target_model_keys != set(canonical_models):
        raise ValueError("Target jobs do not match the canonical 16-model cache matrix")
    gt_coverage = _audit_target_gt_coverage(
        jobs=jobs,
        bundle_root=bundle_root,
        canonical_domains=canonical_domains,
    )
    coverage_path = destination / "target_gt_coverage_audit.json"
    _atomic_json(coverage_path, gt_coverage)
    if gt_coverage["status"] != "PASS":
        _write_blocked_gt_plan(
            destination=destination,
            config_path=config_path,
            run_id=_required_text(config, "run_id"),
            coverage=gt_coverage,
            coverage_path=coverage_path,
        )
        raise TargetGTCoverageBlocked(
            "Target GT_DESCRIPTION coverage is incomplete; Misread request planning is blocked"
        )
    source_labels = _validate_existing_source_labels(
        judgment_root=source_judgment_root,
        bundle_root=bundle_root,
        canonical_models=canonical_models,
        canonical_domains=canonical_domains,
    )

    plan_jobs: list[dict[str, Any]] = []
    request_plan: list[dict[str, Any]] = []
    total_pending_samples = 0
    total_existing_samples = 0
    for raw_job in jobs:
        job = _validate_job(raw_job)
        model_key = job["model_key"]
        if model_key not in assets:
            raise ValueError(f"Unknown matrix model: {model_key}")
        asset = assets[model_key]
        protocol = job["protocol"].upper()
        if protocol.lower() not in asset.protocols:
            raise ValueError(f"{model_key} does not support {protocol}")
        canonical_model = canonical_models.get(model_key)
        if canonical_model is None or canonical_model["protocol"].upper() != protocol:
            raise ValueError(f"Canonical cache protocol mismatch: {job['job_id']}")
        canonical_domain = canonical_domains.get(f"{job['domain']}:{protocol.lower()}")
        if canonical_domain is None:
            raise ValueError(f"Missing canonical domain/protocol: {job['job_id']}")
        source_manifest = (bundle_root / job["manifest_path"]).resolve()
        _require_within(source_manifest, bundle_root)
        if source_manifest != Path(canonical_domain["source_manifest"]):
            raise ValueError(f"Formal manifest mismatch: {job['job_id']}")
        if job["expected_count"] != canonical_domain["expected_samples"]:
            raise ValueError(f"Canonical sample count mismatch: {job['job_id']}")
        rows = _read_jsonl(source_manifest)
        if len(rows) != job["expected_count"]:
            raise ValueError(
                f"{job['job_id']} expected {job['expected_count']} rows, observed {len(rows)}"
            )
        ids = [_required_text(row, "sample_id") for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs in {job['job_id']}")
        if any(str(row.get("protocol", "")).upper() != protocol for row in rows):
            raise ValueError(f"Protocol mismatch in {job['job_id']}")

        job_root = destination / "jobs" / job["job_id"]
        normalized_manifest = job_root / "input_manifest.jsonl"
        gt_manifest = job_root / "gt_description_manifest.jsonl"
        normalized = [
            _normalize_manifest_row(
                row,
                bundle_root=bundle_root,
                dataset=job["dataset"],
                split=job["split"],
                protocol=protocol,
            )
            for row in rows
        ]
        gt_rows = [
            _gt_row(row, domain=job["domain"], dataset=job["dataset"], split=job["split"])
            for row in normalized
        ]
        _atomic_jsonl(normalized_manifest, normalized)
        _atomic_jsonl(gt_manifest, gt_rows)

        diagnostic_root = output_root / "diagnostic_affect" / job["job_id"]
        judgment_root = output_root / "misread_judgment" / job["job_id"]
        diagnostic_config = job_root / "diagnostic.yaml"
        ensemble_config = job_root / "ensemble_judge.yaml"
        diagnostic_payload = {
            "schema_name": DIAGNOSTIC_SCHEMA,
            "run_id": f"{config['run_id']}__{job['job_id']}__diagnostic",
            "asset_config": str(Path(config["asset_config"]).expanduser().resolve()),
            "manifest_path": str(normalized_manifest),
            "output_root": str(diagnostic_root),
            "subject_model_key": model_key,
            "model_path": str(asset.local_model_path.expanduser().resolve()),
            "protocol": protocol,
            "condition": "M12",
            "dataset": job["dataset"],
            "split": job["split"],
            "device": "cuda:0",
            "dtype": canonical_model["dtype"],
            "max_new_tokens": 64,
            "video_fps": 1.0,
            "attn_implementation": "sdpa",
        }
        diagnostic_manifest = diagnostic_root / "manifest.jsonl"
        diagnostic_ready = _diagnostic_manifest_ready(
            diagnostic_manifest,
            sample_ids=set(ids),
            model_key=model_key,
            protocol=protocol,
            split=job["split"],
            run_id=diagnostic_payload["run_id"],
        )
        ensemble_payload = {
            "schema_name": ENSEMBLE_SCHEMA,
            "run_id": f"{config['run_id']}__{job['job_id']}__judge",
            "status": "ready" if diagnostic_ready else "pending",
            "subject_model_key": model_key,
            "protocol": protocol,
            "split": job["split"],
            "api_url": config["api_url"],
            "temperature": 0,
            "confidence_threshold": config["confidence_threshold"],
            "flash_model": config["flash_model"],
            "pro_model": config["pro_model"],
            "flash_replicates": 3,
            "gt_coverage_receipt_path": str(coverage_path),
            "gt_description_manifest_path": str(gt_manifest),
            "diagnostic_affect_description_manifest_path": str(diagnostic_manifest),
            "diagnostic_run_id": diagnostic_payload["run_id"],
            "output_root": str(judgment_root),
            "request_timeout_seconds": config["request_timeout_seconds"],
            "max_concurrency": config["max_concurrency"],
            "pricing": config["pricing"],
        }
        _atomic_yaml(diagnostic_config, diagnostic_payload)
        _atomic_yaml(ensemble_config, ensemble_payload)

        existing = job.get("existing_judgment_path")
        existing_path_text = None
        existing_sha256 = None
        if existing:
            existing_path = Path(existing).expanduser().resolve()
            existing_rows = _read_jsonl(existing_path)
            existing_ids = {_required_text(row, "sample_id") for row in existing_rows}
            if len(existing_rows) != len(existing_ids) or existing_ids != set(ids):
                raise ValueError(f"Existing judgment coverage mismatch: {job['job_id']}")
            if any(row.get("subject_model_key") != model_key for row in existing_rows):
                raise ValueError(f"Existing judgment model mismatch: {job['job_id']}")
            for row in existing_rows:
                if (
                    row.get("schema") != SOURCE_LABEL_SCHEMA
                    or row.get("protocol") != protocol
                    or row.get("final_label") not in {"MISREAD", "NON_MISREAD"}
                ):
                    raise ValueError(f"Existing judgment label mismatch: {job['job_id']}")
                flashes = row.get("flash")
                if not isinstance(flashes, list) or len(flashes) != 3:
                    raise ValueError(
                        f"Existing judgment does not contain three Flash calls: {job['job_id']}"
                    )
                for flash in flashes:
                    if (
                        not isinstance(flash, dict)
                        or flash.get("judge_model") != config["flash_model"]
                        or flash.get("decision") not in {"MISREAD", "NON_MISREAD", "UNCERTAIN"}
                        or not isinstance(flash.get("confidence"), int | float)
                        or not 0 <= flash["confidence"] <= 1
                    ):
                        raise ValueError(f"Existing Flash judgment mismatch: {job['job_id']}")
                pro = row.get("pro_arbitration")
                if pro is not None and (
                    not isinstance(pro, dict)
                    or pro.get("judge_model") != config["pro_model"]
                    or pro.get("decision") not in {"MISREAD", "NON_MISREAD", "UNCERTAIN"}
                ):
                    raise ValueError(f"Existing Pro judgment mismatch: {job['job_id']}")
            status = "existing_verified"
            existing_path_text = str(existing_path)
            existing_sha256 = _sha256(existing_path)
            total_existing_samples += len(ids)
        else:
            status = "pending_diagnostic"
            total_pending_samples += len(ids)
            request_plan.extend(
                _planned_request_records(
                    run_id=config["run_id"],
                    job_id=job["job_id"],
                    model_key=model_key,
                    protocol=protocol,
                    sample_ids=ids,
                    flash_model=config["flash_model"],
                    pro_model=config["pro_model"],
                )
            )
        plan_jobs.append(
            {
                **job,
                "status": status,
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": _sha256(source_manifest),
                "normalized_manifest": str(normalized_manifest),
                "normalized_manifest_sha256": _sha256(normalized_manifest),
                "gt_description_manifest": str(gt_manifest),
                "gt_description_manifest_sha256": _sha256(gt_manifest),
                "diagnostic_config": str(diagnostic_config),
                "ensemble_config": str(ensemble_config),
                "diagnostic_output_root": str(diagnostic_root),
                "judgment_output_root": str(judgment_root),
                "ensemble_status": ensemble_payload["status"],
                "existing_judgment_path": existing_path_text,
                "existing_judgment_sha256": existing_sha256,
            }
        )

    matrix_model_keys = {job["model_key"] for job in jobs if job["domain"] == "target"}
    if len(matrix_model_keys) != 16:
        raise ValueError("Target matrix must contain exactly 16 distinct model keys")
    if (
        sum(job["model_key"] == "phi4_multimodal" and job["domain"] == "source" for job in jobs)
        != 1
    ):
        raise ValueError("The matrix requires exactly one Phi-4 source-domain supplement")
    expected_flash_calls = total_pending_samples * 3
    expected_pro_upper_bound = total_pending_samples
    if len(request_plan) != expected_flash_calls + expected_pro_upper_bound:
        raise ValueError("Planned request count does not match the fixed 3+1 protocol")
    call_ids = [record["call_id"] for record in request_plan]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("Planned request call IDs must be globally unique")
    request_plan_path = destination / "request_plan_ledger.jsonl"
    _atomic_jsonl(request_plan_path, request_plan)
    plan = {
        "schema_name": PLAN_SCHEMA,
        "run_id": config["run_id"],
        "config_path": str(config_path.expanduser().resolve()),
        "config_sha256": _sha256(config_path),
        "asset_config_path": str(asset_config_path),
        "asset_config_sha256": _sha256(asset_config_path),
        "formal_cache_contract": formal_contract,
        "existing_source_labels": source_labels,
        "api_requests_issued": 0,
        "api_key_accessed": False,
        "target_model_count": 16,
        "job_count": len(plan_jobs),
        "pending_sample_count": total_pending_samples,
        "existing_verified_sample_count": total_existing_samples,
        "flash_request_count_if_complete": expected_flash_calls,
        "pro_request_upper_bound": expected_pro_upper_bound,
        "max_api_request_count": len(request_plan),
        "unique_planned_call_id_count": len(set(call_ids)),
        "request_plan_ledger_path": str(request_plan_path),
        "request_plan_ledger_sha256": _sha256(request_plan_path),
        "jobs": plan_jobs,
    }
    _atomic_json(destination / "plan.json", plan)
    _atomic_jsonl(destination / "jobs.jsonl", plan_jobs)
    return plan


def _validate_job(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Each matrix job must be an object")
    required = {
        "job_id",
        "model_key",
        "domain",
        "protocol",
        "manifest_path",
        "expected_count",
        "dataset",
        "split",
    }
    optional = {"existing_judgment_path"}
    if not required.issubset(value) or set(value) - required - optional:
        raise ValueError(f"Invalid matrix job fields: {value}")
    if value["domain"] not in {"source", "target"}:
        raise ValueError("domain must be source or target")
    if value["protocol"] not in {"VT", "VA"}:
        raise ValueError("protocol must be VT or VA")
    if not isinstance(value["expected_count"], int) or value["expected_count"] <= 0:
        raise ValueError("expected_count must be positive")
    for key in required - {"expected_count"}:
        _required_text(value, key)
    return dict(value)


def _normalize_manifest_row(
    row: dict[str, Any], *, bundle_root: Path, dataset: str, split: str, protocol: str
) -> dict[str, Any]:
    result = dict(row)
    result["source_dataset"] = dataset
    result["split"] = split
    result["protocol"] = protocol
    media = result.get("media_paths")
    if not isinstance(media, dict):
        raise ValueError(f"Missing media_paths: {result.get('sample_id')}")
    resolved: dict[str, str] = {}
    for key, raw_path in media.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Invalid media path: {result.get('sample_id')}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (bundle_root / path).resolve()
            _require_within(path, bundle_root)
        else:
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved[key] = str(path)
    result["media_paths"] = resolved
    return result


def _gt_row(row: dict[str, Any], *, domain: str, dataset: str, split: str) -> dict[str, Any]:
    if domain == "target":
        description = _required_text(row, "GT_DESCRIPTION")
        basis = "canonical_target_manifest.GT_DESCRIPTION"
    else:
        description = _required_text(row, "gt_describe")
        basis = "frozen_gt_describe"
    return {
        "schema_name": GT_SCHEMA,
        "gt_input_schema_version": GT_INPUT_SCHEMA,
        "sample_id": _required_text(row, "sample_id"),
        "dataset": dataset,
        "split": split,
        "GT_DESCRIPTION": description,
        "reference_basis": basis,
    }


def _audit_target_gt_coverage(
    *,
    jobs: list[dict[str, Any]],
    bundle_root: Path,
    canonical_domains: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    protocol_inputs: dict[str, tuple[Path, int]] = {}
    for raw_job in jobs:
        job = _validate_job(raw_job)
        if job["domain"] != "target":
            continue
        protocol = job["protocol"].upper()
        canonical = canonical_domains.get(f"target:{protocol.lower()}")
        if canonical is None:
            raise ValueError(f"Missing canonical target protocol: {protocol}")
        manifest = (bundle_root / job["manifest_path"]).resolve()
        _require_within(manifest, bundle_root)
        if manifest != Path(canonical["source_manifest"]):
            raise ValueError(f"Target GT manifest is not canonical: {job['job_id']}")
        expected = int(canonical["expected_samples"])
        if job["expected_count"] != expected:
            raise ValueError(f"Target GT expected count mismatch: {job['job_id']}")
        previous = protocol_inputs.setdefault(protocol, (manifest, expected))
        if previous != (manifest, expected):
            raise ValueError(f"Target jobs disagree on the {protocol} GT input")
    if set(protocol_inputs) != {"VT", "VA"}:
        raise ValueError("Target GT audit requires exactly VT and VA manifests")

    protocols: dict[str, dict[str, Any]] = {}
    id_sets: dict[str, set[str]] = {}
    for protocol, (manifest, expected) in sorted(protocol_inputs.items()):
        rows = _read_jsonl(manifest)
        ids = [
            row.get("sample_id") if isinstance(row.get("sample_id"), str) else ""
            for row in rows
        ]
        nonblank_ids = [sample_id.strip() for sample_id in ids if sample_id.strip()]
        unique_ids = set(nonblank_ids)
        id_sets[protocol] = unique_ids
        nonempty_gt = sum(
            isinstance(row.get("GT_DESCRIPTION"), str)
            and bool(row["GT_DESCRIPTION"].strip())
            for row in rows
        )
        record = {
            "protocol": protocol,
            "manifest_path": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "expected_rows": expected,
            "observed_rows": len(rows),
            "unique_sample_ids": len(unique_ids),
            "blank_sample_ids": len(rows) - len(nonblank_ids),
            "duplicate_sample_ids": len(nonblank_ids) - len(unique_ids),
            "protocol_mismatches": sum(
                str(row.get("protocol", "")).upper() != protocol for row in rows
            ),
            "nonempty_gt_descriptions": nonempty_gt,
            "missing_gt_descriptions": len(rows) - nonempty_gt,
            "sample_id_set_sha256": _hash_json(sorted(unique_ids)),
        }
        record["complete"] = (
            record["observed_rows"] == expected
            and record["unique_sample_ids"] == expected
            and record["blank_sample_ids"] == 0
            and record["duplicate_sample_ids"] == 0
            and record["protocol_mismatches"] == 0
            and record["nonempty_gt_descriptions"] == expected
            and record["missing_gt_descriptions"] == 0
        )
        protocols[protocol] = record
    overlap = id_sets["VT"] & id_sets["VA"]
    complete = all(record["complete"] for record in protocols.values()) and not overlap
    return {
        "schema_name": GT_COVERAGE_SCHEMA,
        "status": "PASS" if complete else "BLOCKED",
        "required_field": "GT_DESCRIPTION",
        "protocols": protocols,
        "cross_protocol_sample_id_overlap": len(overlap),
        "expected_unique_samples": sum(
            record["expected_rows"] for record in protocols.values()
        ),
        "nonempty_gt_descriptions": sum(
            record["nonempty_gt_descriptions"] for record in protocols.values()
        ),
        "missing_gt_descriptions": sum(
            record["missing_gt_descriptions"] for record in protocols.values()
        ),
    }


def _write_blocked_gt_plan(
    *,
    destination: Path,
    config_path: Path,
    run_id: str,
    coverage: dict[str, Any],
    coverage_path: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    invalidated: list[dict[str, str]] = []
    prior_plan_path = destination / "plan.json"
    if prior_plan_path.is_file():
        prior_plan = json.loads(prior_plan_path.read_text(encoding="utf-8"))
        prior_invalidated = prior_plan.get("invalidated_pre_gate_artifacts", [])
        if isinstance(prior_invalidated, list):
            invalidated.extend(
                record
                for record in prior_invalidated
                if isinstance(record, dict)
                and set(record) == {"original", "preserved_as"}
            )
    for name in ("request_plan_ledger.jsonl", "jobs.jsonl", "jobs"):
        source = destination / name
        invalid = destination / f"{name}.pre_gt_gate.INVALID"
        if source.exists():
            if invalid.exists():
                raise FileExistsError(invalid)
            source.replace(invalid)
            record = {"original": str(source), "preserved_as": str(invalid)}
            if record not in invalidated:
                invalidated.append(record)
    plan = {
        "schema_name": PLAN_SCHEMA,
        "status": "blocked_gt_coverage",
        "run_id": run_id,
        "config_path": str(config_path.expanduser().resolve()),
        "config_sha256": _sha256(config_path),
        "api_requests_issued": 0,
        "api_key_accessed": False,
        "request_plan_ledger_path": None,
        "target_gt_coverage_audit_path": str(coverage_path),
        "target_gt_coverage_audit_sha256": _sha256(coverage_path),
        "missing_gt_descriptions": coverage["missing_gt_descriptions"],
        "invalidated_pre_gate_artifacts": invalidated,
    }
    _atomic_json(destination / "plan.json", plan)
    lines = [
        "# Cross-domain Misread preparation",
        "",
        "- Status: `BLOCKED`",
        "- Reason: canonical target `GT_DESCRIPTION` coverage is incomplete",
        f"- Non-empty GT descriptions: `{coverage['nonempty_gt_descriptions']}`",
        f"- Missing GT descriptions: `{coverage['missing_gt_descriptions']}`",
        "- API requests issued: `0`",
        "- API key accessed: `false`",
        "- Request plan: `not generated`",
        f"- Coverage audit: `{coverage_path}`",
        "",
    ]
    _atomic_bytes(destination / "RUN_STATUS.md", "\n".join(lines).encode())


def _validate_formal_cache_contract(
    *,
    cache_matrix_path: Path,
    asset_config_path: Path,
    frame_validation_path: Path,
) -> dict[str, Any]:
    cache = _load_yaml(cache_matrix_path)
    if cache.get("schema") != "mprisk_complete_cache_matrix_v2":
        raise ValueError("Misread matrix requires the canonical cache-matrix v2 config")
    repo_root = cache_matrix_path.parents[2]
    cache_asset = _resolve_path(cache.get("asset_config"), repo_root)
    if cache_asset != asset_config_path:
        raise ValueError("Misread and cache matrices must use the same asset config")
    raw_models = cache.get("models")
    if not isinstance(raw_models, list) or len(raw_models) != 16:
        raise ValueError("Canonical cache matrix must contain exactly 16 models")
    models: dict[str, dict[str, str]] = {}
    for raw in raw_models:
        if not isinstance(raw, dict):
            raise ValueError("Canonical cache model entries must be mappings")
        model_key = _required_text(raw, "model_key")
        if model_key in models:
            raise ValueError(f"Duplicate canonical cache model: {model_key}")
        protocol = _required_text(raw, "protocol").lower()
        if protocol not in {"vt", "va"}:
            raise ValueError(f"Invalid canonical model protocol: {model_key}")
        models[model_key] = {
            "protocol": protocol,
            "dtype": _required_text(raw, "dtype"),
        }
    domains: dict[str, dict[str, Any]] = {}
    raw_domains = cache.get("domains")
    if not isinstance(raw_domains, dict):
        raise ValueError("Canonical cache domains are missing")
    for domain in ("source", "target"):
        domain_value = raw_domains.get(domain)
        protocols = domain_value.get("protocols") if isinstance(domain_value, dict) else None
        if not isinstance(protocols, dict):
            raise ValueError(f"Canonical cache domain is missing: {domain}")
        for protocol in ("vt", "va"):
            value = protocols.get(protocol)
            if not isinstance(value, dict):
                raise ValueError(f"Canonical cache protocol is missing: {domain}/{protocol}")
            expected_samples = value.get("expected_samples")
            if not isinstance(expected_samples, int) or expected_samples <= 0:
                raise ValueError("Canonical expected_samples must be positive")
            domains[f"{domain}:{protocol}"] = {
                "source_manifest": str(_resolve_path(value.get("source_manifest"), repo_root)),
                "expected_samples": expected_samples,
            }
    frame_plans = cache.get("frame_plans")
    if not isinstance(frame_plans, dict) or set(frame_plans) != {"source", "target"}:
        raise ValueError("Canonical cache matrix must define source and target frame plans")
    formal_frame_paths = {
        domain: _resolve_path(value, repo_root) for domain, value in frame_plans.items()
    }
    smoke_root = _resolve_path(cache.get("smoke_root"), repo_root)
    for path in formal_frame_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        if path == smoke_root or smoke_root in path.parents:
            raise ValueError("Smoke subset frame plans cannot bind the formal Misread matrix")
    validation = json.loads(frame_validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError("Formal frame-plan validation has not passed")
    records = validation.get("records")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("Formal frame-plan validation must contain two domains")
    by_domain = {record.get("domain"): record for record in records if isinstance(record, dict)}
    if set(by_domain) != {"source", "target"}:
        raise ValueError("Formal frame-plan validation domain coverage is invalid")
    frame_contracts: dict[str, dict[str, Any]] = {}
    for domain, path in formal_frame_paths.items():
        record = by_domain[domain]
        digest = _sha256(path)
        if Path(str(record.get("path"))).expanduser().resolve() != path:
            raise ValueError(f"Formal frame-plan path mismatch: {domain}")
        if record.get("sha256") != digest:
            raise ValueError(f"Formal frame-plan SHA mismatch: {domain}")
        if (
            record.get("all_selected_lte_context") is not True
            or record.get("candidate_maxima_validated") is not True
            or record.get("sample_order_exact") is not True
            or record.get("no_truncation") is not True
        ):
            raise ValueError(f"Formal frame-plan validation is incomplete: {domain}")
        frame_contracts[domain] = {
            "path": str(path),
            "sha256": digest,
            "samples": int(record["samples"]),
        }
    return {
        "cache_matrix_config_path": str(cache_matrix_path),
        "cache_matrix_config_sha256": _sha256(cache_matrix_path),
        "asset_config_path": str(asset_config_path),
        "asset_config_sha256": _sha256(asset_config_path),
        "frame_plan_validation_path": str(frame_validation_path),
        "frame_plan_validation_sha256": _sha256(frame_validation_path),
        "frame_plans": frame_contracts,
        "models": models,
        "domains": domains,
    }


def _validate_existing_source_labels(
    *,
    judgment_root: Path,
    bundle_root: Path,
    canonical_models: dict[str, dict[str, str]],
    canonical_domains: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _require_within(judgment_root, bundle_root)
    expected = set(canonical_models) - {"phi4_multimodal"}
    phi4_path = judgment_root / "phi4_multimodal" / "judgments.jsonl"
    if phi4_path.exists():
        raise ValueError("Phi-4 source labels already exist; supplement plan is stale")
    observed = {
        path.parent.name for path in judgment_root.glob("*/judgments.jsonl") if path.is_file()
    }
    if observed != expected:
        raise ValueError(
            "Existing source-label model coverage mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    records = []
    for model_key in sorted(expected):
        protocol = canonical_models[model_key]["protocol"]
        manifest_path = Path(canonical_domains[f"source:{protocol}"]["source_manifest"])
        sample_ids = {_required_text(row, "sample_id") for row in _read_jsonl(manifest_path)}
        path = judgment_root / model_key / "judgments.jsonl"
        rows = _read_jsonl(path)
        observed_ids = {_required_text(row, "sample_id") for row in rows}
        if len(rows) != len(observed_ids) or observed_ids != sample_ids:
            raise ValueError(f"Source judgment coverage mismatch: {model_key}")
        flash_counts: Counter[int] = Counter()
        for row in rows:
            if (
                row.get("schema") != SOURCE_LABEL_SCHEMA
                or row.get("subject_model_key") != model_key
                or row.get("protocol") != protocol.upper()
                or row.get("final_label") not in {"MISREAD", "NON_MISREAD"}
            ):
                raise ValueError(f"Invalid frozen source judgment: {model_key}")
            flashes = row.get("flash")
            if not isinstance(flashes, list) or not flashes:
                raise ValueError(f"Frozen source judgment has no Flash evidence: {model_key}")
            flash_counts[len(flashes)] += 1
            for flash in flashes:
                if (
                    not isinstance(flash, dict)
                    or flash.get("judge_model") != "deepseek-v4-flash"
                    or flash.get("decision") not in {"MISREAD", "NON_MISREAD", "UNCERTAIN"}
                ):
                    raise ValueError(f"Invalid source Flash evidence: {model_key}")
        records.append(
            {
                "model_key": model_key,
                "protocol": protocol.upper(),
                "sample_count": len(rows),
                "judgment_path": str(path),
                "judgment_sha256": _sha256(path),
                "flash_replicate_distribution": {
                    str(count): frequency for count, frequency in sorted(flash_counts.items())
                },
            }
        )
    return {
        "status": "verified",
        "existing_model_count": len(records),
        "missing_model_keys": ["phi4_multimodal"],
        "root": str(judgment_root),
        "records": records,
    }


def _diagnostic_manifest_ready(
    path: Path,
    *,
    sample_ids: set[str],
    model_key: str,
    protocol: str,
    split: str,
    run_id: str,
) -> bool:
    if not path.exists():
        return False
    rows = _read_jsonl(path)
    observed_ids = {_required_text(row, "sample_id") for row in rows}
    if len(rows) != len(observed_ids) or observed_ids != sample_ids:
        raise ValueError(f"Diagnostic manifest coverage mismatch: {model_key}")
    for row in rows:
        if (
            row.get("schema_name") != "mprisk_diagnostic_affect_description_v2"
            or row.get("run_id") != run_id
            or row.get("subject_model_key") != model_key
            or row.get("protocol") != protocol
            or row.get("condition") != "M12"
            or row.get("split") != split
        ):
            raise ValueError(f"Diagnostic manifest identity mismatch: {model_key}")
        _required_text(row, "DIAGNOSTIC_AFFECT_DESCRIPTION")
    return True


def _planned_request_records(
    *,
    run_id: str,
    job_id: str,
    model_key: str,
    protocol: str,
    sample_ids: Sequence[str],
    flash_model: str,
    pro_model: str,
) -> list[dict[str, Any]]:
    records = []
    for sample_id in sample_ids:
        for role, model, slots, conditional in (
            ("flash", flash_model, range(3), False),
            ("pro", pro_model, range(1), True),
        ):
            for slot in slots:
                identity = {
                    "run_id": run_id,
                    "job_id": job_id,
                    "sample_id": sample_id,
                    "role": role,
                    "slot": slot,
                    "judge_model": model,
                }
                records.append(
                    {
                        "schema_name": REQUEST_PLAN_SCHEMA,
                        "call_id": _hash_json(identity),
                        **identity,
                        "subject_model_key": model_key,
                        "protocol": protocol,
                        "conditional": conditional,
                        "status": "planned",
                        "request_materialization_status": (
                            "awaiting_diagnostic_affect_description"
                        ),
                        "request_sha256": None,
                        "api_request_issued": False,
                    }
                )
    return records


def _resolve_path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Canonical path must be a non-empty string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the complete Misread matrix without API calls."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = prepare_matrix(args.config, destination=args.destination)
    print(
        json.dumps(
            {
                key: plan[key]
                for key in (
                    "target_model_count",
                    "job_count",
                    "pending_sample_count",
                    "existing_verified_sample_count",
                    "flash_request_count_if_complete",
                    "pro_request_upper_bound",
                    "max_api_request_count",
                    "unique_planned_call_id_count",
                    "api_requests_issued",
                    "api_key_accessed",
                )
            },
            sort_keys=True,
        )
    )
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty {key}")
    return value


def _require_within(path: Path, root: Path) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes frozen bundle root: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_bytes(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode(),
    )


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    )


def _atomic_yaml(path: Path, value: Any) -> None:
    _atomic_bytes(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode())


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
