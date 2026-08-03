"""Executable, fail-closed stages for the in-domain recovery queue."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from mprisk.diagnostic_affect.generation import (
    verify_diagnostic_affect_descriptions,
)
from mprisk.diagnostic_affect.watcher import watch_description_generation
from mprisk.experiments.downstream import (
    CacheJob,
    build_relation_dataset_from_cache,
    validate_completed_cache,
)
from mprisk.judge.ensemble_misread import (
    EnsembleMisreadConfig,
    run_ensemble,
)
from mprisk.judge.ensemble_misread import (
    dry_run as dry_run_ensemble,
)
from mprisk.representation.training import (
    export_frozen_representations,
    load_training_config,
    train_trajectory_encoder,
)
from mprisk.state.patterns import load_thresholds_config
from mprisk.utils.jsonl_receipt import (
    SPHERICAL_EMBEDDING_IDENTITY_FIELDS,
    SPHERICAL_EMBEDDING_REQUIRED_FIELDS,
    read_validated_jsonl,
    write_atomic_jsonl,
)

PIPELINE_SCHEMA = "mprisk_in_domain_recovery_model_v1"
STAGES = (
    "prepare_inputs",
    "description",
    "judgment",
    "formal_judgment_intersection",
    "prepare_relation",
    "train",
    "export",
    "sdr",
    "calibrate",
    "patterns",
)


def load_pipeline_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_name") != PIPELINE_SCHEMA:
        raise ValueError(f"Unsupported recovery model config: {path}")
    for field in (
        "model_key",
        "protocol",
        "repository_root",
        "output_root",
        "legacy_assigned_manifest",
        "formal_manifest",
        "source_cache_manifest",
        "split_assignment",
        "prompt_set",
        "prompt_set_key",
        "training_config",
    ):
        _text(value, field)
    protocol = value["protocol"].lower()
    if protocol not in {"vt", "va"}:
        raise ValueError("Recovery protocol must be vt or va")
    counts = value.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Recovery counts must be a mapping")
    for field in ("diagnostic", "formal", "unmatched", "prompts"):
        if not isinstance(counts.get(field), int) or counts[field] < 0:
            raise ValueError(f"Recovery count must be nonnegative: {field}")
    if counts["formal"] <= 0 or counts["prompts"] <= 0:
        raise ValueError("Formal and prompt counts must be positive")
    hashes = value.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Recovery SHA bindings must be a mapping")
    for field in (
        "legacy_assigned_manifest",
        "formal_manifest",
        "source_cache_manifest",
        "split_assignment",
        "prompt_set",
    ):
        digest = hashes.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid SHA-256 binding: {field}")
    description_config = value.get("description_config")
    if counts["diagnostic"] > 0 and (
        not isinstance(description_config, str) or not description_config
    ):
        raise ValueError("Diagnostic recovery requires description_config")
    _description_retry_failed(value)
    reused_description = value.get("reused_description")
    if counts["diagnostic"] == 0 and not isinstance(reused_description, dict):
        raise ValueError("Non-regenerated descriptions require a reuse contract")
    config = dict(value)
    config["_config_path"] = path.expanduser().resolve()
    _validate_static_inputs(config)
    load_training_config(Path(config["training_config"]))
    return config


def dry_run_stage(config: dict[str, Any], stage: str) -> dict[str, Any]:
    _validate_stage(stage)
    required = _stage_dependencies(config, stage)
    missing = [str(path) for path in required if not path.is_file()]
    result: dict[str, Any] = {
        "stage": stage,
        "status": "ready" if not missing else "blocked_by_dependency",
        "missing_dependencies": missing,
        "would_start_gpu": False,
        "would_issue_api_requests": False,
    }
    if stage == "description":
        result.update(_description_preflight(config))
    elif stage == "judgment" and not missing:
        judge_config = _build_judgment_config(config, publish=False)
        plan = dry_run_ensemble(judge_config)
        result.update(
            {
                "sample_count": plan["sample_count"],
                "max_api_request_count": plan["max_api_request_count"],
                "api_requests_issued": 0,
                "api_key_accessed": False,
            }
        )
    elif stage == "train":
        training = load_training_config(Path(config["training_config"]))
        result["training_identity"] = {
            "model_key": training.model_key,
            "protocol": training.protocol,
            "seed": training.seed,
            "repr_key": training.repr_key,
        }
    return result


def run_stage(config: dict[str, Any], stage: str) -> dict[str, Any]:
    _validate_stage(stage)
    dependencies = _stage_dependencies(config, stage)
    missing = [str(path) for path in dependencies if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Recovery stage {stage} is missing dependencies: {missing}"
        )
    if stage == "prepare_inputs":
        return _prepare_inputs(config)
    if stage == "description":
        return _run_description(config)
    if stage == "judgment":
        return _run_judgment(config)
    if stage == "formal_judgment_intersection":
        return _formal_judgment_intersection(config)
    if stage == "prepare_relation":
        return _prepare_relation(config)
    if stage == "train":
        return _train(config)
    if stage == "export":
        return _export(config)
    if stage == "sdr":
        return _sdr(config)
    if stage == "calibrate":
        return _calibrate(config)
    if stage == "patterns":
        return _patterns(config)
    raise AssertionError(stage)


def _prepare_inputs(config: dict[str, Any]) -> dict[str, Any]:
    legacy = _read_jsonl(Path(config["legacy_assigned_manifest"]))
    formal = _read_jsonl(Path(config["formal_manifest"]))
    counts = config["counts"]
    expected_input_count = counts["diagnostic"] or counts["formal"]
    if len(legacy) != expected_input_count or len(formal) != counts["formal"]:
        raise ValueError("Frozen recovery manifest row counts do not match")
    legacy_by_id = _index(legacy)
    formal_ids = [str(row["sample_id"]) for row in formal]
    if len(formal_ids) != len(set(formal_ids)):
        raise ValueError("Formal manifest contains duplicate sample IDs")
    missing = sorted(set(formal_ids) - set(legacy_by_id))
    unmatched = sorted(set(legacy_by_id) - set(formal_ids))
    if missing or len(unmatched) != counts["unmatched"]:
        raise ValueError(
            f"Formal intersection mismatch: missing={missing}, unmatched={unmatched}"
        )
    paths = _paths(config)
    diagnostic_rows = (
        [
            dict(
                row,
                source_dataset="in_domain_recovery_20260727",
                split="recovery_all",
            )
            for row in legacy
        ]
        if counts["diagnostic"] > 0
        else []
    )
    formal_rows = [legacy_by_id[sample_id] for sample_id in formal_ids]
    gt_rows = (
        [
            {
                "schema_name": "mprisk_gt_description_v1",
                "gt_input_schema_version": "gt_annotation_input_v1",
                "sample_id": str(row["sample_id"]),
                "protocol": config["protocol"].upper(),
                "GT_DESCRIPTION": _required_row_text(row, "gt_describe"),
            }
            for row in legacy
        ]
        if counts["diagnostic"] > 0
        else []
    )
    if diagnostic_rows:
        _publish_jsonl(paths["diagnostic_manifest"], diagnostic_rows)
    _publish_jsonl(paths["formal_labels"], formal_rows)
    if gt_rows:
        _publish_jsonl(paths["gt_manifest"], gt_rows)
        gt_ids = sorted(str(row["sample_id"]) for row in gt_rows)
        gt_receipt = {
            "schema_name": "mprisk_target_gt_coverage_v1",
            "status": "PASS",
            "protocols": {
                config["protocol"].upper(): {
                    "protocol": config["protocol"].upper(),
                    "manifest_path": str(paths["gt_manifest"]),
                    "manifest_sha256": _sha256(paths["gt_manifest"]),
                    "expected_rows": len(gt_rows),
                    "observed_rows": len(gt_rows),
                    "unique_sample_ids": len(set(gt_ids)),
                    "blank_sample_ids": 0,
                    "duplicate_sample_ids": len(gt_ids) - len(set(gt_ids)),
                    "protocol_mismatches": 0,
                    "nonempty_gt_descriptions": len(gt_rows),
                    "missing_gt_descriptions": 0,
                    "sample_id_set_sha256": _hash_json(gt_ids),
                    "complete": True,
                }
            },
        }
        _publish_json(paths["gt_receipt"], gt_receipt)
    reuse_receipt = None
    if counts["diagnostic"] == 0:
        reuse_receipt = _validate_reused_descriptions(config, set(formal_ids))
        _publish_json(paths["reused_description_receipt"], reuse_receipt)
    report = {
        "schema_name": "mprisk_in_domain_formal_intersection_v1",
        "status": "PASS",
        "legacy_manifest": str(Path(config["legacy_assigned_manifest"]).resolve()),
        "legacy_manifest_sha256": config["sha256"]["legacy_assigned_manifest"],
        "legacy_rows": len(legacy),
        "formal_manifest": str(Path(config["formal_manifest"]).resolve()),
        "formal_manifest_sha256": config["sha256"]["formal_manifest"],
        "formal_rows": len(formal_rows),
        "missing_formal_ids": missing,
        "unmatched_ids": unmatched,
        "unmatched_count": len(unmatched),
        "formal_labels": str(paths["formal_labels"]),
        "formal_labels_sha256": _sha256(paths["formal_labels"]),
        "reused_description_receipt": (
            str(paths["reused_description_receipt"]) if reuse_receipt else None
        ),
    }
    _publish_json(paths["intersection_report"], report)
    return {
        "diagnostic_rows": len(diagnostic_rows),
        "formal_rows": len(formal_rows),
        "unmatched_count": len(unmatched),
    }


def _description_preflight(config: dict[str, Any]) -> dict[str, Any]:
    if config["counts"]["diagnostic"] == 0:
        return {"status": "not_required", "sample_count": 0}
    path = Path(config["description_config"])
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Description config must be a mapping")
    expected = _paths(config)["diagnostic_manifest"]
    if Path(value.get("manifest_path", "")).resolve() != expected:
        raise ValueError("Description config is not bound to recovery diagnostic manifest")
    if value.get("subject_model_key") != config["model_key"]:
        raise ValueError("Description config subject model mismatch")
    if value.get("protocol", "").lower() != config["protocol"].lower():
        raise ValueError("Description config protocol mismatch")
    return {
        "description_config": str(path.resolve()),
        "sample_count": config["counts"]["diagnostic"],
    }


def _run_description(config: dict[str, Any]) -> dict[str, Any]:
    if config["counts"]["diagnostic"] == 0:
        return {"status": "not_required"}
    result = watch_description_generation(
        config_path=Path(config["description_config"]),
        python_executable=Path(config["python_executable"]),
        python_environment=config.get("description_python_environment"),
        runtime_contract=config.get("description_runtime_contract"),
        retry_failed=_description_retry_failed(config),
        stall_timeout_seconds=float(config.get("description_stall_timeout_seconds", 1800)),
        poll_interval_seconds=30.0,
        terminate_grace_seconds=30.0,
    )
    if result != 0:
        raise RuntimeError("Description watcher did not complete cleanly")
    description_config = yaml.safe_load(
        Path(config["description_config"]).read_text(encoding="utf-8")
    )
    verification = verify_diagnostic_affect_descriptions(
        manifest_path=_paths(config)["diagnostic_manifest"],
        output_root=_paths(config)["description_manifest"].parent,
        subject_model_key=config["model_key"],
        run_id=str(description_config["run_id"]),
        protocol=config["protocol"],
        condition=str(description_config["condition"]),
        dataset=str(description_config["dataset"]),
        split=str(description_config["split"]),
        strict_full=True,
    )
    if verification["counts"][config["protocol"].upper()] != config["counts"]["diagnostic"]:
        raise RuntimeError("Description verification count mismatch")
    return {
        "status": "completed",
        "count": config["counts"]["diagnostic"],
        "verification": verification,
    }


def _description_retry_failed(config: dict[str, Any]) -> bool:
    value = config.get("description_retry_failed", False)
    if not isinstance(value, bool):
        raise ValueError("description_retry_failed must be boolean")
    return value


def _run_judgment(config: dict[str, Any]) -> dict[str, Any]:
    if config["counts"]["diagnostic"] == 0:
        return {"status": "not_required"}
    judge_config = _build_judgment_config(config, publish=True)
    summary = asyncio.run(run_ensemble(judge_config))
    if summary["unresolved"] != 0 or summary["calls_failed"] != 0:
        raise RuntimeError(f"Judgment did not complete: {summary}")
    return dict(summary)


def _build_judgment_config(
    config: dict[str, Any], *, publish: bool
) -> EnsembleMisreadConfig:
    paths = _paths(config)
    provenance = json.loads(
        paths["description_provenance"].read_text(encoding="utf-8")
    )
    signature = provenance.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("Description provenance has no immutable signature")
    payload = {
        "schema_name": "mprisk_ensemble_misread_judgment_config_v2",
        "run_id": f"{config['model_key']}_in_domain_judgment_20260727",
        "status": "ready",
        "subject_model_key": config["model_key"],
        "protocol": config["protocol"].upper(),
        "split": "recovery_all",
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "temperature": 0,
        "confidence_threshold": 0.5,
        "flash_model": "deepseek-v4-flash",
        "pro_model": "deepseek-v4-pro",
        "flash_replicates": 3,
        "gt_coverage_receipt_path": paths["gt_receipt"],
        "gt_description_manifest_path": paths["gt_manifest"],
        "diagnostic_affect_description_manifest_path": paths[
            "description_manifest"
        ],
        "diagnostic_run_id": signature["run_id"],
        "diagnostic_manifest_sha256": _sha256(paths["description_manifest"]),
        "diagnostic_prompt_sha256": signature["prompt_sha256"],
        "diagnostic_generation_policy_sha256": signature[
            "generation_policy_sha256"
        ],
        "diagnostic_request_protocol_signature_sha256": signature[
            "request_protocol_signature_sha256"
        ],
        "output_root": paths["judgment_root"],
        "request_timeout_seconds": 120.0,
        "max_concurrency": 16,
        "pricing": {
            "deepseek-v4-flash": {
                "input_usd_per_million": None,
                "output_usd_per_million": None,
            },
            "deepseek-v4-pro": {
                "input_usd_per_million": None,
                "output_usd_per_million": None,
            },
        },
    }
    parsed = EnsembleMisreadConfig.model_validate(payload)
    if publish:
        _publish_yaml(paths["judgment_config"], parsed.model_dump(mode="json"))
    return parsed


def _formal_judgment_intersection(config: dict[str, Any]) -> dict[str, Any]:
    if config["counts"]["diagnostic"] == 0:
        return {"status": "not_required"}
    paths = _paths(config)
    rows = _read_jsonl(paths["judgments"])
    formal = _read_jsonl(Path(config["formal_manifest"]))
    by_id = _index(rows)
    formal_ids = [str(row["sample_id"]) for row in formal]
    missing = sorted(set(formal_ids) - set(by_id))
    unmatched = sorted(set(by_id) - set(formal_ids))
    if missing or len(unmatched) != config["counts"]["unmatched"]:
        raise ValueError(
            f"Formal judgment intersection mismatch: missing={missing}, unmatched={unmatched}"
        )
    selected = [by_id[sample_id] for sample_id in formal_ids]
    _publish_jsonl(paths["formal_judgments"], selected)
    _publish_json(
        paths["formal_judgment_report"],
        {
            "schema_name": "mprisk_formal_judgment_intersection_v1",
            "status": "PASS",
            "input_path": str(paths["judgments"]),
            "input_sha256": _sha256(paths["judgments"]),
            "input_rows": len(rows),
            "formal_manifest_path": str(Path(config["formal_manifest"]).resolve()),
            "formal_manifest_sha256": config["sha256"]["formal_manifest"],
            "formal_rows": len(formal),
            "intersection_rows": len(selected),
            "missing_formal_ids": missing,
            "unmatched_ids": unmatched,
            "unmatched_count": len(unmatched),
            "output_path": str(paths["formal_judgments"]),
            "output_sha256": _sha256(paths["formal_judgments"]),
        },
    )
    return {"formal_rows": len(selected), "unmatched_count": len(unmatched)}


def _prepare_relation(config: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(config)
    job = CacheJob(
        seed=20260717,
        model_key=config["model_key"],
        protocol=config["protocol"],
        source_manifest=paths["formal_labels"],
        prompt_set=Path(config["prompt_set"]),
        cache_root=Path(config["source_cache_manifest"]).parent,
        expected_tasks=(
            config["counts"]["formal"] * config["counts"]["prompts"] * 3
        ),
    )
    gate = validate_completed_cache(job, verify_artifacts=True)
    dataset_path, summary_path = build_relation_dataset_from_cache(
        job,
        split_assignment_path=Path(config["split_assignment"]),
        training_config_path=Path(config["training_config"]),
        output_dir=paths["relation_root"],
        cache_gate=gate,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("sample_count") != config["counts"]["formal"]
        or summary.get("row_count")
        != config["counts"]["formal"] * config["counts"]["prompts"]
    ):
        raise RuntimeError("Formal relation dataset count mismatch")
    return {
        "samples": summary["sample_count"],
        "rows": summary["row_count"],
        "relation_dataset_sha256": _sha256(dataset_path),
        "cache_gate_sha256": _hash_json(gate),
    }


def _train(config: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(config)
    training = load_training_config(Path(config["training_config"]))
    resume = (
        paths["last_checkpoint"]
        if paths["last_checkpoint"].is_file()
        and not paths["train_metrics"].is_file()
        else None
    )
    result = train_trajectory_encoder(
        dataset_path=paths["relation_dataset"],
        config=training,
        output_dir=paths["training_root"],
        resume_checkpoint=resume,
        device=config.get("training_device", "cuda:0"),
    )
    return {
        "best_checkpoint": str(result.best_checkpoint_path),
        "best_checkpoint_sha256": _sha256(result.best_checkpoint_path),
        "resumed_from": str(resume) if resume else None,
    }


def _export(config: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(config)
    result = export_frozen_representations(
        dataset_path=paths["relation_dataset"],
        checkpoint_path=paths["best_checkpoint"],
        output_dir=paths["frozen_root"],
    )
    expected_relation_rows = config["counts"]["formal"] * config["counts"]["prompts"]
    if result.count != expected_relation_rows:
        raise RuntimeError("Frozen relation export count mismatch")
    bundle_rows = read_validated_jsonl(
        result.bundle_manifest_path,
        required_fields=SPHERICAL_EMBEDDING_REQUIRED_FIELDS,
        identity_fields=SPHERICAL_EMBEDDING_IDENTITY_FIELDS,
    )
    if len(bundle_rows) != config["counts"]["formal"]:
        raise RuntimeError("Frozen bundle count is not formal cache-closed count")
    return {
        "count": len(bundle_rows),
        "relation_rows": result.count,
        "manifest_sha256": _sha256(result.bundle_manifest_path),
        "receipt_sha256": _sha256(paths["spherical_receipt"]),
    }


def _sdr(config: dict[str, Any]) -> dict[str, Any]:
    from mprisk.state.pipeline import compute_sdr_scores

    paths = _paths(config)
    result = compute_sdr_scores(
        embedding_manifest_path=paths["spherical_manifest"],
        output_dir=paths["sdr_root"],
    )
    if result.count != config["counts"]["formal"]:
        raise RuntimeError("SDR count is not formal cache-closed count")
    return {"count": result.count, "sdr_sha256": _sha256(result.scores_path)}


def _calibrate(config: dict[str, Any]) -> dict[str, Any]:
    from mprisk.data.manifests import read_jsonl
    from mprisk.state.thresholds import calibrate_registered_aligned_thresholds

    paths = _paths(config)
    payload = calibrate_registered_aligned_thresholds(
        read_jsonl(paths["sdr_scores"]), quantile_level=0.95
    )
    _publish_json(paths["thresholds"], payload)
    load_thresholds_config(paths["thresholds"])
    return {
        "aligned_count": payload["aligned_count"],
        "thresholds_sha256": _sha256(paths["thresholds"]),
    }


def _patterns(config: dict[str, Any]) -> dict[str, Any]:
    from mprisk.state.pipeline import assign_state_patterns

    paths = _paths(config)
    result = assign_state_patterns(
        sdr_scores_path=paths["sdr_scores"],
        thresholds=paths["thresholds"],
        output_dir=paths["patterns_root"],
    )
    if result.count != config["counts"]["formal"]:
        raise RuntimeError("State-pattern count is not formal cache-closed count")
    return {"count": result.count, "patterns_sha256": _sha256(result.patterns_path)}


def _stage_dependencies(config: dict[str, Any], stage: str) -> list[Path]:
    paths = _paths(config)
    mapping = {
        "prepare_inputs": [
            Path(config["legacy_assigned_manifest"]),
            Path(config["formal_manifest"]),
        ],
        "description": [paths["diagnostic_manifest"]],
        "judgment": [
            paths["description_manifest"],
            paths["description_provenance"],
            paths["gt_manifest"],
            paths["gt_receipt"],
        ],
        "formal_judgment_intersection": [
            paths["judgments"],
            Path(config["formal_manifest"]),
        ],
        "prepare_relation": [
            paths["formal_labels"],
            Path(config["source_cache_manifest"]),
            Path(config["split_assignment"]),
            Path(config["prompt_set"]),
            Path(config["training_config"]),
        ],
        "train": [paths["relation_dataset"], Path(config["training_config"])],
        "export": [paths["relation_dataset"], paths["best_checkpoint"]],
        "sdr": [paths["spherical_manifest"], paths["spherical_receipt"]],
        "calibrate": [paths["sdr_scores"]],
        "patterns": [paths["sdr_scores"], paths["thresholds"]],
    }
    if stage == "prepare_relation":
        if config["counts"]["diagnostic"] == 0:
            mapping[stage].append(paths["reused_description_receipt"])
        elif config["counts"]["unmatched"] == 0:
            mapping[stage].extend([paths["judgments"], paths["judgment_summary"]])
        else:
            mapping[stage].extend(
                [paths["formal_judgments"], paths["formal_judgment_report"]]
            )
    return mapping[stage]


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_root"]).expanduser().resolve()
    return {
        "root": root,
        "diagnostic_manifest": root / "inputs" / "diagnostic_manifest.jsonl",
        "formal_labels": root / "inputs" / "formal_labels.jsonl",
        "gt_manifest": root / "inputs" / "gt_descriptions.jsonl",
        "gt_receipt": root / "inputs" / "gt_coverage_receipt.json",
        "intersection_report": root / "inputs" / "formal_intersection_report.json",
        "reused_description_receipt": root
        / "inputs"
        / "reused_description_receipt.json",
        "description_manifest": root / "descriptions" / "manifest.jsonl",
        "description_provenance": root / "descriptions" / "provenance.json",
        "judgment_root": root / "judgments",
        "judgment_config": root / "judgments" / "config.yaml",
        "judgments": root / "judgments" / "judgments.jsonl",
        "judgment_summary": root / "judgments" / "summary.json",
        "formal_judgments": root / "judgments" / "formal_cache_closed.jsonl",
        "formal_judgment_report": root
        / "judgments"
        / "formal_intersection_report.json",
        "relation_root": root / "relation",
        "relation_dataset": root / "relation" / "relation_dataset.jsonl",
        "training_root": root / "training",
        "best_checkpoint": root / "training" / "best_checkpoint.pt",
        "last_checkpoint": root / "training" / "last_checkpoint.pt",
        "train_metrics": root / "training" / "train_metrics.json",
        "frozen_root": root / "frozen_export",
        "spherical_manifest": root
        / "frozen_export"
        / "spherical_embedding_manifest.jsonl",
        "spherical_receipt": root
        / "frozen_export"
        / "spherical_embedding_manifest.receipt.json",
        "sdr_root": root / "sdr",
        "sdr_scores": root / "sdr" / "sdr_scores.jsonl",
        "thresholds": root / "calibration" / "thresholds.json",
        "patterns_root": root / "states",
        "state_patterns": root / "states" / "state_patterns.jsonl",
    }


def _validate_static_inputs(config: dict[str, Any]) -> None:
    for field in (
        "legacy_assigned_manifest",
        "formal_manifest",
        "source_cache_manifest",
        "split_assignment",
        "prompt_set",
    ):
        path = Path(config[field]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256(path)
        expected = config["sha256"][field]
        if observed != expected:
            raise ValueError(
                f"Static input SHA mismatch for {field}: expected {expected}, got {observed}"
            )
    reused = config.get("reused_description")
    if isinstance(reused, dict):
        path_text = reused.get("path")
        expected = reused.get("sha256")
        if not isinstance(path_text, str) or not isinstance(expected, str):
            raise ValueError("Reused description contract requires path and sha256")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                "Reused description SHA mismatch: "
                f"expected {expected}, got {observed}"
            )


def _validate_reused_descriptions(
    config: dict[str, Any], expected_ids: set[str]
) -> dict[str, Any]:
    contract = config["reused_description"]
    path = Path(contract["path"]).expanduser().resolve()
    rows = _read_jsonl(path)
    expected_rows = int(contract["rows"])
    expected_protocol = str(contract["protocol"]).upper()
    expected_model = str(contract["subject_model_key"])
    expected_schema = str(contract["schema"])
    expected_tokens = int(contract["max_new_tokens"])
    observed_ids = [str(row.get("sample_id", "")) for row in rows]
    failures = {
        "row_count": len(rows) != expected_rows,
        "sample_id_coverage": set(observed_ids) != expected_ids,
        "duplicate_sample_ids": len(observed_ids) != len(set(observed_ids)),
        "schema": any(row.get("schema") != expected_schema for row in rows),
        "subject_model_key": any(
            row.get("subject_model_key") != expected_model for row in rows
        ),
        "protocol": any(str(row.get("protocol", "")).upper() != expected_protocol for row in rows),
        "max_new_tokens": any(
            row.get("max_new_tokens") != expected_tokens for row in rows
        ),
        "blank_generated_description": any(
            not str(row.get("generated_description", "")).strip() for row in rows
        ),
    }
    failed = sorted(name for name, present in failures.items() if present)
    if failed:
        raise ValueError(f"Reused description contract failed: {failed}")
    return {
        "schema_name": "mprisk_reused_description_receipt_v1",
        "status": "PASS",
        "path": str(path),
        "sha256": _sha256(path),
        "rows": len(rows),
        "sample_id_set_sha256": _hash_json(sorted(observed_ids)),
        "generation_contract": {
            "schema": expected_schema,
            "subject_model_key": expected_model,
            "protocol": expected_protocol,
            "max_new_tokens": expected_tokens,
        },
    }


def _publish_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_atomic_jsonl(path, rows)


def _publish_json(path: Path, value: dict[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    _publish_bytes(path, encoded)


def _publish_yaml(path: Path, value: dict[str, Any]) -> None:
    _publish_bytes(
        path,
        yaml.safe_dump(value, sort_keys=True).encode("utf-8"),
    )


def _publish_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite mismatched recovery artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row.get("sample_id", "")): row for row in rows}
    if "" in result or len(result) != len(rows):
        raise ValueError("Manifest sample IDs must be non-empty and unique")
    return result


def _required_row_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field is empty: {field}")
    return value


def _text(value: dict[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise ValueError(f"Recovery field must be non-empty text: {field}")
    return item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _validate_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unsupported recovery stage: {stage}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one strict in-domain recovery stage.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pipeline_config(args.config)
    result = (
        dry_run_stage(config, args.stage)
        if args.dry_run
        else run_stage(config, args.stage)
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0
