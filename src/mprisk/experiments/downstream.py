"""Resumable, identity-locked downstream experiments for completed P=8 caches."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from mprisk.data.manifests import read_final_manifest, read_jsonl
from mprisk.data.representation_splits import load_representation_split_assignment
from mprisk.evaluation.downstream_metrics import (
    aggregate_three_seeds,
    evaluate_official_representation,
)
from mprisk.evaluation.misread_probe import write_pending_conflict_misread_probe
from mprisk.representation.relation_dataset import LABEL_TO_ID
from mprisk.representation.relation_models import TME_PROXY_ANCHOR_V1
from mprisk.representation.training import (
    export_frozen_baseline_representations,
    export_frozen_representations,
    load_training_config,
    train_trajectory_encoder,
)
from mprisk.state.pipeline import assign_state_patterns, compute_sdr_scores
from mprisk.state.thresholds import calibrate_registered_aligned_thresholds
from mprisk.utils.io import write_json, write_jsonl

PLAN_SCHEMA = "mprisk_downstream_queue_v1"
CONDITIONS = ("M1", "M2", "M12")
OFFICIAL_TEST = "official_test"
CALIBRATION = "aligned_calibration"
TRAINING_SPLITS = frozenset({"relation_train", "relation_val"})
REPRESENTATIONS = (
    "single_point_binary_v1",
    "trajectory_mlp_binary_v1",
    "tme_proxy_anchor_v1",
)


class CacheNotReady(RuntimeError):
    """A recoverable state: extraction has not completed yet."""


@dataclass(frozen=True)
class CacheJob:
    seed: int
    model_key: str
    protocol: str
    source_manifest: Path
    prompt_set: Path
    cache_root: Path
    expected_tasks: int

    @property
    def prompt_set_key(self) -> str:
        return str(_load_yaml(self.prompt_set)["key"])

    @property
    def run_key(self) -> str:
        return f"seed{self.seed}/{self.model_key}/{self.prompt_set_key}"


@dataclass(frozen=True)
class DownstreamPlan:
    repo_root: Path
    jobs: tuple[CacheJob, ...]
    split_assignment: Path
    config_root: Path
    output_root: Path
    physical_gpu: int
    device: str
    max_gpu_memory_fraction: float
    poll_seconds: int
    lock_path: Path
    retention_seed: int
    retention_fractions: tuple[float, ...]


def load_plan(path: str | Path) -> DownstreamPlan:
    plan_path = Path(path).resolve()
    root = plan_path.parents[2]
    payload = _load_yaml(plan_path)
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"downstream plan schema must be {PLAN_SCHEMA}")
    jobs = tuple(
        CacheJob(
            seed=int(row["seed"]),
            model_key=str(row["model_key"]),
            protocol=str(row["protocol"]),
            source_manifest=_resolve(root, row["source_manifest"]),
            prompt_set=_resolve(root, row["prompt_set"]),
            cache_root=_resolve(root, row["cache_root"]),
            expected_tasks=int(row["expected_tasks"]),
        )
        for row in payload.get("jobs", [])
    )
    if len(jobs) != 9 or len({job.run_key for job in jobs}) != 9:
        raise ValueError("downstream plan requires exactly three models across three seeds")
    if {job.seed for job in jobs} != {20260715, 20260716, 20260717}:
        raise ValueError("downstream plan has an unexpected prompt seed")
    resource = payload.get("resource_gate") or {}
    fraction = float(resource.get("max_gpu_memory_fraction", 0.9))
    if not 0 < fraction < 0.9:
        raise ValueError("max_gpu_memory_fraction must be strictly below 0.90")
    return DownstreamPlan(
        repo_root=root,
        jobs=jobs,
        split_assignment=_resolve(root, payload["split_assignment"]),
        config_root=_resolve(root, payload["training_config_root"]),
        output_root=_resolve(root, payload["output_root"]),
        physical_gpu=int(resource["physical_gpu"]),
        device=str(resource.get("device", "cuda:0")),
        max_gpu_memory_fraction=fraction,
        poll_seconds=int(payload.get("poll_seconds", 60)),
        lock_path=_resolve(root, payload["lock_path"]),
        retention_seed=int(payload.get("retention_seed", 20260717)),
        retention_fractions=tuple(
            float(value) for value in payload.get("retention_fractions", [0.1, 0.25, 0.5, 1.0])
        ),
    )


def validate_completed_cache(job: CacheJob, *, verify_artifacts: bool = True) -> dict[str, Any]:
    ledger = job.cache_root / "batch_state.sqlite3"
    manifest = job.cache_root / "manifest.jsonl"
    if not ledger.is_file():
        raise CacheNotReady(f"missing ledger: {ledger}")
    with sqlite3.connect(ledger) as connection:
        counts = dict(
            connection.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall()
        )
        total = sum(int(value) for value in counts.values())
        failed = int(counts.get("failed", 0))
        pending = int(counts.get("pending", 0))
        running = int(counts.get("running", 0))
        completed = int(counts.get("completed", 0))
        identities = connection.execute(
            """SELECT DISTINCT model_key,protocol,prompt_set_key FROM tasks"""
        ).fetchall()
    if failed:
        raise ValueError(f"cache ledger contains {failed} failed tasks: {ledger}")
    if total != job.expected_tasks:
        raise ValueError(f"cache ledger total {total} != expected {job.expected_tasks}")
    if pending or running or completed != job.expected_tasks:
        raise CacheNotReady(
            f"cache incomplete: completed={completed}, pending={pending}, running={running}"
        )
    if identities != [(job.model_key, job.protocol, job.prompt_set_key)]:
        raise ValueError("cache ledger identity does not match its downstream job")
    if not manifest.is_file():
        raise ValueError("completed cache ledger has no manifest.jsonl")

    prompt_payload = _load_yaml(job.prompt_set)
    expected_prompt_ids = tuple(
        str(row["prompt_id"]) for row in prompt_payload["templates"] if row.get("enabled", True)
    )
    if len(expected_prompt_ids) != 8 or len(set(expected_prompt_ids)) != 8:
        raise ValueError("downstream cache gate requires exactly eight prompt IDs")
    entries = read_jsonl(manifest)
    if len(entries) != job.expected_tasks:
        raise ValueError("completed cache manifest row count does not match the ledger")
    keys: set[tuple[str, ...]] = set()
    sample_prompt_conditions: dict[tuple[str, str], set[str]] = defaultdict(set)
    sample_prompts: dict[str, set[str]] = defaultdict(set)
    for row in entries:
        key = tuple(
            str(row.get(field, ""))
            for field in (
                "sample_id",
                "model_key",
                "protocol",
                "prompt_set_key",
                "prompt_id",
                "condition",
            )
        )
        if not all(key) or key in keys:
            raise ValueError("cache manifest contains an empty or duplicate task identity")
        keys.add(key)
        sample_id, model_key, protocol, prompt_key, prompt_id, condition = key
        if (model_key, protocol, prompt_key) != (
            job.model_key,
            job.protocol,
            job.prompt_set_key,
        ):
            raise ValueError("cache manifest identity does not match its downstream job")
        if prompt_id not in expected_prompt_ids or condition not in CONDITIONS:
            raise ValueError("cache manifest has an unregistered prompt or condition")
        checksum = str(row.get("checksum", ""))
        if len(checksum) != 64:
            raise ValueError("cache manifest entry is missing a SHA-256 checksum")
        sample_prompts[sample_id].add(prompt_id)
        sample_prompt_conditions[(sample_id, prompt_id)].add(condition)
        if verify_artifacts:
            _verify_cache_artifact(row)
    if any(prompts != set(expected_prompt_ids) for prompts in sample_prompts.values()):
        raise ValueError("cache manifest contains a seven-prompt or mismatched-prompt sample")
    if any(conditions != set(CONDITIONS) for conditions in sample_prompt_conditions.values()):
        raise ValueError("cache manifest sample/prompt does not contain exactly M1/M2/M12")
    if len(sample_prompt_conditions) != len(sample_prompts) * 8:
        raise ValueError("cache manifest is not a complete synchronized P=8 grid")
    report = {
        "schema": "mprisk_completed_cache_gate_v1",
        "status": "complete",
        "seed": job.seed,
        "model_key": job.model_key,
        "protocol": job.protocol,
        "prompt_set_key": job.prompt_set_key,
        "prompt_set_artifact_sha256": _sha256(job.prompt_set),
        "prompt_ids": list(expected_prompt_ids),
        "sample_count": len(sample_prompts),
        "task_count": len(entries),
        "ledger_counts": {
            key: int(counts.get(key, 0)) for key in ("completed", "pending", "running", "failed")
        },
        "ledger_sha256": _sha256(ledger),
        "manifest_sha256": _sha256(manifest),
        "artifacts_verified": verify_artifacts,
    }
    return report


def build_relation_dataset_from_cache(
    job: CacheJob,
    *,
    split_assignment_path: str | Path,
    training_config_path: str | Path,
    output_dir: str | Path,
    cache_gate: dict[str, Any],
) -> tuple[Path, Path]:
    config = load_training_config(training_config_path)
    if (
        config.model_key != job.model_key
        or config.protocol != job.protocol
        or config.prompt_set_key != job.prompt_set_key
    ):
        raise ValueError("training config identity does not match completed cache")
    if config.prompt_set_artifact_sha256 != _sha256(job.prompt_set):
        raise ValueError("training config prompt artifact SHA does not match the prompt YAML")
    prompt_ids = tuple(config.expected_prompt_ids)
    source_rows = [
        row
        for row in read_final_manifest(job.source_manifest, protocol=job.protocol)
        if row.sample_type in LABEL_TO_ID
    ]
    assignments = load_representation_split_assignment(split_assignment_path)
    split_sha = _sha256(Path(split_assignment_path))
    cache_rows = read_jsonl(job.cache_root / "manifest.jsonl")
    by_key = {
        (str(row["sample_id"]), str(row["prompt_id"]), str(row["condition"])): row
        for row in cache_rows
    }
    relation_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    seen_samples: set[str] = set()
    for source in source_rows:
        assignment = assignments.get(source.split_group_id)
        if assignment is None or source.sample_id not in set(map(str, assignment["sample_ids"])):
            raise ValueError(f"sample {source.sample_id} is absent from the registered split")
        master_split = str(assignment["master_split"])
        representation_split = str(assignment["representation_split"])
        if str(source.model_dump().get("split")) != master_split:
            raise ValueError("source master split disagrees with the registered assignment")
        calibration_split = CALIBRATION if representation_split == CALIBRATION else ""
        seen_samples.add(source.sample_id)
        split_counts[representation_split] += 1
        type_counts[source.sample_type] += 1
        for prompt_id in prompt_ids:
            conditions = {
                condition: by_key[(source.sample_id, prompt_id, condition)]
                for condition in CONDITIONS
            }
            relation_rows.append(
                {
                    "schema": "mprisk_relation_sample_v1",
                    "row_id": f"{source.sample_id}:{prompt_id}",
                    "sample_id": source.sample_id,
                    "sample_type": source.sample_type,
                    "label_id": LABEL_TO_ID[source.sample_type],
                    "model_key": job.model_key,
                    "protocol": job.protocol,
                    "prompt_set_key": job.prompt_set_key,
                    "prompt_set_artifact_sha256": config.prompt_set_artifact_sha256,
                    "prompt_id": prompt_id,
                    "split_group_id": source.split_group_id,
                    "master_split": master_split,
                    "representation_split": representation_split,
                    "calibration_split": calibration_split,
                    "split_assignment_key": str(assignment["config_key"]),
                    "split_assignment_sha256": split_sha,
                    "conditions": conditions,
                }
            )
    if len(seen_samples) * 8 * 3 != job.expected_tasks:
        raise ValueError("source/split sample scope does not match the complete cache grid")
    output_root = Path(output_dir)
    dataset_path = write_jsonl(output_root / "relation_dataset.jsonl", relation_rows)
    summary = {
        "schema": "mprisk_relation_dataset_from_prefill_summary_v1",
        "model_key": job.model_key,
        "protocol": job.protocol,
        "seed": job.seed,
        "prompt_set_key": job.prompt_set_key,
        "prompt_set_artifact_sha256": config.prompt_set_artifact_sha256,
        "cache_manifest_sha256": cache_gate["manifest_sha256"],
        "cache_gate_sha256": _json_sha256(cache_gate),
        "split_assignment_key": next(iter(assignments.values()))["config_key"],
        "split_assignment_sha256": split_sha,
        "sample_count": len(seen_samples),
        "row_count": len(relation_rows),
        "representation_split_counts": dict(sorted(split_counts.items())),
        "sample_type_counts": dict(sorted(type_counts.items())),
        "expected_prompt_count": 8,
        "expected_prompt_ids": list(prompt_ids),
        "dataset_sha256": _sha256(dataset_path),
    }
    summary_path = write_json(output_root / "relation_dataset_summary.json", summary)
    return dataset_path, summary_path


def official_test_rows(
    rows: Iterable[dict[str, Any]], *, source_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = list(rows)
    if not all_rows:
        raise ValueError(f"{source_name} is empty")
    split_counts = Counter(str(row.get("representation_split", "")) for row in all_rows)
    included = [row for row in all_rows if row.get("representation_split") == OFFICIAL_TEST]
    if not included:
        raise ValueError(f"{source_name} has no official_test rows")
    forbidden = [
        row
        for row in included
        if row.get("representation_split") in TRAINING_SPLITS | {CALIBRATION}
    ]
    if forbidden:
        raise ValueError("official paper inputs include training/calibration rows")
    sample_ids = [str(row["sample_id"]) for row in included]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("official paper inputs require one row per sample")
    identity = {
        "selection_rule": "representation_split=official_test",
        "source_name": source_name,
        "source_count": len(all_rows),
        "source_split_counts": dict(sorted(split_counts.items())),
        "included_count": len(included),
        "included_sample_ids_sha256": hashlib.sha256(
            json.dumps(sorted(sample_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "split_assignment_key": _one(included, "split_assignment_key"),
        "split_assignment_sha256": _one(included, "split_assignment_sha256"),
    }
    return included, identity


def run_queue(plan_path: str | Path, *, once: bool = False) -> int:
    plan = load_plan(plan_path)
    _configure_resources(plan)
    plan.output_root.mkdir(parents=True, exist_ok=True)
    write_pending_conflict_misread_probe(plan.output_root / "misread_probe")
    plan.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = plan.lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another downstream queue owns the configured lock") from exc
    while True:
        progressed = False
        ready: list[tuple[CacheJob, Path]] = []
        for job in plan.jobs:
            run_root = plan.output_root / job.run_key
            gate_path = run_root / "cache_gate.json"
            relation_path = run_root / "relation" / "relation_dataset.jsonl"
            try:
                quick_gate = validate_completed_cache(job, verify_artifacts=False)
                if gate_path.is_file():
                    gate = json.loads(gate_path.read_text(encoding="utf-8"))
                    if (
                        gate.get("manifest_sha256") != quick_gate["manifest_sha256"]
                        or gate.get("task_count") != quick_gate["task_count"]
                        or gate.get("prompt_set_artifact_sha256")
                        != quick_gate["prompt_set_artifact_sha256"]
                        or gate.get("artifacts_verified") is not True
                    ):
                        raise ValueError("persisted cache gate is stale or not fully verified")
                else:
                    gate = validate_completed_cache(job)
                    write_json(gate_path, gate)
                    progressed = True
                config_path = _training_config_path(plan, job, TME_PROXY_ANCHOR_V1)
                if not relation_path.is_file():
                    build_relation_dataset_from_cache(
                        job,
                        split_assignment_path=plan.split_assignment,
                        training_config_path=config_path,
                        output_dir=relation_path.parent,
                        cache_gate=gate,
                    )
                    progressed = True
                ready.append((job, relation_path))
            except CacheNotReady:
                continue
        if ready and _gpu_available(plan):
            for job, relation_path in ready:
                if _run_model_seed(plan, job, relation_path):
                    progressed = True
                    break
        if _aggregate_ready_models(plan):
            progressed = True
        _write_runtime_status(plan, ready)
        if _all_runs_complete(plan):
            return 0
        if once:
            return 0
        time.sleep(plan.poll_seconds if not progressed else 1)


def _run_model_seed(plan: DownstreamPlan, job: CacheJob, relation_path: Path) -> bool:
    run_root = plan.output_root / job.run_key
    for repr_key in REPRESENTATIONS:
        repr_root = run_root / repr_key
        done = repr_root / "RUN_COMPLETE.json"
        if done.is_file():
            continue
        config_path = _training_config_path(plan, job, repr_key)
        config = load_training_config(config_path)
        training_root = repr_root / "training"
        result = _train_until_converged(
            dataset_path=relation_path,
            config=config,
            output_dir=training_root,
            device=plan.device,
        )
        if repr_key == TME_PROXY_ANCHOR_V1:
            official_manifest = _export_tme_state_outputs(
                relation_path=relation_path,
                checkpoint=result.best_checkpoint_path,
                output_root=repr_root,
            )
        else:
            exported = export_frozen_baseline_representations(
                dataset_path=relation_path,
                checkpoint_path=result.best_checkpoint_path,
                output_dir=repr_root / "official_test",
                representation_split=OFFICIAL_TEST,
            )
            official_manifest = exported.manifest_path
        evaluation = evaluate_official_representation(
            manifest_path=official_manifest,
            checkpoint_path=result.best_checkpoint_path,
            output_dir=repr_root / "official_test" / "ac_evaluation",
        )
        if job.seed == plan.retention_seed:
            _run_retention_sensitivity(
                job=job,
                repr_key=repr_key,
                relation_path=relation_path,
                config_path=config_path,
                primary_metrics=Path(evaluation["metrics_path"]),
                output_root=repr_root / "conflict_retention",
                fractions=plan.retention_fractions,
                device=plan.device,
            )
        retention_complete = repr_root / "conflict_retention/RETENTION_COMPLETE.json"
        write_json(
            done,
            {
                "schema": "mprisk_downstream_run_complete_v1",
                "seed": job.seed,
                "model_key": job.model_key,
                "prompt_set_key": job.prompt_set_key,
                "repr_key": repr_key,
                "best_checkpoint": str(result.best_checkpoint_path),
                "best_checkpoint_sha256": _sha256(result.best_checkpoint_path),
                "training_metrics_sha256": _sha256(result.metrics_path),
                "official_manifest": str(official_manifest),
                "official_manifest_sha256": _sha256(official_manifest),
                "official_ac_metrics": evaluation["metrics_path"],
                "official_ac_metrics_sha256": _sha256(Path(evaluation["metrics_path"])),
                "retention_complete": (
                    str(retention_complete) if retention_complete.is_file() else None
                ),
                "retention_complete_sha256": (
                    _sha256(retention_complete) if retention_complete.is_file() else None
                ),
            },
        )
        return True
    return False


def _run_retention_sensitivity(
    *,
    job: CacheJob,
    repr_key: str,
    relation_path: Path,
    config_path: Path,
    primary_metrics: Path,
    output_root: Path,
    fractions: tuple[float, ...],
    device: str,
) -> None:
    if tuple(fractions) != (0.1, 0.25, 0.5, 1.0):
        raise ValueError("registered Conflict-retention fractions must be 0.10/0.25/0.50/1.00")
    source_rows = read_jsonl(relation_path)
    result_rows: list[dict[str, Any]] = []
    for fraction in fractions:
        fraction_key = f"{fraction:.2f}"
        fraction_root = output_root / f"fraction_{fraction_key}"
        metrics_path = fraction_root / "official_test_metrics.json"
        if fraction == 1.0:
            payload = json.loads(primary_metrics.read_text(encoding="utf-8"))
            payload = {
                **payload,
                "retained_conflict_fraction": 1.0,
                "retention_dataset": str(relation_path),
                "retention_dataset_sha256": _sha256(relation_path),
            }
            write_json(metrics_path, payload)
        elif not metrics_path.is_file():
            filtered, metadata = _retained_conflict_rows(
                source_rows,
                fraction=fraction,
                seed=job.seed,
            )
            retained_path = write_jsonl(fraction_root / "relation_dataset.jsonl", filtered)
            config = load_training_config(config_path)
            training_root = fraction_root / "training"
            training = _train_until_converged(
                dataset_path=retained_path,
                config=config,
                output_dir=training_root,
                device=device,
            )
            if repr_key == TME_PROXY_ANCHOR_V1:
                frozen = export_frozen_representations(
                    dataset_path=retained_path,
                    checkpoint_path=training.best_checkpoint_path,
                    output_dir=fraction_root / "frozen_all_registered_splits",
                )
                official, provenance = official_test_rows(
                    read_jsonl(frozen.bundle_manifest_path),
                    source_name=str(frozen.bundle_manifest_path),
                )
                feature_path = write_jsonl(fraction_root / "official_test_features.jsonl", official)
                write_json(fraction_root / "official_test_provenance.json", provenance)
            else:
                exported = export_frozen_baseline_representations(
                    dataset_path=retained_path,
                    checkpoint_path=training.best_checkpoint_path,
                    output_dir=fraction_root / "official_test",
                    representation_split=OFFICIAL_TEST,
                )
                feature_path = exported.manifest_path
            metrics = evaluate_official_representation(
                manifest_path=feature_path,
                checkpoint_path=training.best_checkpoint_path,
                output_dir=fraction_root / "evaluation",
            )
            payload = json.loads(Path(metrics["metrics_path"]).read_text(encoding="utf-8"))
            payload.update(
                {
                    "retained_conflict_fraction": fraction,
                    "retention_dataset": str(retained_path),
                    "retention_dataset_sha256": _sha256(retained_path),
                    **metadata,
                }
            )
            write_json(metrics_path, payload)
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        result_rows.append(
            {
                "model_key": job.model_key,
                "seed": job.seed,
                "repr_key": repr_key,
                "retained_conflict_fraction": fraction,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "auprc": result["auprc"],
            }
        )
    _write_csv(output_root / "conflict_retention_sensitivity.csv", result_rows)
    write_json(
        output_root / "RETENTION_COMPLETE.json",
        {
            "schema": "mprisk_conflict_retention_sensitivity_v1",
            "task": "Conflict_vs_Aligned",
            "training_policy": (
                "retain registered fractions of relation_train Conflict groups; "
                "keep all Aligned and all held-out splits"
            ),
            "fractions": list(fractions),
            "seed": job.seed,
            "repr_key": repr_key,
            "results": result_rows,
        },
    )


def _retained_conflict_rows(
    rows: list[dict[str, Any]], *, fraction: float, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conflict_groups = sorted(
        {
            str(row["split_group_id"])
            for row in rows
            if row["representation_split"] == "relation_train" and row["sample_type"] == "Conflict"
        },
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    keep_count = max(1, math.ceil(len(conflict_groups) * fraction))
    kept_groups = set(conflict_groups[:keep_count])
    retained = [
        row
        for row in rows
        if not (
            row["representation_split"] == "relation_train"
            and row["sample_type"] == "Conflict"
            and row["split_group_id"] not in kept_groups
        )
    ]
    protected_before = {
        str(row["row_id"])
        for row in rows
        if row["representation_split"] in {CALIBRATION, OFFICIAL_TEST}
    }
    retained_ids = {str(row["row_id"]) for row in retained}
    if not protected_before <= retained_ids:
        raise ValueError("Conflict retention must never alter calibration or official test rows")
    return retained, {
        "available_relation_train_conflict_groups": len(conflict_groups),
        "retained_relation_train_conflict_groups": keep_count,
        "retained_group_ids_sha256": hashlib.sha256(
            json.dumps(sorted(kept_groups), separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _train_until_converged(
    *,
    dataset_path: Path,
    config: Any,
    output_dir: Path,
    device: str,
) -> Any:
    last_checkpoint = output_dir / "last_checkpoint.pt"
    epoch_limit = config.max_epochs
    if last_checkpoint.is_file():
        previous = torch.load(last_checkpoint, map_location="cpu")
        epoch_limit = max(epoch_limit, int(previous["epoch"]) + config.max_epochs)
    current = replace(config, max_epochs=epoch_limit)
    history: list[dict[str, Any]] = []
    while True:
        result = train_trajectory_encoder(
            dataset_path=dataset_path,
            config=current,
            output_dir=output_dir,
            resume_checkpoint=last_checkpoint if last_checkpoint.is_file() else None,
            device=device,
        )
        history.append(
            {
                "max_epochs": current.max_epochs,
                "final_epoch": result.metrics["final_epoch"],
                "best_epoch": result.metrics["best_epoch"],
                "patience": current.patience,
                "min_delta": current.min_delta,
                "stop_reason": result.metrics["stop_reason"],
            }
        )
        write_json(
            output_dir / "convergence_history.json",
            {
                "schema": "mprisk_training_convergence_history_v1",
                "completion_rule": "early_stopping_only",
                "extensions": history,
            },
        )
        if result.metrics["stop_reason"] == "early_stopping":
            return result
        if result.metrics["stop_reason"] != "max_epochs":
            raise ValueError("training stopped without convergence or a registered epoch boundary")
        current = replace(current, max_epochs=current.max_epochs + config.max_epochs)


def _export_tme_state_outputs(*, relation_path: Path, checkpoint: Path, output_root: Path) -> Path:
    frozen = export_frozen_representations(
        dataset_path=relation_path,
        checkpoint_path=checkpoint,
        output_dir=output_root / "frozen_all_registered_splits",
    )
    scores = compute_sdr_scores(
        embedding_manifest_path=frozen.bundle_manifest_path,
        output_dir=output_root / "state_all_registered_splits",
    )
    all_scores = read_jsonl(scores.scores_path)
    calibration = calibrate_registered_aligned_thresholds(all_scores)
    calibration_path = write_json(output_root / "calibration" / "thresholds.json", calibration)
    patterns = assign_state_patterns(
        sdr_scores_path=scores.scores_path,
        thresholds=calibration_path,
        output_dir=output_root / "state_all_registered_splits",
    )
    official_scores, score_provenance = official_test_rows(
        all_scores, source_name=str(scores.scores_path)
    )
    official_patterns, pattern_provenance = official_test_rows(
        read_jsonl(patterns.patterns_path), source_name=str(patterns.patterns_path)
    )
    official_root = output_root / "official_test"
    official_frozen, frozen_provenance = official_test_rows(
        read_jsonl(frozen.bundle_manifest_path),
        source_name=str(frozen.bundle_manifest_path),
    )
    frozen_path = write_jsonl(official_root / "frozen_tme_representations.jsonl", official_frozen)
    score_path = write_jsonl(official_root / "sdr_scores.jsonl", official_scores)
    pattern_path = write_jsonl(official_root / "state_patterns.jsonl", official_patterns)
    calibration_ids = {
        str(row["sample_id"])
        for row in all_scores
        if row.get("representation_split") == CALIBRATION
    }
    official_ids = {str(row["sample_id"]) for row in official_scores}
    if calibration_ids & official_ids:
        raise ValueError("calibration samples leaked into official_test state outputs")
    write_json(
        official_root / "provenance.json",
        {
            "schema": "mprisk_official_test_state_provenance_v1",
            "sdr": score_provenance,
            "patterns": pattern_provenance,
            "frozen_representations": frozen_provenance,
            "calibration_selection_rule": (
                "representation_split=aligned_calibration then sample_type=Aligned"
            ),
            "calibration_count": calibration["aligned_count"],
            "calibration_sample_ids_sha256": calibration["sample_ids_sha256"],
            "calibration_artifact": str(calibration_path),
            "calibration_artifact_sha256": _sha256(calibration_path),
            "official_sdr_sha256": _sha256(score_path),
            "official_patterns_sha256": _sha256(pattern_path),
            "official_frozen_sha256": _sha256(frozen_path),
            "calibration_official_disjoint": True,
        },
    )
    return frozen_path


def _verify_cache_artifact(row: dict[str, Any]) -> None:
    root = Path(str(row["cache_root"]))
    shard = root / str(row["shard_path"])
    metadata = row.get("metadata") or {}
    sidecar = root / str(metadata.get("sidecar_path", ""))
    if not shard.is_file() or not sidecar.is_file():
        raise ValueError("cache manifest points to a missing shard/sidecar pair")
    if _sha256(shard) != row["checksum"]:
        raise ValueError(f"cache checksum mismatch: {shard}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("schema") != "mprisk_prefill_cache_sidecar_v1":
        raise ValueError("cache sidecar schema mismatch")
    if payload.get("entry") != row:
        raise ValueError("cache sidecar entry does not match manifest entry")


def _configure_resources(plan: DownstreamPlan) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(plan.physical_gpu):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must equal the configured physical_gpu before queue start"
        )
    if plan.device != "cuda:0":
        raise ValueError("a single visible physical GPU must be addressed as cuda:0")
    cpu_count = os.cpu_count() or 1
    thread_count = max(1, math.floor(cpu_count * 0.5))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(max(1, min(4, thread_count)))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("downstream queue requires exactly one mapped CUDA device")
    torch.cuda.set_per_process_memory_fraction(plan.max_gpu_memory_fraction, device=0)


def _all_runs_complete(plan: DownstreamPlan) -> bool:
    return all(
        (plan.output_root / job.run_key / repr_key / "RUN_COMPLETE.json").is_file()
        for job in plan.jobs
        for repr_key in REPRESENTATIONS
    )


def _aggregate_ready_models(plan: DownstreamPlan) -> bool:
    progressed = False
    for model_key in sorted({job.model_key for job in plan.jobs}):
        model_jobs = sorted(
            (job for job in plan.jobs if job.model_key == model_key),
            key=lambda job: job.seed,
        )
        if any(
            not (plan.output_root / job.run_key / repr_key / "RUN_COMPLETE.json").is_file()
            for job in model_jobs
            for repr_key in REPRESENTATIONS
        ):
            continue
        aggregate_root = plan.output_root / "aggregates" / model_key
        if (aggregate_root / "aggregation_provenance.json").is_file():
            continue
        runs = []
        for job in model_jobs:
            run_root = plan.output_root / job.run_key
            runs.append(
                {
                    "seed": job.seed,
                    "prompt_set_key": job.prompt_set_key,
                    "state_patterns": str(
                        run_root / TME_PROXY_ANCHOR_V1 / "official_test/state_patterns.jsonl"
                    ),
                    "state_provenance": str(
                        run_root / TME_PROXY_ANCHOR_V1 / "official_test/provenance.json"
                    ),
                    "classification_metrics": {
                        repr_key: str(
                            run_root
                            / repr_key
                            / "official_test/ac_evaluation/official_test_metrics.json"
                        )
                        for repr_key in REPRESENTATIONS
                    },
                }
            )
        aggregate_three_seeds(model_key=model_key, runs=runs, output_dir=aggregate_root)
        progressed = True
    return progressed


def _gpu_available(plan: DownstreamPlan) -> bool:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={plan.physical_gpu}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    used, total = (float(part.strip()) for part in query.split(","))
    if used / total >= plan.max_gpu_memory_fraction:
        return False
    process_output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={plan.physical_gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    processes = {
        int(line.strip()) for line in process_output.splitlines() if line.strip().isdigit()
    }
    return processes <= {os.getpid()}


def _write_runtime_status(plan: DownstreamPlan, ready: list[tuple[CacheJob, Path]]) -> None:
    completed = sum(
        (plan.output_root / job.run_key / repr_key / "RUN_COMPLETE.json").is_file()
        for job in plan.jobs
        for repr_key in REPRESENTATIONS
    )
    write_json(
        plan.output_root / "queue_status.json",
        {
            "schema": "mprisk_downstream_queue_status_v1",
            "status": "complete" if completed == 27 else "running",
            "pid": os.getpid(),
            "physical_gpu": plan.physical_gpu,
            "cache_ready_runs": len(ready),
            "cache_total_runs": len(plan.jobs),
            "completed_representation_runs": completed,
            "total_representation_runs": 27,
            "waiting_reason": (None if completed == 27 else "cache_or_exclusive_gpu_gate"),
            "updated_unix": time.time(),
        },
    )


def _training_config_path(plan: DownstreamPlan, job: CacheJob, repr_key: str) -> Path:
    path = plan.config_root / f"seed{job.seed}" / f"{job.model_key}_{repr_key}.yaml"
    if not path.is_file():
        raise ValueError(f"missing immutable training config: {path}")
    return path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def _one(rows: list[dict[str, Any]], field: str) -> str:
    values = {str(row.get(field, "")) for row in rows}
    if len(values) != 1 or not next(iter(values)):
        raise ValueError(f"official paper inputs require homogeneous {field}")
    return next(iter(values))
