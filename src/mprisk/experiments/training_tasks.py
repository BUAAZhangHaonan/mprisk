"""Per-run training tasks: TME/SP/TMLP training, retention sweep, TME state outputs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from mprisk.data.manifests import read_jsonl
from mprisk.evaluation.downstream_metrics import evaluate_official_representation
from mprisk.experiments._io_utils import _sha256, _training_config_path, _write_csv
from mprisk.experiments.jobs import (
    CALIBRATION,
    OFFICIAL_TEST,
    REPRESENTATIONS,
    CacheJob,
    DownstreamPlan,
)
from mprisk.experiments.relation_dataset import official_test_rows
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
