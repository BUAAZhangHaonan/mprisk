"""Fail-closed delivery-specific TME ablation plan binding."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import mannwhitneyu, ttest_rel
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from mprisk.cache.cache_union import UNION_SCHEMA
from mprisk.representation.training import load_training_config

TEMPLATE_SCHEMA = "mprisk_delivery_representation_queue_template_v1"
RUNNABLE_SCHEMA = "mprisk_delivery_representation_queue_v1"
PENDING_CACHE_UNION = "PENDING_CACHE_UNION"
RUNNABLE = "RUNNABLE"
PARTIALLY_RUNNABLE = "PARTIALLY_RUNNABLE"
SEED = 20260717
CONDITIONS = ("M1", "M2", "M12")
METHODS = ("tme_pa_only_v1", "tme_pa_dtheta_v1")
MODEL_PROTOCOLS = {
    "qwen3_vl_8b": "vt",
    "internvl3_5_8b": "vt",
    "qwen2_5_omni_7b": "va",
}


class DeliveryPlanError(ValueError):
    """Raised when a delivery plan is incomplete or its evidence has drifted."""


@dataclass(frozen=True)
class DeliveryJob:
    model_key: str
    protocol: str
    seed: int
    run_id: str
    output_dir: Path
    state_manifest: Path
    cache_union: Path
    cache_union_sha256: str
    training_configs: dict[str, Path]


@dataclass(frozen=True)
class DeliveryPlan:
    path: Path
    output_root: Path
    lock_path: Path
    split_assignment: Path
    device: str
    max_gpu_memory_fraction: float
    selected_model_keys: tuple[str, ...]
    pending_model_keys: tuple[str, ...]
    jobs: tuple[DeliveryJob, ...]


def bind_delivery_plan(
    template_path: str | Path,
    *,
    cache_unions: dict[str, str | Path],
    output_path: str | Path,
    model_keys: set[str] | None = None,
) -> DeliveryPlan:
    """Bind completed cache unions to a pending template and atomically make it runnable."""
    template = Path(template_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output == template:
        raise DeliveryPlanError("runnable plan must not overwrite its immutable template")
    if output.exists():
        raise DeliveryPlanError(f"runnable plan already exists and is immutable: {output}")
    payload = _read_yaml(template)
    _validate_static_plan(payload, template, expect_template=True)
    selected = _normalize_model_selection(model_keys, default_to_all=True)
    if set(cache_unions) != selected:
        raise DeliveryPlanError(f"cache unions must be supplied for exactly {sorted(selected)}")
    root = _repo_root(template)
    bound = dict(payload)
    bound["schema"] = RUNNABLE_SCHEMA
    bound["status"] = RUNNABLE if selected == set(MODEL_PROTOCOLS) else PARTIALLY_RUNNABLE
    bound["selected_model_keys"] = sorted(selected)
    jobs: list[dict[str, Any]] = []
    for raw_job in payload["jobs"]:
        job = dict(raw_job)
        model_key = str(job["model_key"])
        if model_key not in selected:
            if job.get("cache_union") != PENDING_CACHE_UNION:
                raise DeliveryPlanError(f"unselected job is not pending: {model_key}")
            jobs.append(job)
            continue
        union = Path(cache_unions[model_key]).expanduser()
        if not union.is_absolute():
            union = (root / union).resolve()
        _validate_cache_union(union, job, template)
        job["cache_union"] = {
            "path": _portable_path(union, root),
            "sha256": _sha256(union),
        }
        jobs.append(job)
    bound["jobs"] = jobs
    _atomic_yaml(output, bound)
    try:
        return load_delivery_plan(output, model_keys=selected)
    except Exception:
        output.unlink(missing_ok=True)
        raise


def load_delivery_plan(
    path: str | Path, *, model_keys: set[str] | None = None
) -> DeliveryPlan:
    """Load only a fully bound plan; pending templates are deliberately non-runnable."""
    plan_path = Path(path).expanduser().resolve()
    payload = _read_yaml(plan_path)
    if payload.get("schema") == TEMPLATE_SCHEMA or payload.get("status") == PENDING_CACHE_UNION:
        raise DeliveryPlanError("cache unions are pending; bind the template before running")
    selected = _selected_model_keys_for_load(payload, requested=model_keys)
    _validate_static_plan(
        payload,
        plan_path,
        expect_template=False,
        selected_model_keys=selected,
    )
    root = _repo_root(plan_path)
    split_path = _validated_file(payload["split_assignment"], plan_path, root)
    split_rows = _read_jsonl(split_path)
    split_ids = {
        str(sample_id)
        for row in split_rows
        for sample_id in _require_list(row, "sample_ids")
    }
    jobs: list[DeliveryJob] = []
    for raw_job in payload["jobs"]:
        state_path = _validated_file(raw_job["state_manifest"], plan_path, root)
        state_rows = _read_jsonl(state_path)
        state_ids = _validate_manifest_rows(state_rows, raw_job)
        if not state_ids <= split_ids:
            missing = sorted(state_ids - split_ids)[:3]
            raise DeliveryPlanError(f"state samples missing from split assignment: {missing}")
        source_path = _validated_file(raw_job["source_manifest"], plan_path, root)
        source_rows = _read_jsonl(source_path)
        if len(source_rows) != int(raw_job["expected_counts"]["source_samples"]):
            raise DeliveryPlanError("source manifest sample count differs from the plan")
        if {str(row.get("protocol", "")).lower() for row in source_rows} != {
            str(raw_job["protocol"]).lower()
        }:
            raise DeliveryPlanError("source manifest protocol differs from the job")
        invalid = raw_job.get("invalid_assets")
        if invalid is not None:
            _validated_file(invalid, plan_path, root)
        prompt_path = _validated_file(raw_job["prompt_set"], plan_path, root)
        _validate_prompt_set(prompt_path, raw_job["prompt_set"])
        training_configs = _validate_training_configs(raw_job, plan_path, root)
        if str(raw_job["model_key"]) not in selected:
            continue
        union_spec = raw_job["cache_union"]
        union_path = _validated_file(union_spec, plan_path, root)
        _validate_cache_union(union_path, raw_job, plan_path)
        jobs.append(
            DeliveryJob(
                model_key=str(raw_job["model_key"]),
                protocol=str(raw_job["protocol"]),
                seed=int(raw_job["seed"]),
                run_id=str(raw_job["run_id"]),
                output_dir=_resolve_path(str(raw_job["output_dir"]), root),
                state_manifest=state_path,
                cache_union=union_path,
                cache_union_sha256=str(union_spec["sha256"]),
                training_configs=training_configs,
            )
        )
    return DeliveryPlan(
        path=plan_path,
        output_root=_resolve_path(str(payload["output_root"]), root),
        lock_path=_resolve_path(str(payload["lock_path"]), root),
        split_assignment=split_path,
        device=str(payload["resource_gate"]["device"]),
        max_gpu_memory_fraction=float(payload["resource_gate"]["max_gpu_memory_fraction"]),
        selected_model_keys=tuple(sorted(selected)),
        pending_model_keys=tuple(sorted(set(MODEL_PROTOCOLS) - selected)),
        jobs=tuple(jobs),
    )


def run_delivery_plan(path: str | Path, *, model_keys: set[str] | None = None) -> int:
    """Run both registered TME methods end to end from an immutable cache union."""
    import fcntl

    import torch

    from mprisk.experiments.downstream import (
        _export_tme_state_outputs,
        _train_until_converged,
    )
    from mprisk.representation.training import load_training_config
    from mprisk.utils.io import write_json

    plan = load_delivery_plan(path, model_keys=model_keys)
    if not plan.device.startswith("cuda:"):
        raise DeliveryPlanError("delivery representation training requires an explicit CUDA device")
    device_index = int(plan.device.split(":", 1)[1])
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise DeliveryPlanError(f"configured CUDA device is unavailable: {plan.device}")
    torch.cuda.set_per_process_memory_fraction(plan.max_gpu_memory_fraction, device_index)
    plan.output_root.mkdir(parents=True, exist_ok=True)
    plan.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = plan.lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise DeliveryPlanError("another delivery representation runner owns the lock") from exc
    try:
        for job in plan.jobs:
            relation_path = _materialize_relation_dataset(plan, job)
            method_outputs: dict[str, Path] = {}
            for method in METHODS:
                method_root = job.output_dir / method
                done = method_root / "RUN_COMPLETE.json"
                if done.is_file():
                    completion = json.loads(done.read_text(encoding="utf-8"))
                    scores = Path(str(completion["official_sdr_scores"]))
                    if not scores.is_file() or _sha256(scores) != completion["official_sdr_sha256"]:
                        raise DeliveryPlanError(f"stale completion marker: {done}")
                    method_outputs[method] = scores
                    continue
                config = load_training_config(job.training_configs[method])
                result = _train_until_converged(
                    dataset_path=relation_path,
                    config=config,
                    output_dir=method_root / "training",
                    device=plan.device,
                )
                official_manifest = _export_tme_state_outputs(
                    relation_path=relation_path,
                    checkpoint=result.best_checkpoint_path,
                    output_root=method_root,
                )
                scores = method_root / "official_test" / "sdr_scores.jsonl"
                patterns = method_root / "official_test" / "state_patterns.jsonl"
                geometry = _write_geometry_metrics(
                    scores_path=scores,
                    frozen_path=official_manifest,
                    checkpoint_path=result.best_checkpoint_path,
                    output_path=method_root / "official_test" / "geometry_metrics.json",
                )
                write_json(
                    done,
                    {
                        "schema": "mprisk_delivery_tme_run_complete_v1",
                        "delivery": "delivery_20260716",
                        "seed": SEED,
                        "model_key": job.model_key,
                        "method": method,
                        "training_config": str(job.training_configs[method]),
                        "training_config_sha256": _sha256(job.training_configs[method]),
                        "cache_union": str(job.cache_union),
                        "cache_union_sha256": job.cache_union_sha256,
                        "best_checkpoint": str(result.best_checkpoint_path),
                        "best_checkpoint_sha256": _sha256(result.best_checkpoint_path),
                        "official_frozen": str(official_manifest),
                        "official_frozen_sha256": _sha256(official_manifest),
                        "official_sdr_scores": str(scores),
                        "official_sdr_sha256": _sha256(scores),
                        "official_patterns": str(patterns),
                        "official_patterns_sha256": _sha256(patterns),
                        "geometry_metrics": str(geometry),
                        "geometry_metrics_sha256": _sha256(geometry),
                        "misread_labels_used": False,
                    },
                )
                method_outputs[method] = scores
            _write_paired_geometry_comparison(job, method_outputs)
        return 0
    finally:
        lock_handle.close()


def _materialize_relation_dataset(plan: DeliveryPlan, job: DeliveryJob) -> Path:
    from mprisk.data.manifests import read_final_manifest
    from mprisk.data.representation_splits import load_representation_split_assignment
    from mprisk.representation.relation_dataset import LABEL_TO_ID
    from mprisk.representation.training import load_training_config
    from mprisk.utils.io import write_json, write_jsonl

    output_root = job.output_dir / "relation"
    dataset_path = output_root / "relation_dataset.jsonl"
    summary_path = output_root / "relation_dataset_summary.json"
    if dataset_path.is_file() or summary_path.is_file():
        if not dataset_path.is_file() or not summary_path.is_file():
            raise DeliveryPlanError("partial relation materialization must be removed explicitly")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("dataset_sha256") != _sha256(dataset_path)
            or summary.get("cache_union_sha256") != job.cache_union_sha256
            or summary.get("split_assignment_sha256") != _sha256(plan.split_assignment)
        ):
            raise DeliveryPlanError("existing relation dataset is stale for the bound plan")
        return dataset_path
    union = json.loads(job.cache_union.read_text(encoding="utf-8"))
    entries = union["entries"]
    by_key = {
        (str(row["sample_id"]), str(row["prompt_id"]), str(row["condition"]).upper()): row
        for row in entries
    }
    config = load_training_config(job.training_configs["tme_pa_dtheta_v1"])
    source_rows = [
        row
        for row in read_final_manifest(job.state_manifest, protocol=job.protocol)
        if row.sample_type in LABEL_TO_ID
    ]
    assignments = load_representation_split_assignment(plan.split_assignment)
    split_sha = _sha256(plan.split_assignment)
    relation_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for source in source_rows:
        assignment = assignments.get(source.split_group_id)
        if assignment is None or source.sample_id not in set(map(str, assignment["sample_ids"])):
            raise DeliveryPlanError(f"sample absent from registered split: {source.sample_id}")
        master_split = str(assignment["master_split"])
        representation_split = str(assignment["representation_split"])
        if str(source.model_dump().get("split")) != master_split:
            raise DeliveryPlanError("source master split differs from registered assignment")
        split_counts[representation_split] += 1
        type_counts[source.sample_type] += 1
        for prompt_id in config.expected_prompt_ids:
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
                    "prompt_set_key": config.prompt_set_key,
                    "prompt_set_artifact_sha256": config.prompt_set_artifact_sha256,
                    "prompt_id": prompt_id,
                    "split_group_id": source.split_group_id,
                    "master_split": master_split,
                    "representation_split": representation_split,
                    "calibration_split": (
                        "aligned_calibration"
                        if representation_split == "aligned_calibration"
                        else ""
                    ),
                    "split_assignment_key": str(assignment["config_key"]),
                    "split_assignment_sha256": split_sha,
                    "conditions": conditions,
                }
            )
    if len(relation_rows) != len(source_rows) * config.expected_prompt_count:
        raise DeliveryPlanError("relation row count differs from synchronized prompt grid")
    write_jsonl(dataset_path, relation_rows)
    write_json(
        summary_path,
        {
            "schema": "mprisk_relation_dataset_from_cache_union_v1",
            "delivery": "delivery_20260716",
            "seed": SEED,
            "model_key": job.model_key,
            "protocol": job.protocol,
            "cache_union": str(job.cache_union),
            "cache_union_sha256": job.cache_union_sha256,
            "split_assignment": str(plan.split_assignment),
            "split_assignment_sha256": split_sha,
            "sample_count": len(source_rows),
            "row_count": len(relation_rows),
            "representation_split_counts": dict(sorted(split_counts.items())),
            "sample_type_counts": dict(sorted(type_counts.items())),
            "dataset_sha256": _sha256(dataset_path),
        },
    )
    return dataset_path


def _write_geometry_metrics(
    *, scores_path: Path, frozen_path: Path, checkpoint_path: Path, output_path: Path
) -> Path:
    from mprisk.state.spherical import spherical_center, spherical_distance
    from mprisk.utils.io import write_json

    scores = _read_jsonl(scores_path)
    bundles = {str(row["sample_id"]): row for row in _read_jsonl(frozen_path)}
    by_type: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in scores:
        sample_id = str(row["sample_id"])
        bundle = bundles[sample_id]
        prompt_ids = sorted(bundle["embeddings"]["M1"])
        centers = {
            condition: spherical_center(
                [bundle["embeddings"][condition][prompt_id] for prompt_id in prompt_ids]
            )
            for condition in ("M1", "M2")
        }
        angle_deg = math.degrees(spherical_distance(centers["M1"], centers["M2"]))
        by_type[str(row["sample_type"])].append((float(row["D"]), angle_deg))
    if set(by_type) != {"Aligned", "Conflict"}:
        raise DeliveryPlanError("official geometry requires both Aligned and Conflict samples")
    aligned = np.asarray(by_type["Aligned"], dtype=np.float64)
    conflict = np.asarray(by_type["Conflict"], dtype=np.float64)
    d_test = mannwhitneyu(conflict[:, 0], aligned[:, 0], alternative="two-sided")
    angle_test = mannwhitneyu(conflict[:, 1], aligned[:, 1], alternative="two-sided")
    relation_rows = sorted(bundles.values(), key=lambda row: str(row["sample_id"]))
    relation_features = np.asarray(
        [row["sample_relation_feature"] for row in relation_rows], dtype=np.float64
    )
    relation_labels = np.asarray(
        [0 if row["sample_type"] == "Aligned" else 1 for row in relation_rows],
        dtype=np.int64,
    )
    if relation_features.ndim != 2 or relation_features.shape[0] < 7:
        raise DeliveryPlanError("official relation_r features are invalid for clustering metrics")
    neighbors = NearestNeighbors(n_neighbors=6, metric="cosine").fit(relation_features)
    neighbor_indexes = neighbors.kneighbors(
        relation_features, n_neighbors=6, return_distance=False
    )[:, 1:]
    neighbor_purity = float(
        (relation_labels[neighbor_indexes] == relation_labels[:, None]).mean()
    )
    proxy_angle_deg = _proxy_angle_degrees(checkpoint_path)
    payload = {
        "schema": "mprisk_tme_geometry_metrics_v1",
        "split": "official_test",
        "relation_r_clustering": {
            "cosine_silhouette": float(
                silhouette_score(relation_features, relation_labels, metric="cosine")
            ),
            "five_nn_label_purity": neighbor_purity,
            "proxy_angular_separation_deg": proxy_angle_deg,
            "sample_count": int(relation_features.shape[0]),
        },
        "metrics": {
            "D": _class_gap_metrics(aligned[:, 0], conflict[:, 0], float(d_test.pvalue)),
            "split_angle_deg": _class_gap_metrics(
                aligned[:, 1], conflict[:, 1], float(angle_test.pvalue)
            ),
        },
    }
    return write_json(output_path, payload)


def _proxy_angle_degrees(checkpoint_path: Path) -> float | None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    proxy_state = checkpoint.get("proxy_state_dict")
    if not isinstance(proxy_state, dict) or "proxies" not in proxy_state:
        return None
    proxies = proxy_state["proxies"].detach().to(dtype=torch.float64)
    if tuple(proxies.shape[:1]) != (2,):
        raise DeliveryPlanError("Proxy Anchor checkpoint must contain exactly two proxies")
    proxies = torch.nn.functional.normalize(proxies, dim=1)
    cosine = float(torch.clamp(proxies[0] @ proxies[1], -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _class_gap_metrics(
    aligned: np.ndarray, conflict: np.ndarray, p_value: float
) -> dict[str, Any]:
    pooled = math.sqrt((float(aligned.var()) + float(conflict.var())) / 2.0)
    gap = float(conflict.mean() - aligned.mean())
    return {
        "aligned_count": int(aligned.size),
        "conflict_count": int(conflict.size),
        "aligned_mean": float(aligned.mean()),
        "conflict_mean": float(conflict.mean()),
        "conflict_minus_aligned": gap,
        "pooled_effect_size": gap / pooled if pooled > 1e-12 else 0.0,
        "mann_whitney_two_sided_p": p_value,
    }


def _write_paired_geometry_comparison(
    job: DeliveryJob, method_outputs: dict[str, Path]
) -> Path:
    from mprisk.utils.io import write_json

    rows = {
        method: {str(row["sample_id"]): row for row in _read_jsonl(path)}
        for method, path in method_outputs.items()
    }
    if set(rows[METHODS[0]]) != set(rows[METHODS[1]]):
        raise DeliveryPlanError("paired geometry methods have different official sample sets")
    comparison: dict[str, Any] = {}
    for sample_type in ("Aligned", "Conflict"):
        sample_ids = sorted(
            sample_id
            for sample_id, row in rows[METHODS[0]].items()
            if row["sample_type"] == sample_type
        )
        pa = np.asarray([float(rows[METHODS[0]][sample_id]["D"]) for sample_id in sample_ids])
        final = np.asarray([float(rows[METHODS[1]][sample_id]["D"]) for sample_id in sample_ids])
        test = ttest_rel(final, pa)
        comparison[sample_type] = {
            "count": len(sample_ids),
            "pa_only_D_mean": float(pa.mean()),
            "pa_dtheta_D_mean": float(final.mean()),
            "paired_D_delta_mean": float((final - pa).mean()),
            "paired_t_two_sided_p": float(test.pvalue),
        }
    comparison["class_gap"] = {
        "pa_only": (
            comparison["Conflict"]["pa_only_D_mean"]
            - comparison["Aligned"]["pa_only_D_mean"]
        ),
        "pa_dtheta": (
            comparison["Conflict"]["pa_dtheta_D_mean"]
            - comparison["Aligned"]["pa_dtheta_D_mean"]
        ),
    }
    comparison["class_gap"]["pa_dtheta_minus_pa_only"] = (
        comparison["class_gap"]["pa_dtheta"] - comparison["class_gap"]["pa_only"]
    )
    clustering = {
        method: json.loads(
            (job.output_dir / method / "official_test" / "geometry_metrics.json").read_text(
                encoding="utf-8"
            )
        )["relation_r_clustering"]
        for method in METHODS
    }
    comparison["relation_r_clustering"] = {
        "by_method": clustering,
        "pa_dtheta_minus_pa_only": {
            metric: clustering[METHODS[1]][metric] - clustering[METHODS[0]][metric]
            for metric in ("cosine_silhouette", "five_nn_label_purity")
        },
    }
    return write_json(
        job.output_dir / "paired_geometry_comparison.json",
        {
            "schema": "mprisk_tme_paired_geometry_comparison_v1",
            "delivery": "delivery_20260716",
            "seed": SEED,
            "model_key": job.model_key,
            "methods": list(METHODS),
            "misread_labels_used": False,
            "comparison": comparison,
        },
    )


def _validate_static_plan(
    payload: dict[str, Any],
    path: Path,
    *,
    expect_template: bool,
    selected_model_keys: set[str] | None = None,
) -> None:
    schema = TEMPLATE_SCHEMA if expect_template else RUNNABLE_SCHEMA
    if payload.get("schema") != schema:
        raise DeliveryPlanError(f"plan must use schema={schema}")
    if expect_template:
        if payload.get("status") != PENDING_CACHE_UNION:
            raise DeliveryPlanError(f"template status must be {PENDING_CACHE_UNION}")
        if payload.get("selected_model_keys") is not None:
            raise DeliveryPlanError("pending template must not declare selected_model_keys")
        selected = set()
    else:
        if payload.get("status") not in {RUNNABLE, PARTIALLY_RUNNABLE}:
            raise DeliveryPlanError("bound plan status is not runnable")
        if selected_model_keys is None:
            raise DeliveryPlanError("bound plan validation requires an explicit selection")
        selected = set(selected_model_keys)
    if payload.get("delivery") != "delivery_20260716" or payload.get("seed") != SEED:
        raise DeliveryPlanError("plan must bind delivery_20260716 and seed 20260717")
    root = _repo_root(path)
    for field in ("output_root", "lock_path"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise DeliveryPlanError(f"{field} must be a non-empty path")
    if "outputs/downstream/three_seed_v1" in str(payload["output_root"]).replace("\\", "/"):
        raise DeliveryPlanError("delivery runs must not reuse the legacy downstream output root")
    resource = payload.get("resource_gate")
    if not isinstance(resource, dict) or not str(resource.get("device", "")).startswith("cuda:"):
        raise DeliveryPlanError("resource_gate.device must name one CUDA device")
    memory_fraction = float(resource.get("max_gpu_memory_fraction", 0.0))
    if not 0.0 < memory_fraction < 0.9:
        raise DeliveryPlanError("max_gpu_memory_fraction must be positive and below 0.9")
    _validated_file(payload.get("split_assignment"), path, root)
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 3:
        raise DeliveryPlanError("delivery plan must contain exactly three jobs")
    mapping = {str(job.get("model_key")): str(job.get("protocol")) for job in jobs}
    if mapping != MODEL_PROTOCOLS or len(mapping) != len(jobs):
        raise DeliveryPlanError("delivery jobs must be the fixed three model/protocol pairs")
    run_ids = [str(job.get("run_id")) for job in jobs]
    output_dirs = [str(job.get("output_dir")) for job in jobs]
    if len(set(run_ids)) != 3 or len(set(output_dirs)) != 3:
        raise DeliveryPlanError("run_id and output_dir must be unique per model")
    for job in jobs:
        if job.get("seed") != SEED:
            raise DeliveryPlanError("every delivery job must use seed 20260717")
        if set(job.get("training_configs", {})) != set(METHODS):
            raise DeliveryPlanError(f"training configs must be exactly {list(METHODS)}")
        for field in ("source_manifest", "state_manifest", "prompt_set"):
            _validated_file(job.get(field), path, root)
        invalid = job.get("invalid_assets")
        if str(job["protocol"]) == "va":
            _validated_file(invalid, path, root)
        elif invalid is not None:
            raise DeliveryPlanError("VT jobs must not declare invalid VA assets")
        _validate_expected_counts(job)
        _validate_training_configs(job, path, root)
        union = job.get("cache_union")
        if expect_template:
            if union != PENDING_CACHE_UNION:
                raise DeliveryPlanError("template cache_union must be PENDING_CACHE_UNION")
        elif str(job["model_key"]) in selected:
            if not isinstance(union, dict):
                raise DeliveryPlanError("selected cache_union must bind path and sha256")
        elif union != PENDING_CACHE_UNION:
            raise DeliveryPlanError("unselected cache_union must remain PENDING_CACHE_UNION")


def _normalize_model_selection(
    model_keys: set[str] | None, *, default_to_all: bool
) -> set[str]:
    if model_keys is None:
        if default_to_all:
            return set(MODEL_PROTOCOLS)
        raise DeliveryPlanError("an explicit model selection is required")
    selected = {str(model_key) for model_key in model_keys}
    if not selected:
        raise DeliveryPlanError("model selection must not be empty")
    unknown = selected - set(MODEL_PROTOCOLS)
    if unknown:
        raise DeliveryPlanError(f"unknown model keys: {sorted(unknown)}")
    return selected


def _selected_model_keys_for_load(
    payload: dict[str, Any], *, requested: set[str] | None
) -> set[str]:
    raw_declared = payload.get("selected_model_keys")
    if not isinstance(raw_declared, list) or not raw_declared:
        raise DeliveryPlanError("bound plan must declare selected_model_keys")
    if len(raw_declared) != len(set(map(str, raw_declared))):
        raise DeliveryPlanError("selected_model_keys must be unique")
    declared = _normalize_model_selection(set(map(str, raw_declared)), default_to_all=False)
    status = payload.get("status")
    all_models = set(MODEL_PROTOCOLS)
    if status == RUNNABLE and declared != all_models:
        raise DeliveryPlanError("RUNNABLE plan must bind all model jobs")
    if status == PARTIALLY_RUNNABLE and not declared < all_models:
        raise DeliveryPlanError("PARTIALLY_RUNNABLE plan must bind a proper model subset")
    if status not in {RUNNABLE, PARTIALLY_RUNNABLE}:
        raise DeliveryPlanError("bound plan status is not runnable")
    if requested is None:
        if status == PARTIALLY_RUNNABLE:
            raise DeliveryPlanError("partial plan requires explicit --model-key selection")
        return declared
    normalized_requested = _normalize_model_selection(requested, default_to_all=False)
    if normalized_requested != declared:
        raise DeliveryPlanError(
            "runner model selection must exactly match the plan selected_model_keys"
        )
    return declared


def _validate_expected_counts(job: dict[str, Any]) -> None:
    counts = job.get("expected_counts")
    if not isinstance(counts, dict):
        raise DeliveryPlanError("expected_counts must be a mapping")
    sample_count = int(counts.get("state_samples", -1))
    labels = counts.get("sample_types")
    if (
        sample_count <= 0
        or not isinstance(labels, dict)
        or set(labels) != {"Aligned", "Conflict"}
        or sum(int(value) for value in labels.values()) != sample_count
    ):
        raise DeliveryPlanError("state sample and label counts are inconsistent")
    resolved = sample_count * int(counts.get("prompt_count", -1)) * len(CONDITIONS)
    if resolved != int(counts.get("resolved_tasks", -1)):
        raise DeliveryPlanError("resolved task count must equal samples * prompts * conditions")
    if resolved + int(counts.get("blocked_tasks", -1)) != int(counts.get("raw_tasks", -1)):
        raise DeliveryPlanError("raw task count must equal resolved plus blocked")


def _validate_manifest_rows(rows: list[dict[str, Any]], job: dict[str, Any]) -> set[str]:
    expected = job["expected_counts"]
    if len(rows) != int(expected["state_samples"]):
        raise DeliveryPlanError("state manifest sample count differs from the plan")
    ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise DeliveryPlanError("state manifest sample IDs must be non-empty and unique")
    protocol = str(job["protocol"]).upper()
    if {str(row.get("protocol", "")).upper() for row in rows} != {protocol}:
        raise DeliveryPlanError("state manifest protocol differs from the job")
    labels = Counter(str(row.get("sample_type", "")) for row in rows)
    if dict(labels) != {key: int(value) for key, value in expected["sample_types"].items()}:
        raise DeliveryPlanError("state manifest sample_type counts differ from the plan")
    return set(ids)


def _validate_prompt_set(path: Path, spec: dict[str, Any]) -> None:
    payload = _read_yaml(path)
    enabled = [
        str(row["prompt_id"])
        for row in payload.get("templates", [])
        if isinstance(row, dict) and row.get("enabled") is True
    ]
    if payload.get("key") != spec.get("key") or payload.get("protocol") != spec.get("protocol"):
        raise DeliveryPlanError("prompt-set identity differs from the plan")
    if enabled != list(spec.get("prompt_ids", [])) or len(enabled) != int(spec.get("count", -1)):
        raise DeliveryPlanError("prompt IDs/count differ from the plan")


def _validate_training_configs(
    job: dict[str, Any], plan_path: Path, root: Path
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for method, spec in job["training_configs"].items():
        config_path = _validated_file(spec, plan_path, root)
        config = load_training_config(config_path)
        if (
            config.model_key != job["model_key"]
            or config.protocol != job["protocol"]
            or config.seed != SEED
            or config.prompt_set_key != job["prompt_set"]["key"]
            or list(config.expected_prompt_ids) != list(job["prompt_set"]["prompt_ids"])
        ):
            raise DeliveryPlanError(f"training config identity differs from job: {method}")
        if method == "tme_pa_only_v1" and config.enable_state_supervision:
            raise DeliveryPlanError("PA-only config must disable state supervision")
        if method == "tme_pa_dtheta_v1" and not config.enable_state_supervision:
            raise DeliveryPlanError("PA+D/theta config must enable state supervision")
        result[method] = config_path
    return result


def _validate_cache_union(union_path: Path, job: dict[str, Any], plan_path: Path) -> None:
    if not union_path.is_file():
        raise DeliveryPlanError(f"cache union does not exist: {union_path}")
    union = json.loads(union_path.read_text(encoding="utf-8"))
    if union.get("schema") != UNION_SCHEMA:
        raise DeliveryPlanError("cache union schema is not supported")
    expected = job["expected_counts"]
    entries = union.get("entries")
    blocked = union.get("blocked_tasks")
    counts = union.get("provenance", {}).get("counts", {})
    if not isinstance(entries, list) or not isinstance(blocked, list):
        raise DeliveryPlanError("cache union entries/blocked_tasks must be lists")
    expected_tuple = (
        int(expected["resolved_tasks"]),
        int(expected["blocked_tasks"]),
        int(expected["raw_tasks"]),
    )
    actual_tuple = (len(entries), len(blocked), len(entries) + len(blocked))
    provenance_tuple = tuple(int(counts.get(key, -1)) for key in (
        "resolved_tasks", "blocked_tasks", "raw_tasks"
    ))
    if actual_tuple != expected_tuple or provenance_tuple != expected_tuple:
        raise DeliveryPlanError("cache union task counts differ from the delivery contract")
    root = _repo_root(plan_path)
    manifest_path = _resolve_spec_path(job["state_manifest"], root)
    sample_ids = {str(row["sample_id"]) for row in _read_jsonl(manifest_path)}
    prompt_ids = tuple(str(value) for value in job["prompt_set"]["prompt_ids"])
    task_grid: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    for entry in entries:
        if (
            entry.get("model_key") != job["model_key"]
            or str(entry.get("protocol", "")).lower() != job["protocol"]
            or entry.get("prompt_set_key") != job["prompt_set"]["key"]
        ):
            raise DeliveryPlanError("cache union entry identity differs from the job")
        sample_id = str(entry.get("sample_id", ""))
        prompt_id = str(entry.get("prompt_id", ""))
        condition = str(entry.get("condition", "")).upper()
        if (
            sample_id not in sample_ids
            or prompt_id not in prompt_ids
            or condition not in CONDITIONS
        ):
            raise DeliveryPlanError("cache union contains a task outside the required grid")
        task_grid[sample_id].add((prompt_id, condition))
    expected_grid = {(prompt_id, condition) for prompt_id in prompt_ids for condition in CONDITIONS}
    if set(task_grid) != sample_ids or any(grid != expected_grid for grid in task_grid.values()):
        raise DeliveryPlanError(
            "cache union does not contain the exact sample/prompt/condition grid"
        )
    signature = union.get("provenance", {}).get("expected_signature", {})
    if (
        signature.get("model_key") != job["model_key"]
        or str(signature.get("protocol", "")).lower() != job["protocol"]
        or list(signature.get("prompt_ids", [])) != list(prompt_ids)
        or signature.get("prompt_set_sha256") != job["prompt_set"]["sha256"]
        or [str(value).upper() for value in signature.get("conditions", [])]
        != list(CONDITIONS)
        or union.get("provenance", {}).get("prefill_strategy") != "full_prefill"
        or union.get("provenance", {}).get("prefill_strategy_version") != "v1"
    ):
        raise DeliveryPlanError("cache union expected signature differs from the job")
    invalid_spec = job.get("invalid_assets")
    invalid_ids = (
        {str(row["sample_id"]) for row in _read_jsonl(_resolve_spec_path(invalid_spec, root))}
        if invalid_spec is not None
        else set()
    )
    blocked_grid = {
        (
            str(row.get("sample_id", "")),
            str(row.get("prompt_id", "")),
            str(row.get("condition", "")).upper(),
        )
        for row in blocked
    }
    expected_blocked = {
        (sample_id, prompt_id, condition)
        for sample_id in invalid_ids
        for prompt_id in prompt_ids
        for condition in CONDITIONS
    }
    if blocked_grid != expected_blocked or any(
        row.get("exposed_as_cache_entry") is not False for row in blocked
    ):
        raise DeliveryPlanError("blocked cache tasks differ from invalid-asset accounting")


def _validated_file(spec: Any, plan_path: Path, root: Path) -> Path:
    if not isinstance(spec, dict) or set(spec) < {"path", "sha256"}:
        raise DeliveryPlanError("file reference must bind path and sha256")
    path = _resolve_spec_path(spec, root)
    if not path.is_file():
        raise DeliveryPlanError(f"bound file does not exist: {path}")
    digest = str(spec["sha256"])
    if len(digest) != 64 or digest != _sha256(path):
        raise DeliveryPlanError(f"sha256 mismatch for bound file: {path}")
    return path


def _resolve_spec_path(spec: dict[str, Any], root: Path) -> Path:
    return _resolve_path(str(spec["path"]), root)


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _repo_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise DeliveryPlanError(f"cannot locate repository root from {path}")


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise DeliveryPlanError(f"YAML must contain a mapping: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise DeliveryPlanError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(row)
    return rows


def _require_list(row: dict[str, Any], field: str) -> list[Any]:
    value = row.get(field)
    if not isinstance(value, list):
        raise DeliveryPlanError(f"split assignment field {field} must be a list")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
