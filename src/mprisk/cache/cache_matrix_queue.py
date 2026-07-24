"""Fail-closed orchestration for the complete source/target cache matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from mprisk.assets.registry import index_assets, load_model_assets
from mprisk.cache.context_window import (
    audit_smoke_cache_context,
    load_context_ceiling,
)
from mprisk.cache.integrity import (
    build_checkpoint_digest,
    build_extractor_semantic_digest,
    build_model_asset_inventory,
    completion_receipt_status,
    validate_accepted_bundle,
    write_completion_receipt,
)
from mprisk.cache.llava_v15_frame_plan import (
    CONTEXT_BUDGET_MODE,
    FRAME_PLAN_SCHEMA,
    SELECTION_CONDITIONS,
    build_frame_plan_resumable,
    load_frame_plan,
)
from mprisk.cache.llava_v15_frame_plan import (
    FRAME_PROTOCOL as LLAVA_FRAME_PROTOCOL,
)
from mprisk.config.loader import load_yaml
from mprisk.prompts.template_bank import load_equiv_prompt_set

SCHEMA = "mprisk_complete_cache_matrix_v2"
SMOKE_SCHEMA = "mprisk_cache_smoke_evidence_v2"
CONDITIONS = ("M1", "M2", "M12")
FRAME_PROTOCOL = "fixed_uniform_temporal_samples_v1"
LLAVA_MODEL_KEY = "llava_v1_5_7b"
WRAPPER_FILES = {
    "gemma3": "src/mprisk/models/gemma3.py",
    "gemma4": "src/mprisk/models/gemma4.py",
    "glm4v": "src/mprisk/models/glm4v.py",
    "internvl": "src/mprisk/models/internvl.py",
    "llava_v15": "src/mprisk/models/llava.py",
    "llava_onevision": "src/mprisk/models/llava_onevision.py",
    "minicpm_v": "src/mprisk/models/minicpm_v.py",
    "phi3_vision": "src/mprisk/models/phi3_vision.py",
    "phi4_multimodal": "src/mprisk/models/phi4_mm.py",
    "qwen2_5_vl": "src/mprisk/models/qwen2_5_vl.py",
    "qwen3_5": "src/mprisk/models/qwen3_5.py",
    "qwen_omni": "src/mprisk/models/qwen_omni.py",
    "qwen_vl": "src/mprisk/models/qwen_vl.py",
}
PROCESSOR_CONTRACT_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "audio_processor_config.json",
    "tokenizer_config.json",
    "chat_template.json",
    "chat_template.jinja",
)
FORBIDDEN_BUDGET_ARGS = frozenset(
    {
        "--max-length",
        "--max-seq-length",
        "--max-tokens",
        "--truncate",
        "--truncation",
    }
)


class GPUCapacityBusy(RuntimeError):
    """The selected GPU is currently occupied by another compute process."""


@dataclass(frozen=True)
class DomainProtocol:
    domain: str
    protocol: str
    source_manifest: Path
    prepared_manifest: Path
    media_root: Path
    source_dataset: str
    split: str
    expected_samples: int

    @property
    def expected_tasks(self) -> int:
        return self.expected_samples * 8 * len(CONDITIONS)


@dataclass(frozen=True)
class AuxiliaryPackage:
    module: str
    distribution: str


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    family: str
    protocol: str
    dtype: str
    python: Path
    python_no_user_site: bool
    env_isolation: bool
    gpu_lane: int
    trajectory_shape: tuple[int, int]
    requested_frames: int | None
    frame_protocol: str
    video_sampling_method: str
    auxiliary_packages: tuple[AuxiliaryPackage, ...]
    extra_args: tuple[str, ...]
    invalidated_domains: dict[str, str]
    accepted_bundle_domains: dict[str, dict[str, Any]]
    max_candidate_frames: int | None = None
    context_budget_mode: str | None = None

    @property
    def frame_count_argument(self) -> int:
        value = (
            self.max_candidate_frames
            if self.context_budget_mode is not None
            else self.requested_frames
        )
        if value is None:
            raise ValueError(f"{self.model_key} has no frame-count argument")
        return value

    @property
    def uses_dynamic_context(self) -> bool:
        return self.context_budget_mode is not None


@dataclass(frozen=True)
class CacheJob:
    domain: DomainProtocol
    model: ModelSpec
    output_root: Path
    smoke_evidence: Path
    frame_plan: Path | None = None

    @property
    def job_id(self) -> str:
        return f"{self.domain.domain}:{self.model.model_key}"

    @property
    def asset_signature_evidence(self) -> Path:
        return self.output_root / "ASSET_SIGNATURE.json"


@dataclass(frozen=True)
class MatrixConfig:
    source_path: Path
    repo_root: Path
    bundle_root: Path
    bundle_validation_report: Path
    bundle_inventory: Path
    asset_config: Path
    extract_script: Path
    job_runner: Path
    prompt_sets: dict[str, Path]
    frame_plans: dict[str, Path]
    domains: dict[tuple[str, str], DomainProtocol]
    models: tuple[ModelSpec, ...]
    jobs: tuple[CacheJob, ...]
    output_root: Path
    runtime_record: Path
    lock_path: Path
    tmux_session: str
    max_gpu_memory_fraction: float
    cpu_threads_per_job: int
    max_projected_filesystem_utilization: float


def load_matrix_config(path: str | Path) -> MatrixConfig:
    source_path = Path(path).expanduser().resolve()
    raw = load_yaml(source_path)
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"Matrix schema must be {SCHEMA}")
    repo_root = source_path.parents[2]

    def resolve(value: str, *, base: Path = repo_root) -> Path:
        candidate = Path(value).expanduser()
        return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    def environment_path(value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        # Do not resolve the bin/python symlink: its original path selects the venv.
        return Path(os.path.abspath(candidate))

    bundle_root = resolve(_required_str(raw, "bundle_root"))
    output_root = resolve(_required_str(raw, "output_root"))
    prompt_sets = {
        str(key).lower(): resolve(str(value))
        for key, value in _required_mapping(raw, "prompt_sets").items()
    }
    if set(prompt_sets) != {"vt", "va"}:
        raise ValueError("prompt_sets must define exactly vt and va")
    frame_plans = {
        str(key): resolve(str(value))
        for key, value in _required_mapping(raw, "frame_plans").items()
    }
    if set(frame_plans) != {"source", "target"}:
        raise ValueError("frame_plans must define exactly source and target")

    domains: dict[tuple[str, str], DomainProtocol] = {}
    for domain_name, domain_raw in _required_mapping(raw, "domains").items():
        domain_map = _mapping(domain_raw, f"domains.{domain_name}")
        for protocol, protocol_raw in _required_mapping(domain_map, "protocols").items():
            protocol_key = str(protocol).lower()
            if protocol_key not in {"vt", "va"}:
                raise ValueError(f"Unsupported protocol {protocol!r}")
            spec = _mapping(protocol_raw, f"domains.{domain_name}.protocols.{protocol}")
            key = (str(domain_name), protocol_key)
            if key in domains:
                raise ValueError(f"Duplicate domain/protocol {key}")
            domains[key] = DomainProtocol(
                domain=str(domain_name),
                protocol=protocol_key,
                source_manifest=resolve(_required_str(spec, "source_manifest")),
                prepared_manifest=resolve(_required_str(spec, "prepared_manifest")),
                media_root=resolve(_required_str(spec, "media_root")),
                source_dataset=_required_str(spec, "source_dataset"),
                split=_required_str(spec, "split"),
                expected_samples=_positive_int(spec, "expected_samples"),
            )

    environments = {
        str(key): environment_path(str(value))
        for key, value in _required_mapping(raw, "environments").items()
    }
    models: list[ModelSpec] = []
    for item in _required_list(raw, "models"):
        spec = _mapping(item, "models[]")
        environment_key = _required_str(spec, "environment")
        if environment_key not in environments:
            raise ValueError(f"Unknown environment {environment_key!r}")
        protocol = _required_str(spec, "protocol").lower()
        extra_args = tuple(str(value) for value in spec.get("extra_args", []))
        if "--video-num-segments" in extra_args:
            raise ValueError(
                "models[].requested_frames is the only frame-count contract; "
                "do not duplicate it in extra_args"
            )
        forbidden = sorted(FORBIDDEN_BUDGET_ARGS.intersection(extra_args))
        if forbidden:
            raise ValueError(
                "Shared token budgets or truncation are forbidden by the cache protocol: "
                + ", ".join(forbidden)
            )
        auxiliary_packages = tuple(
            AuxiliaryPackage(
                module=_required_str(_mapping(value, "auxiliary_packages[]"), "module"),
                distribution=_required_str(
                    _mapping(value, "auxiliary_packages[]"), "distribution"
                ),
            )
            for value in _required_list(spec, "auxiliary_packages")
        )
        if len({package.module for package in auxiliary_packages}) != len(
            auxiliary_packages
        ):
            raise ValueError("auxiliary_packages modules must be unique per model")
        is_llava_v15 = _required_str(spec, "model_key") == LLAVA_MODEL_KEY
        if is_llava_v15:
            if "requested_frames" in spec:
                raise ValueError(
                    "LLaVA-v1.5 uses max_candidate_frames, not requested_frames"
                )
            requested_frames = None
            max_candidate_frames = _positive_int(spec, "max_candidate_frames")
            context_budget_mode = _required_str(spec, "context_budget_mode")
        else:
            if "max_candidate_frames" in spec or "context_budget_mode" in spec:
                raise ValueError(
                    "Dynamic context fields are only valid for LLaVA-v1.5"
                )
            requested_frames = _positive_int(spec, "requested_frames")
            max_candidate_frames = None
            context_budget_mode = None
        model = ModelSpec(
            model_key=_required_str(spec, "model_key"),
            family=_required_str(spec, "family"),
            protocol=protocol,
            dtype=_required_str(spec, "dtype"),
            python=environments[environment_key],
            python_no_user_site=_required_bool(spec, "python_no_user_site"),
            env_isolation=_required_bool(spec, "env_isolation"),
            gpu_lane=_nonnegative_int(spec, "gpu_lane"),
            trajectory_shape=_shape(spec.get("trajectory_shape"), "trajectory_shape"),
            requested_frames=requested_frames,
            max_candidate_frames=max_candidate_frames,
            context_budget_mode=context_budget_mode,
            frame_protocol=_required_str(spec, "frame_protocol"),
            video_sampling_method=_required_str(spec, "video_sampling_method"),
            auxiliary_packages=auxiliary_packages,
            extra_args=extra_args,
            invalidated_domains={
                str(key): str(value)
                for key, value in _mapping(
                    spec.get("invalidated_domains", {}), "invalidated_domains"
                ).items()
            },
            accepted_bundle_domains={
                str(key): _mapping(value, f"accepted_bundle_domains.{key}")
                for key, value in _mapping(
                    spec.get("accepted_bundle_domains", {}), "accepted_bundle_domains"
                ).items()
            },
        )
        if protocol not in {"vt", "va"}:
            raise ValueError(f"Unsupported model protocol {protocol!r}")
        if model.dtype not in {"bfloat16", "float16"}:
            raise ValueError(f"Unsupported model dtype {model.dtype!r}")
        if model.python_no_user_site != model.env_isolation:
            raise ValueError(
                f"{model.model_key} must keep python_no_user_site and env_isolation equal"
            )
        requires_isolation = model.model_key in {"gemma4_12b", "phi4_multimodal"}
        if model.env_isolation is not requires_isolation:
            raise ValueError(
                f"{model.model_key} env_isolation must be {requires_isolation}"
            )
        if model.gpu_lane not in {0, 1}:
            raise ValueError("gpu_lane must be 0 or 1")
        if model.model_key == LLAVA_MODEL_KEY:
            if (
                model.max_candidate_frames != 8
                or model.context_budget_mode != CONTEXT_BUDGET_MODE
                or model.frame_protocol != LLAVA_FRAME_PROTOCOL
            ):
                raise ValueError(
                    "LLaVA-v1.5 requires dynamic F1..8 shared context planning"
                )
        elif model.requested_frames != 8:
            raise ValueError(
                f"{model.model_key} requires F8, got F{model.requested_frames}"
            )
        if model.model_key != LLAVA_MODEL_KEY and model.frame_protocol != FRAME_PROTOCOL:
            raise ValueError(
                f"{model.model_key} must use frame_protocol={FRAME_PROTOCOL!r}"
            )
        models.append(model)
    if len(models) != 16 or len({model.model_key for model in models}) != 16:
        raise ValueError("Complete matrix requires exactly 16 unique models")
    if Counter(model.protocol for model in models) != {"vt": 13, "va": 3}:
        raise ValueError("Complete matrix requires 13 VT and 3 VA models")

    jobs: list[CacheJob] = []
    smoke_root = resolve(_required_str(raw, "smoke_root"))
    for domain_name in ("source", "target"):
        for model in models:
            domain = domains[(domain_name, model.protocol)]
            jobs.append(
                CacheJob(
                    domain=domain,
                    model=model,
                    output_root=output_root / domain_name / model.model_key,
                    smoke_evidence=smoke_root
                    / domain_name
                    / model.model_key
                    / "SMOKE_COMPLETE.json",
                    frame_plan=(
                        frame_plans[domain_name]
                        if model.model_key == LLAVA_MODEL_KEY
                        else None
                    ),
                )
            )
    execution = _required_mapping(raw, "execution")
    memory_fraction = float(execution.get("max_gpu_memory_fraction", 0.88))
    if not 0 < memory_fraction < 0.90:
        raise ValueError("max_gpu_memory_fraction must be positive and below 0.90")
    filesystem_limit = float(execution.get("max_projected_filesystem_utilization", 0.90))
    if not 0 < filesystem_limit <= 0.90:
        raise ValueError("max_projected_filesystem_utilization must be in (0, 0.90]")
    return MatrixConfig(
        source_path=source_path,
        repo_root=repo_root,
        bundle_root=bundle_root,
        bundle_validation_report=resolve(_required_str(raw, "bundle_validation_report")),
        bundle_inventory=resolve(_required_str(raw, "bundle_inventory")),
        asset_config=resolve(_required_str(raw, "asset_config")),
        extract_script=resolve(_required_str(raw, "extract_script")),
        job_runner=resolve(_required_str(raw, "job_runner")),
        prompt_sets=prompt_sets,
        frame_plans=frame_plans,
        domains=domains,
        models=tuple(models),
        jobs=tuple(jobs),
        output_root=output_root,
        runtime_record=resolve(_required_str(raw, "runtime_record")),
        lock_path=resolve(_required_str(raw, "lock_path")),
        tmux_session=_required_str(execution, "tmux_session"),
        max_gpu_memory_fraction=memory_fraction,
        cpu_threads_per_job=_positive_int(execution, "cpu_threads_per_job"),
        max_projected_filesystem_utilization=filesystem_limit,
    )


def normalize_manifest(domain: DomainProtocol) -> tuple[list[dict[str, Any]], str]:
    if not domain.source_manifest.is_file():
        raise FileNotFoundError(domain.source_manifest)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with domain.source_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Manifest line {line_number} is not an object")
            protocol = str(row.get("protocol", "")).lower()
            if protocol != domain.protocol:
                raise ValueError(f"Manifest line {line_number} is not {domain.protocol}")
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen:
                raise ValueError(f"Invalid or duplicate sample_id at line {line_number}")
            seen.add(sample_id)
            media = row.get("media_paths")
            if not isinstance(media, dict) or not media:
                raise ValueError(f"Invalid media_paths for {sample_id}")
            normalized_media: dict[str, str] = {}
            for key, value in media.items():
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    path = domain.media_root / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Missing media for {sample_id}: {path}")
                normalized_media[str(key)] = str(path)
            normalized = dict(row)
            normalized.update(
                {
                    "protocol": domain.protocol.upper(),
                    "media_paths": normalized_media,
                    "source_dataset": str(row.get("source_dataset") or domain.source_dataset),
                    "split": str(row.get("split") or domain.split),
                    "sample_type": str(row.get("sample_type") or ""),
                    "use_in_main": bool(row.get("use_in_main", True)),
                    "annotation_count": int(row.get("annotation_count", 0)),
                }
            )
            if normalized["sample_type"] not in {"Aligned", "Conflict"}:
                raise ValueError(f"Invalid sample_type for {sample_id}")
            rows.append(normalized)
    if len(rows) != domain.expected_samples:
        raise ValueError(
            f"{domain.domain}/{domain.protocol} expected "
            f"{domain.expected_samples} rows, got {len(rows)}"
        )
    text = "".join(_canonical_json(row) + "\n" for row in rows)
    return rows, hashlib.sha256(text.encode()).hexdigest()


def prepare_manifests(config: MatrixConfig) -> dict[str, Any]:
    prepared: list[dict[str, Any]] = []
    for domain in config.domains.values():
        rows, digest = normalize_manifest(domain)
        text = "".join(_canonical_json(row) + "\n" for row in rows)
        _atomic_text(domain.prepared_manifest, text)
        prepared.append(
            {
                "domain": domain.domain,
                "protocol": domain.protocol,
                "rows": len(rows),
                "sha256": digest,
                "path": str(domain.prepared_manifest),
            }
        )
    return {"schema": "mprisk_cache_matrix_prepared_manifests_v1", "prepared": prepared}


def prepare_frame_plans(
    config: MatrixConfig, *, domains: tuple[str, ...] = ("source", "target")
) -> dict[str, Any]:
    assets = index_assets(load_model_assets(config.asset_config))
    model = next(
        (item for item in config.models if item.model_key == LLAVA_MODEL_KEY),
        None,
    )
    if model is None:
        raise KeyError(f"Missing matrix model {LLAVA_MODEL_KEY}")
    asset = assets.get(model.model_key)
    if asset is None:
        raise KeyError(f"Missing model asset {model.model_key}")
    records: list[dict[str, Any]] = []
    if not domains or any(value not in {"source", "target"} for value in domains):
        raise ValueError("Frame-plan domains must be source and/or target")
    for domain_name in domains:
        domain = config.domains[(domain_name, "vt")]
        if not domain.prepared_manifest.is_file():
            raise FileNotFoundError(
                f"Prepare normalized manifests before frame plans: {domain.prepared_manifest}"
            )
        payload = build_frame_plan_resumable(
            manifest_path=domain.prepared_manifest,
            prompt_set_path=config.prompt_sets["vt"],
            model_path=asset.local_model_path,
            model_key=model.model_key,
            output_path=config.frame_plans[domain_name],
            max_candidate_frames=model.frame_count_argument,
        )
        path = config.frame_plans[domain_name]
        selected = Counter(
            int(entry["context_budget_contract"]["selected_frames"])
            for entry in payload["entries"]
        )
        records.append(
            {
                "domain": domain_name,
                "path": str(path),
                "sha256": _sha256(path),
                "samples": len(payload["entries"]),
                "selected_frame_counts": {
                    str(key): value for key, value in sorted(selected.items())
                },
            }
        )
    return {"schema": "mprisk_llava_v15_frame_plan_preparation_v1", "plans": records}


def audit_matrix(config: MatrixConfig) -> dict[str, Any]:
    _validate_bundle(config)
    assets = index_assets(load_model_assets(config.asset_config))
    manifest_status: dict[tuple[str, str], dict[str, Any]] = {}
    for key, domain in config.domains.items():
        rows, digest = normalize_manifest(domain)
        prepared_ok = False
        if domain.prepared_manifest.is_file():
            prepared_ok = _sha256(domain.prepared_manifest) == digest
        manifest_status[key] = {
            "samples": len(rows),
            "expected_tasks": domain.expected_tasks,
            "normalized_sha256": digest,
            "prepared": prepared_ok,
            "sample_types": dict(Counter(str(row["sample_type"]) for row in rows)),
        }

    environment_checks: dict[str, dict[str, Any]] = {}
    asset_signature_checks: dict[str, dict[str, Any]] = {}
    for model in config.models:
        asset = assets.get(model.model_key)
        if asset is None:
            raise KeyError(f"Missing asset {model.model_key}")
        if asset.family != model.family:
            raise ValueError(
                f"Asset family mismatch for {model.model_key}: {asset.family} != {model.family}"
            )
        if not asset.local_model_path.is_dir():
            raise FileNotFoundError(asset.local_model_path)
        if not model.python.is_file():
            raise FileNotFoundError(model.python)
        check_key = f"{model.python}:{model.family}"
        if check_key not in environment_checks:
            environment_checks[check_key] = _check_wrapper_import(config, model)
        asset_signature_checks[model.model_key] = _asset_signature_status(config, model)

    jobs = [
        _audit_job(
            config,
            job,
            manifest_status,
            asset_signature_checks[job.model.model_key],
        )
        for job in config.jobs
    ]
    counts = Counter(str(job["status"]) for job in jobs)
    pending = [job for job in jobs if job["status"] not in {"complete", "accepted_bundle"}]
    capacity = _capacity_status(config, jobs)
    task_estimate = _task_estimate(jobs)
    ready = (
        all(job["status"] == "ready" for job in pending)
        and all(check["passed"] for check in environment_checks.values())
        and all(check["passed"] for check in asset_signature_checks.values())
        and capacity["safe"]
    )
    return {
        "schema": "mprisk_complete_cache_matrix_audit_v1",
        "status": "ready" if ready else "blocked",
        "ready_to_launch": ready,
        "models": len(config.models),
        "jobs": len(config.jobs),
        "matrix": {"vt_models": 13, "va_models": 3, "domains": ["source", "target"]},
        "manifest_status": {
            f"{domain}:{protocol}": status for (domain, protocol), status in manifest_status.items()
        },
        "environment_checks": list(environment_checks.values()),
        "asset_signature_checks": asset_signature_checks,
        "capacity": capacity,
        "task_estimate": task_estimate,
        "job_status_counts": dict(counts),
        "job_records": jobs,
    }


def _task_estimate(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    total_tasks = sum(int(job["expected_tasks"]) for job in jobs)
    completed_or_accepted_tasks = 0
    remaining_tasks = 0
    remaining_by_domain: Counter[str] = Counter()
    remaining_by_protocol: Counter[str] = Counter()
    for job in jobs:
        expected = int(job["expected_tasks"])
        if job["status"] in {"complete", "accepted_bundle"}:
            completed_or_accepted_tasks += expected
            continue
        missing = int(job.get("ledger", {}).get("missing", expected))
        completed_or_accepted_tasks += expected - missing
        remaining_tasks += missing
        remaining_by_domain[str(job["domain"])] += missing
        remaining_by_protocol[str(job["protocol"])] += missing
    if completed_or_accepted_tasks + remaining_tasks != total_tasks:
        raise ValueError("Cache task estimate does not partition the complete matrix")
    return {
        "total_tasks": total_tasks,
        "completed_or_accepted_tasks": completed_or_accepted_tasks,
        "remaining_tasks": remaining_tasks,
        "remaining_by_domain": dict(remaining_by_domain),
        "remaining_by_protocol": dict(remaining_by_protocol),
    }


def _capacity_status(config: MatrixConfig, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    projected_bytes = 0
    projected_inodes = 0
    missing_shapes: list[str] = []
    for job in jobs:
        if job["status"] in {"complete", "accepted_bundle"}:
            continue
        shape = job.get("trajectory_shape")
        if not (
            isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(value, int) and value > 0 for value in shape)
        ):
            missing_shapes.append(str(job["job_id"]))
            continue
        ledger = job.get("ledger", {})
        missing = int(ledger.get("missing", job["expected_tasks"]))
        trajectory_bytes = int(shape[0]) * int(shape[1]) * 4
        projected_bytes += missing * (trajectory_bytes + 8192)
        projected_inodes += missing * 2 + 16

    filesystem = config.output_root if config.output_root.exists() else config.output_root.parent
    filesystem.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(filesystem)
    stat = os.statvfs(filesystem)
    total_inodes = int(stat.f_files)
    free_inodes = int(stat.f_ffree)
    projected_utilization = (usage.used + projected_bytes) / usage.total
    projected_inode_utilization = (
        (total_inodes - free_inodes + projected_inodes) / total_inodes if total_inodes > 0 else 1.0
    )
    safe = (
        not missing_shapes
        and projected_utilization <= config.max_projected_filesystem_utilization
        and projected_inode_utilization <= config.max_projected_filesystem_utilization
    )
    return {
        "safe": safe,
        "filesystem": str(filesystem),
        "projected_bytes": projected_bytes,
        "projected_inodes": projected_inodes,
        "projected_utilization": projected_utilization,
        "projected_inode_utilization": projected_inode_utilization,
        "limit": config.max_projected_filesystem_utilization,
        "missing_smoke_shapes": missing_shapes,
    }


def execute_matrix(
    config: MatrixConfig,
    *,
    stage: str = "all",
    lane: int | None = None,
    wait_for_gpu: bool = False,
) -> None:
    if stage not in {"all", "source", "target"}:
        raise ValueError(f"Unsupported cache stage: {stage!r}")
    if lane not in {None, 0, 1}:
        raise ValueError(f"Unsupported GPU lane: {lane!r}")
    selected_stages = ("source", "target") if stage == "all" else (stage,)
    selected_jobs = [
        job
        for job in config.jobs
        if job.domain.domain in selected_stages
        and (lane is None or job.model.gpu_lane == lane)
    ]
    if not selected_jobs:
        raise ValueError(f"No cache jobs selected for stage={stage}, lane={lane}")
    audit = audit_matrix(config)
    selected_ids = {job.job_id for job in selected_jobs}
    blockers = [
        f"{row['job_id']}={row['status']}"
        for row in audit["job_records"]
        if row["job_id"] in selected_ids
        and row["status"] not in {"complete", "accepted_bundle", "ready"}
    ]
    if blockers or not audit["capacity"]["safe"]:
        if not audit["capacity"]["safe"]:
            blockers.append("capacity=unsafe")
        raise RuntimeError("Cache matrix is not launchable: " + ", ".join(blockers))
    lock_path, runtime_record = _scoped_execution_paths(
        config, stage=stage, lane=lane
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(lock_fd, f"{os.getpid()}\n".encode())
    os.close(lock_fd)
    try:
        for selected_stage in selected_stages:
            stage_jobs = [
                job
                for job in selected_jobs
                if job.domain.domain == selected_stage
            ]
            _execute_stage(
                config,
                stage_jobs,
                runtime_record=runtime_record,
                wait_for_gpu=wait_for_gpu,
            )
    finally:
        lock_path.unlink(missing_ok=True)


def _scoped_execution_paths(
    config: MatrixConfig, *, stage: str, lane: int | None
) -> tuple[Path, Path]:
    if stage == "all" and lane is None:
        return config.lock_path, config.runtime_record
    scope = stage + ("" if lane is None else f".gpu{lane}")
    lock_path = config.lock_path.with_name(
        f"{config.lock_path.stem}.{scope}{config.lock_path.suffix}"
    )
    runtime_record = config.runtime_record.with_name(
        f"{config.runtime_record.stem}.{scope}{config.runtime_record.suffix}"
    )
    return lock_path, runtime_record


def launch_tmux(config: MatrixConfig) -> None:
    audit = audit_matrix(config)
    if not audit["ready_to_launch"]:
        raise RuntimeError("Dry-run audit is blocked; refusing to create tmux session")
    exists = subprocess.run(
        ["tmux", "has-session", "-t", config.tmux_session],
        check=False,
        capture_output=True,
    )
    if exists.returncode == 0:
        raise RuntimeError(f"tmux session already exists: {config.tmux_session}")
    command = [
        sys.executable,
        str(config.repo_root / "scripts" / "run_cache_matrix_queue.py"),
        "--config",
        str(config.source_path),
        "--execute",
    ]
    subprocess.run(["tmux", "new-session", "-d", "-s", config.tmux_session, *command], check=True)


def _execute_stage(
    config: MatrixConfig,
    jobs: list[CacheJob],
    *,
    runtime_record: Path | None = None,
    wait_for_gpu: bool = False,
) -> None:
    selected_runtime_record = runtime_record or config.runtime_record
    queues = {lane: deque(job for job in jobs if job.model.gpu_lane == lane) for lane in (0, 1)}
    running: dict[int, tuple[CacheJob, subprocess.Popen[Any], Any]] = {}
    try:
        while any(queues.values()) or running:
            for lane in (0, 1):
                if lane in running or not queues[lane]:
                    continue
                job = queues[lane].popleft()
                status = _audit_job(config, job, {})
                if status["status"] in {"complete", "accepted_bundle"}:
                    continue
                if status["status"] != "ready":
                    raise RuntimeError(
                        f"Job became unready: {job.job_id}: {status['status']}"
                    )
                if wait_for_gpu:
                    _wait_for_gpu_capacity(lane, config.max_gpu_memory_fraction)
                else:
                    _require_gpu_capacity(lane, config.max_gpu_memory_fraction)
                signature = build_asset_signature(config, job.model)
                _write_cache_asset_signature(job, signature)
                log_path = job.output_root / "runtime.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("a", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        _job_command(config, job),
                        cwd=config.repo_root,
                        env=build_job_environment(config, job, lane),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    handle.close()
                    raise
                running[lane] = (job, process, handle)
                _write_runtime(selected_runtime_record, jobs, running)
            if not running:
                continue
            time.sleep(10)
            for lane, (job, process, handle) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                process.wait()
                handle.close()
                del running[lane]
                if return_code != 0:
                    _write_runtime(selected_runtime_record, jobs, running)
                    raise RuntimeError(
                        f"Cache extraction failed: {job.job_id}, exit={return_code}"
                    )
                status = _ledger_status(job.output_root, job.domain.expected_tasks)
                if status["status"] != "complete":
                    raise RuntimeError(
                        f"Cache extraction ended incomplete: {job.job_id}: {status}"
                    )
                write_completion_receipt(
                    job.output_root,
                    expected_signature=_expected_batch_signature(config, job),
                    expected_tasks=job.domain.expected_tasks,
                )
                _write_runtime(selected_runtime_record, jobs, running)
    finally:
        _terminate_running_processes(running)
        _write_runtime(selected_runtime_record, jobs, running)


def _terminate_running_processes(
    running: dict[int, tuple[CacheJob, subprocess.Popen[Any], Any]],
    *,
    timeout_seconds: float = 30.0,
) -> None:
    """Terminate every live lane, then wait, kill, wait, and close all handles."""
    records = list(running.items())
    for _, (_, process, _) in records:
        if process.poll() is None:
            process.terminate()
    for _, (_, process, _) in records:
        if process.poll() is None:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        else:
            process.wait()
    for lane, (_, _, handle) in records:
        handle.close()
        running.pop(lane, None)


def _audit_job(
    config: MatrixConfig,
    job: CacheJob,
    manifest_status: dict[tuple[str, str], dict[str, Any]],
    asset_signature_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "job_id": job.job_id,
        "domain": job.domain.domain,
        "model_key": job.model.model_key,
        "family": job.model.family,
        "protocol": job.model.protocol,
        "gpu_lane": job.model.gpu_lane,
        "python": str(job.model.python),
        "python_no_user_site": job.model.python_no_user_site,
        "env_isolation": job.model.env_isolation,
        "runtime_library_path": str(model_runtime_library_path(job.model)),
        "expected_samples": job.domain.expected_samples,
        "expected_tasks": job.domain.expected_tasks,
        "trajectory_shape": list(job.model.trajectory_shape),
        "requested_frames": job.model.requested_frames,
        "max_candidate_frames": job.model.max_candidate_frames,
        "context_budget_mode": job.model.context_budget_mode,
        "frame_protocol": job.model.frame_protocol,
        "video_sampling_method": job.model.video_sampling_method,
        "output_root": str(job.output_root),
    }
    invalidation = job.model.invalidated_domains.get(job.domain.domain)
    if invalidation:
        record["invalidation_reason"] = invalidation
    if asset_signature_status is None:
        asset_signature_status = _asset_signature_status(config, job.model)
    record["asset_signature"] = asset_signature_status
    ledger = _ledger_status(job.output_root, job.domain.expected_tasks)
    accepted = job.model.accepted_bundle_domains.get(job.domain.domain)
    if accepted and not invalidation:
        _validate_accepted_bundle(
            config,
            job,
            accepted,
            asset_signature=asset_signature_status["signature"],
        )
        record.update(status="accepted_bundle", accepted_bundle=accepted)
        return record
    if not asset_signature_status["passed"]:
        record.update(status="blocked_asset_signature", ledger=ledger)
        return record
    if ledger["status"] != "absent":
        cache_signature = _cache_asset_signature_status(
            job, asset_signature_status["signature"]
        )
        record["cache_asset_signature"] = cache_signature
        if not cache_signature["passed"]:
            record.update(status="blocked_cache_asset_signature", ledger=ledger)
            return record
    if ledger["status"] == "complete":
        completion = completion_receipt_status(
            job.output_root,
            expected_signature=_expected_batch_signature(config, job),
            expected_tasks=job.domain.expected_tasks,
        )
        record["completion_receipt"] = completion
        if not completion["passed"]:
            record.update(status="blocked_completion_receipt", ledger=ledger)
            return record
        record.update(status="complete", ledger=ledger)
        return record
    if ledger["status"] in {"failed", "invalid"}:
        record.update(status=ledger["status"], ledger=ledger)
        return record
    prepared = job.domain.prepared_manifest.is_file()
    if manifest_status:
        prepared = bool(manifest_status[(job.domain.domain, job.domain.protocol)]["prepared"])
    if not prepared:
        record.update(status="blocked_manifest_not_prepared", ledger=ledger)
        return record
    smoke = _smoke_status(config, job)
    if not smoke["passed"]:
        record.update(status="blocked_smoke", smoke=smoke, ledger=ledger)
        return record
    record.update(status="ready", smoke=smoke, ledger=ledger)
    return record


def _smoke_status(config: MatrixConfig, job: CacheJob) -> dict[str, Any]:
    path = job.smoke_evidence
    if not path.is_file():
        return {"passed": False, "reason": "missing", "path": str(path)}
    payload = _read_json(path)
    smoke_frame_plan = job.smoke_evidence.parent / "frame_plan.json"
    try:
        smoke_asset_signature = build_asset_signature(
            config,
            job.model,
            frame_plan_paths=(
                {job.domain.domain: smoke_frame_plan}
                if job.model.uses_dynamic_context
                else None
            ),
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        return {
            "passed": False,
            "reason": "asset_signature_failed",
            "path": str(path),
            "asset_signature": {"passed": False, "error": str(exc)},
        }
    try:
        context_ceiling = load_context_ceiling(
            family=job.model.family,
            python=job.model.python,
            model_path=str(smoke_asset_signature["model_path"]),
            expected_model_config_sha256=str(
                smoke_asset_signature["model_config_sha256"]
            ),
            environment=build_job_environment(config, job),
            cwd=config.repo_root,
        )
        audit_smoke_cache_context(
            cache_root=job.smoke_evidence.parent / "cache",
            model_key=job.model.model_key,
            context_ceiling=context_ceiling,
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "passed": False,
            "reason": "context_window_validation_failed",
            "path": str(path),
            "context_window": {"passed": False, "error": str(exc)},
        }
    expected = {
        "schema": SMOKE_SCHEMA,
        "status": "PASS",
        "model_key": job.model.model_key,
        "family": job.model.family,
        "protocol": job.model.protocol,
        "domain": job.domain.domain,
        "expected_tasks": 48,
        "completed_tasks": 48,
        "failed_tasks": 0,
        "prompt_set_sha256": _sha256(config.prompt_sets[job.model.protocol]),
        "environment_python": str(job.model.python),
        "python_no_user_site": job.model.python_no_user_site,
        "env_isolation": job.model.env_isolation,
        "runtime_library_path": str(model_runtime_library_path(job.model)),
        "dtype": job.model.dtype,
        "requested_frames": job.model.requested_frames,
        "max_candidate_frames": job.model.max_candidate_frames,
        "context_budget_mode": job.model.context_budget_mode,
        "frame_plan_sha256": (
            _sha256(smoke_frame_plan) if job.model.uses_dynamic_context else None
        ),
        "frame_protocol": job.model.frame_protocol,
        "video_sampling_method": job.model.video_sampling_method,
        "asset_signature": smoke_asset_signature,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    shape = payload.get("trajectory_shape")
    if shape != list(job.model.trajectory_shape):
        mismatches["trajectory_shape"] = {
            "expected": list(job.model.trajectory_shape),
            "actual": shape,
        }
    prompt_ids = payload.get("prompt_ids")
    if not isinstance(prompt_ids, list) or len(prompt_ids) != 8 or len(set(prompt_ids)) != 8:
        mismatches["prompt_ids"] = {"expected": "8 unique IDs", "actual": prompt_ids}
    budget_evidence = payload.get("context_budget_evidence")
    if job.model.uses_dynamic_context:
        if (
            not isinstance(budget_evidence, dict)
            or budget_evidence.get("schema")
            != "mprisk_llava_v15_context_budget_smoke_evidence_v1"
            or budget_evidence.get("frame_plan_schema") != FRAME_PLAN_SCHEMA
            or budget_evidence.get("all_token_counts_within_context") is not True
            or budget_evidence.get("no_truncation") is not True
        ):
            mismatches["context_budget_evidence"] = {
                "expected": "validated dynamic LLaVA context evidence",
                "actual": budget_evidence,
            }
    elif budget_evidence is not None:
        mismatches["context_budget_evidence"] = {
            "expected": None,
            "actual": budget_evidence,
        }
    return {
        "passed": not mismatches,
        "path": str(path),
        "mismatches": mismatches,
        "trajectory_shape": shape,
        "asset_signature": smoke_asset_signature,
    }


def _ledger_status(output_root: Path, expected_tasks: int) -> dict[str, Any]:
    path = output_root / "batch_state.sqlite3"
    if not path.is_file():
        return {"status": "absent", "completed": 0, "missing": expected_tasks}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        counts = {
            str(status): int(count)
            for status, count in connection.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status"
            ).fetchall()
        }
        total = int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
        connection.close()
    except (sqlite3.Error, OSError) as exc:
        return {"status": "invalid", "error": str(exc)}
    if total != expected_tasks:
        return {"status": "invalid", "total": total, "expected": expected_tasks}
    if counts.get("failed", 0):
        return {
            "status": "failed",
            "counts": counts,
            "missing": expected_tasks - counts.get("completed", 0),
        }
    if counts.get("completed", 0) == expected_tasks:
        return {"status": "complete", "counts": counts, "missing": 0}
    return {
        "status": "incomplete",
        "counts": counts,
        "completed": counts.get("completed", 0),
        "missing": expected_tasks - counts.get("completed", 0),
    }


def _validate_bundle(config: MatrixConfig) -> None:
    report = _read_json(config.bundle_validation_report)
    if (
        report.get("schema") != "taffc_complete_bundle_validation_v1"
        or report.get("status") != "PASS"
    ):
        raise ValueError("Canonical bundle validation report is not PASS")
    inventory = _read_json(config.bundle_inventory)
    if Path(str(inventory.get("bundle_path", ""))).resolve() != config.bundle_root:
        raise ValueError("Bundle inventory path does not match bundle_root")


def _validate_accepted_bundle(
    config: MatrixConfig,
    job: CacheJob,
    accepted: dict[str, Any],
    *,
    asset_signature: dict[str, Any],
) -> None:
    inventory = _read_json(config.bundle_inventory)
    node: Any = inventory
    pointer = _required_str(accepted, "inventory_pointer")
    for key in pointer.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing inventory pointer {pointer}")
        node = node[key]
    if not isinstance(node, dict):
        raise ValueError(f"Inventory pointer is not an object: {pointer}")
    expected = {
        "samples": job.domain.expected_samples,
        "successful_tasks": job.domain.expected_tasks,
        "protocol": job.domain.protocol.upper(),
    }
    for key, value in expected.items():
        if node.get(key) != value:
            raise ValueError(
                f"Accepted bundle mismatch for {job.job_id} {key}: {node.get(key)} != {value}"
            )
    index_path = config.bundle_root / _required_str(accepted, "index_path")
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    prompt_ids = _prompt_ids(config.prompt_sets[job.model.protocol])
    sample_ids = _manifest_sample_ids(job.domain.prepared_manifest)
    task_keys = sorted(
        [
            sample_id,
            prompt_id,
            condition,
            job.model.model_key,
            job.domain.protocol,
        ]
        for sample_id in sample_ids
        for prompt_id in prompt_ids
        for condition in CONDITIONS
    )
    waiver_value = accepted.get("equivalence_waiver")
    waiver_path = None
    if waiver_value is not None:
        waiver_path = Path(str(waiver_value)).expanduser()
        if not waiver_path.is_absolute():
            waiver_path = (config.repo_root / waiver_path).resolve()
    validate_accepted_bundle(
        index_path,
        expected_identity={
            "model_key": job.model.model_key,
            "family": job.model.family,
            "protocol": job.domain.protocol,
            "dtype": job.model.dtype,
            "manifest_sha256": _sha256(job.domain.prepared_manifest),
            "prompt_set_sha256": _sha256(config.prompt_sets[job.model.protocol]),
            "prompt_ids": list(prompt_ids),
            "conditions": list(CONDITIONS),
            "model_path": asset_signature["model_path"],
            "prefill_strategy": "full_prefill",
            "prefill_strategy_version": "v1",
            "expected_tasks": job.domain.expected_tasks,
            "task_set_sha256": hashlib.sha256(
                _canonical_json(task_keys).encode()
            ).hexdigest(),
            "model_asset_fingerprint": asset_signature[
                "model_asset_fingerprint"
            ],
            "extractor_semantic_fingerprint": asset_signature[
                "extractor_semantic_sha256"
            ],
        },
        equivalence_waiver=waiver_path,
    )


def _expected_batch_signature(
    config: MatrixConfig, job: CacheJob
) -> dict[str, Any]:
    return {
        "manifest_sha256": _sha256(job.domain.prepared_manifest),
        "prompt_set_sha256": _sha256(config.prompt_sets[job.model.protocol]),
        "prompt_ids": _prompt_ids(config.prompt_sets[job.model.protocol]),
        "protocol": job.domain.protocol,
        "conditions": list(CONDITIONS),
        "model_key": job.model.model_key,
        "family": job.model.family,
        "dtype": job.model.dtype,
        "prefill_strategy": "full_prefill",
        "prefill_strategy_version": "v1",
    }


def _prompt_ids(path: Path) -> list[str]:
    prompt_set = load_equiv_prompt_set(path)
    if not prompt_set.active:
        raise ValueError(f"Prompt set is inactive: {prompt_set.key}")
    prompt_ids = [template.prompt_id for template in prompt_set.enabled_templates()]
    if len(prompt_ids) != 8 or len(set(prompt_ids)) != 8:
        raise ValueError(f"Expected exactly 8 unique prompt IDs: {path}")
    return prompt_ids


def _manifest_sample_ids(path: Path) -> list[str]:
    sample_ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Invalid sample_id at {path}:{line_number}")
            sample_ids.append(sample_id)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return sample_ids


def _check_wrapper_import(config: MatrixConfig, model: ModelSpec) -> dict[str, Any]:
    env = build_model_environment(config, model, model.gpu_lane)
    code = (
        "from mprisk.models.wrapper_registry import get_wrapper; "
        f"w=get_wrapper({model.family!r}); "
        f"assert w.family == {model.family!r}; print(w.__name__)"
    )
    completed = subprocess.run(
        [str(model.python), "-c", code],
        cwd=config.repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "python": str(model.python),
        "python_no_user_site": model.python_no_user_site,
        "env_isolation": model.env_isolation,
        "family": model.family,
        "passed": completed.returncode == 0,
        "detail": completed.stdout.strip()
        if completed.returncode == 0
        else completed.stderr.strip(),
    }


def _asset_signature_status(config: MatrixConfig, model: ModelSpec) -> dict[str, Any]:
    try:
        signature = build_asset_signature(config, model)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        return {
            "passed": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return {"passed": True, "signature": signature}


def _cache_asset_signature_status(
    job: CacheJob, expected_signature: dict[str, Any]
) -> dict[str, Any]:
    path = job.asset_signature_evidence
    if not path.is_file():
        return {"passed": False, "reason": "missing", "path": str(path)}
    try:
        actual = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "reason": "invalid",
            "path": str(path),
            "error": str(exc),
        }
    return {
        "passed": actual == expected_signature,
        "reason": "match" if actual == expected_signature else "mismatch",
        "path": str(path),
    }


def _write_cache_asset_signature(
    job: CacheJob, signature: dict[str, Any]
) -> None:
    status = _ledger_status(job.output_root, job.domain.expected_tasks)
    if status["status"] != "absent":
        existing = _cache_asset_signature_status(job, signature)
        if not existing["passed"]:
            raise RuntimeError(
                f"Refusing to resume {job.job_id} with stale asset signature: {existing}"
            )
        return
    _atomic_text(
        job.asset_signature_evidence,
        json.dumps(signature, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def build_asset_signature(
    config: MatrixConfig,
    model: ModelSpec,
    *,
    frame_plan_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    assets = index_assets(load_model_assets(config.asset_config))
    asset = assets.get(model.model_key)
    if asset is None:
        raise KeyError(f"Missing asset {model.model_key}")
    model_path = asset.local_model_path.resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    processor_files = {
        name: _sha256(model_path / name)
        for name in PROCESSOR_CONTRACT_FILES
        if (model_path / name).is_file()
    }
    if not processor_files:
        raise FileNotFoundError(
            f"No processor contract files found under model asset {model_path}"
        )
    processor_contract_sha256 = hashlib.sha256(
        _canonical_json(processor_files).encode()
    ).hexdigest()
    checkpoint_receipt_path = (
        config.output_root
        / "receipts"
        / "checkpoints"
        / f"{model.model_key}.json"
    )
    checkpoint = build_checkpoint_digest(
        model_path,
        receipt_path=checkpoint_receipt_path,
    )
    extractor = build_extractor_semantic_digest(
        config.repo_root,
        family=model.family,
        model_path=model_path,
    )
    model_asset = build_model_asset_inventory(
        model_path,
        checkpoint_receipt=checkpoint,
    )

    wrapper_relative = WRAPPER_FILES.get(model.family)
    if wrapper_relative is None:
        raise KeyError(f"No wrapper file registered for family {model.family!r}")
    wrapper_path = config.repo_root / wrapper_relative
    if not wrapper_path.is_file():
        raise FileNotFoundError(wrapper_path)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", wrapper_relative],
        cwd=config.repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            f"Wrapper must be committed before smoke evidence is valid: {wrapper_relative}"
        )
    wrapper_git_sha = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", wrapper_relative],
        cwd=config.repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(wrapper_git_sha) != 40:
        raise RuntimeError(f"Wrapper git SHA is unavailable: {wrapper_relative}")

    auxiliary = tuple(
        (package.module, package.distribution) for package in model.auxiliary_packages
    )
    runtime_library_path = model_runtime_library_path(model)
    runtime = _inspect_runtime(
        str(model.python),
        auxiliary,
        str(runtime_library_path),
        model.python_no_user_site,
        model.env_isolation,
        model.family,
    )
    signature = {
        "schema": "mprisk_cache_asset_signature_v3",
        "model_key": model.model_key,
        "family": model.family,
        "dtype": model.dtype,
        "python_no_user_site": model.python_no_user_site,
        "env_isolation": model.env_isolation,
        "frame_protocol": model.frame_protocol,
        "requested_frames": model.requested_frames,
        "video_sampling_method": model.video_sampling_method,
        "runtime_library_path": str(runtime_library_path),
        "sys_executable": runtime["sys_executable"],
        "transformers": runtime["transformers"],
        "transformers_classes": runtime["transformers_classes"],
        "auxiliary_packages": runtime["auxiliary_packages"],
        "model_path": str(model_path),
        "model_config_sha256": _sha256(config_path),
        "checkpoint_digest_schema": checkpoint["schema"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_digest_receipt": str(checkpoint_receipt_path),
        "model_asset_fingerprint": model_asset["sha256"],
        "extractor_semantic_schema": extractor["schema"],
        "extractor_semantic_sha256": extractor["sha256"],
        "extractor_semantic_files": {
            "repository": extractor["repository_files_sha256"],
            "trust_remote_code": extractor["trust_remote_code_files_sha256"],
        },
        "processor_contract_sha256": processor_contract_sha256,
        "processor_files": processor_files,
        "wrapper_path": wrapper_relative,
        "wrapper_git_sha": wrapper_git_sha,
        "wrapper_file_sha256": _sha256(wrapper_path),
    }
    if model.model_key == LLAVA_MODEL_KEY:
        signature["max_candidate_frames"] = model.max_candidate_frames
        signature["context_budget_mode"] = model.context_budget_mode
        selected_plan_paths = config.frame_plans if frame_plan_paths is None else frame_plan_paths
        if not selected_plan_paths:
            raise ValueError("LLaVA asset signature requires at least one frame plan")
        plans: dict[str, dict[str, Any]] = {}
        context_limits: set[int] = set()
        for domain_name, path in sorted(selected_plan_paths.items()):
            payload = load_frame_plan(path)
            if payload.get("model_key") != model.model_key:
                raise ValueError(f"Frame plan model mismatch: {path}")
            if Path(str(payload.get("model_path"))).resolve() != model_path:
                raise ValueError(f"Frame plan checkpoint mismatch: {path}")
            if payload.get("model_config_sha256") != _sha256(config_path):
                raise ValueError(f"Frame plan checkpoint config is stale: {path}")
            if payload.get("prompt_set_sha256") != _sha256(config.prompt_sets["vt"]):
                raise ValueError(f"Frame plan prompt set is stale: {path}")
            if payload.get("max_candidate_frames") != model.max_candidate_frames:
                raise ValueError(f"Frame plan candidate limit mismatch: {path}")
            context_limits.add(int(payload["max_position_embeddings"]))
            plans[domain_name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "schema": payload["schema"],
                "manifest_sha256": payload["manifest_sha256"],
                "entries": len(payload["entries"]),
            }
        if len(context_limits) != 1:
            raise ValueError("LLaVA frame plans disagree on checkpoint context limit")
        signature["context_budget_algorithm"] = {
            "schema": FRAME_PLAN_SCHEMA,
            "mode": CONTEXT_BUDGET_MODE,
            "max_position_embeddings": context_limits.pop(),
            "max_candidate_frames": model.max_candidate_frames,
            "selection_conditions": list(SELECTION_CONDITIONS),
            "selection_rule": "largest_f_with_all_p8_m1_m12_tokens_lte_context",
            "no_truncation": True,
        }
        signature["frame_plan_scope"] = (
            "full_matrix" if frame_plan_paths is None else "smoke_subset"
        )
        signature["frame_plan_manifests"] = plans
    return signature


@cache
def _inspect_runtime(
    python: str,
    auxiliary_packages: tuple[tuple[str, str], ...],
    runtime_library_path: str,
    python_no_user_site: bool,
    env_isolation: bool,
    family: str,
) -> dict[str, Any]:
    code = """
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import site
import sys

packages = json.loads(sys.argv[1])
family = sys.argv[2]
expected_no_user_site = sys.argv[3] == "1"
if expected_no_user_site and not sys.flags.no_user_site:
    raise RuntimeError("PYTHONNOUSERSITE did not disable the user site")
transformers = importlib.import_module("transformers")
auxiliary = {}
for item in packages:
    module = importlib.import_module(item["module"])
    module_path = Path(module.__file__).resolve()
    auxiliary[item["module"]] = {
        "distribution": item["distribution"],
        "path": str(module_path),
        "version": importlib.metadata.version(item["distribution"]),
    }
transformers_classes = {}
if family == "gemma4":
    for name in (
        "Gemma4UnifiedProcessor",
        "Gemma4UnifiedConfig",
        "Gemma4UnifiedForConditionalGeneration",
    ):
        value = getattr(transformers, name)
        source_path = Path(inspect.getfile(value)).resolve()
        transformers_classes[name] = {
            "module": value.__module__,
            "source_path": str(source_path),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
print(json.dumps({
    "sys_executable": str(Path(sys.executable).resolve()),
    "python_no_user_site": bool(sys.flags.no_user_site),
    "site_enable_user_site": bool(site.ENABLE_USER_SITE),
    "transformers": {
        "path": str(Path(transformers.__file__).resolve()),
        "version": str(transformers.__version__),
    },
    "transformers_classes": transformers_classes,
    "auxiliary_packages": auxiliary,
}, sort_keys=True))
"""
    package_payload = [
        {"module": module, "distribution": distribution}
        for module, distribution in auxiliary_packages
    ]
    env = dict(os.environ)
    inherited_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = runtime_library_path + (
        f":{inherited_library_path}" if inherited_library_path else ""
    )
    _apply_python_isolation(
        env,
        python_no_user_site=python_no_user_site,
        env_isolation=env_isolation,
    )
    completed = subprocess.run(
        [
            python,
            "-c",
            code,
            json.dumps(package_payload, sort_keys=True),
            family,
            "1" if python_no_user_site else "0",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Runtime signature inspection failed for {python}: {completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Runtime signature inspection returned invalid JSON for {python}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Runtime signature inspection returned non-object for {python}")
    if bool(value.get("python_no_user_site")) != python_no_user_site:
        raise RuntimeError(
            f"Runtime user-site flag mismatch for {python}: "
            f"expected={python_no_user_site}, actual={value.get('python_no_user_site')}"
        )
    if python_no_user_site and value.get("site_enable_user_site") is not False:
        raise RuntimeError(f"User site remains enabled for isolated runtime {python}")
    return value


def _job_command(config: MatrixConfig, job: CacheJob) -> list[str]:
    command = [
        str(job.model.python),
        str(config.job_runner),
        "--gpu-memory-fraction",
        str(config.max_gpu_memory_fraction),
        "--",
        "--manifest",
        str(job.domain.prepared_manifest),
        "--prompt-set",
        str(config.prompt_sets[job.model.protocol]),
        "--protocol",
        job.model.protocol,
        "--model-key",
        job.model.model_key,
        "--asset-config",
        str(config.asset_config),
        "--device",
        "cuda:0",
        "--dtype",
        job.model.dtype,
        "--prefill-strategy",
        "full_prefill",
        "--output-root",
        str(job.output_root),
        "--materialize-every",
        "100",
        "--video-num-segments",
        str(job.model.frame_count_argument),
    ]
    if job.frame_plan is not None:
        command.extend(["--frame-plan", str(job.frame_plan)])
    command.extend(job.model.extra_args)
    return command


def model_runtime_library_path(model: ModelSpec) -> Path:
    environment_lib = model.python.parent.parent / "lib"
    if not environment_lib.is_dir():
        raise FileNotFoundError(
            "Selected Python environment has no runtime library directory: "
            f"{environment_lib}"
        )
    return environment_lib.resolve()


def build_model_environment(
    config: MatrixConfig, model: ModelSpec, lane: int
) -> dict[str, str]:
    env = dict(os.environ)
    runtime_library_path = model_runtime_library_path(model)
    inherited_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(runtime_library_path) + (
        f":{inherited_library_path}" if inherited_library_path else ""
    )
    _apply_python_isolation(
        env,
        python_no_user_site=model.python_no_user_site,
        env_isolation=model.env_isolation,
    )
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(lane),
            "PYTHONPATH": str(config.repo_root / "src"),
            "OMP_NUM_THREADS": str(config.cpu_threads_per_job),
            "MKL_NUM_THREADS": str(config.cpu_threads_per_job),
            "OPENBLAS_NUM_THREADS": str(config.cpu_threads_per_job),
            "NUMEXPR_NUM_THREADS": str(config.cpu_threads_per_job),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _apply_python_isolation(
    env: dict[str, str], *, python_no_user_site: bool, env_isolation: bool
) -> None:
    if python_no_user_site != env_isolation:
        raise ValueError("python_no_user_site and env_isolation must be equal")
    if env_isolation:
        env.update(
            {
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    else:
        env.pop("PYTHONNOUSERSITE", None)


def build_job_environment(
    config: MatrixConfig, job: CacheJob, lane: int | None = None
) -> dict[str, str]:
    selected_lane = job.model.gpu_lane if lane is None else lane
    return build_model_environment(config, job.model, selected_lane)


def _require_gpu_capacity(lane: int, fraction: float) -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            str(lane),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used, total = (float(value.strip()) for value in completed.stdout.split(","))
    if used / total >= fraction:
        raise GPUCapacityBusy(
            f"GPU {lane} memory is already {used / total:.1%} utilized"
        )
    gpu_uuid = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
            "-i",
            str(lane),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    foreign_processes = []
    for line in compute.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) == 3 and parts[0] == gpu_uuid:
            foreign_processes.append(
                {"pid": int(parts[1]), "used_memory_mib": int(parts[2])}
            )
    if foreign_processes:
        raise GPUCapacityBusy(
            f"GPU {lane} has active compute processes: {foreign_processes}"
        )


def _wait_for_gpu_capacity(
    lane: int,
    fraction: float,
    *,
    poll_interval_seconds: float = 30.0,
) -> None:
    while True:
        try:
            _require_gpu_capacity(lane, fraction)
        except GPUCapacityBusy as exc:
            print(f"Waiting for GPU {lane} capacity: {exc}", flush=True)
            time.sleep(poll_interval_seconds)
            continue
        return


def _write_runtime(
    runtime_record: Path,
    stage_jobs: list[CacheJob],
    running: dict[int, tuple[CacheJob, subprocess.Popen[Any], Any]],
) -> None:
    payload = {
        "schema": "mprisk_cache_matrix_runtime_v1",
        "updated_at_unix": time.time(),
        "running": [
            {"gpu": lane, "job_id": value[0].job_id, "pid": value[1].pid}
            for lane, value in running.items()
        ],
        "jobs": [
            _ledger_status(job.output_root, job.domain.expected_tasks) | {"job_id": job.job_id}
            for job in stage_jobs
        ],
    }
    _atomic_text(runtime_record, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("all", "source", "target"), default="all")
    parser.add_argument("--lane", type=int, choices=(0, 1))
    parser.add_argument("--wait-for-gpu", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare-manifests", action="store_true")
    mode.add_argument("--prepare-frame-plans", action="store_true")
    mode.add_argument(
        "--prepare-frame-plan-domain", choices=("source", "target")
    )
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_matrix_config(args.config)
    if args.prepare_manifests:
        payload = prepare_manifests(config)
    elif args.prepare_frame_plans:
        payload = prepare_frame_plans(config)
    elif args.prepare_frame_plan_domain:
        payload = prepare_frame_plans(
            config, domains=(args.prepare_frame_plan_domain,)
        )
    elif args.dry_run:
        payload = audit_matrix(config)
    elif args.launch:
        launch_tmux(config)
        payload = {"status": "launched", "tmux_session": config.tmux_session}
    else:
        execute_matrix(
            config,
            stage=args.stage,
            lane=args.lane,
            wait_for_gpu=args.wait_for_gpu,
        )
        payload = {
            "status": "complete",
            "stage": args.stage,
            "lane": args.lane,
            "wait_for_gpu": args.wait_for_gpu,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _required_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    return _mapping(mapping.get(key), key)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required_list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return value


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_bool(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _shape(value: Any, label: str) -> tuple[int, int]:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
    ):
        raise ValueError(f"{label} must be [positive layers, positive hidden]")
    return int(value[0]), int(value[1])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(cli())
