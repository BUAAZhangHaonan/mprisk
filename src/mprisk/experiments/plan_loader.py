"""Loader for the YAML plan describing the downstream queue."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mprisk.experiments._io_utils import _load_yaml, _resolve
from mprisk.experiments.jobs import (
    AllowedExternalGpuContext,
    CacheJob,
    DownstreamPlan,
    PLAN_SCHEMA,
)


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
    resource: dict[str, Any] = payload.get("resource_gate") or {}
    fraction = float(resource.get("max_gpu_memory_fraction", 0.9))
    if not 0 < fraction < 0.9:
        raise ValueError("max_gpu_memory_fraction must be strictly below 0.90")
    external_context_rows = resource.get("allowed_external_gpu_contexts")
    if not isinstance(external_context_rows, list):
        raise ValueError("allowed_external_gpu_contexts must be an explicit list")
    external_contexts = tuple(
        AllowedExternalGpuContext(
            process_name=str(row["process_name"]),
            command_substring=str(row["command_substring"]),
            max_process_count=int(row["max_process_count"]),
            max_gpu_memory_mib_per_process=float(row["max_gpu_memory_mib_per_process"]),
        )
        for row in external_context_rows
    )
    if any(not rule.process_name or not rule.command_substring for rule in external_contexts):
        raise ValueError(
            "allowed external GPU context process_name and command_substring cannot be empty"
        )
    context_identities = {
        (rule.process_name, rule.command_substring) for rule in external_contexts
    }
    if len(context_identities) != len(external_contexts):
        raise ValueError("allowed external GPU context identities must be unique")
    if any(rule.max_process_count <= 0 for rule in external_contexts):
        raise ValueError("allowed external GPU context max_process_count must be positive")
    if any(rule.max_gpu_memory_mib_per_process <= 0 for rule in external_contexts):
        raise ValueError(
            "allowed external GPU context max_gpu_memory_mib_per_process must be positive"
        )
    max_external_memory = float(resource["max_external_gpu_context_memory_mib"])
    if max_external_memory <= 0:
        raise ValueError("max_external_gpu_context_memory_mib must be positive")
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
        producer_tmux_sessions=tuple(
            str(value) for value in resource.get("producer_tmux_sessions", [])
        ),
        producer_command_substrings=tuple(
            str(value) for value in resource.get("producer_command_substrings", [])
        ),
        allowed_external_gpu_contexts=external_contexts,
        max_external_gpu_context_memory_mib=max_external_memory,
    )
