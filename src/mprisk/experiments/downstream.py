"""Resumable, identity-locked downstream experiments for completed P=8 caches.

This module is now a thin orchestrator. The bulk of the logic lives in:

- :mod:`mprisk.experiments.jobs`             - dataclasses + plan constants
- :mod:`mprisk.experiments.plan_loader`      - YAML plan parsing
- :mod:`mprisk.experiments.cache_validation` - completed-cache gate
- :mod:`mprisk.experiments.relation_dataset` - relation dataset builder + official rows
- :mod:`mprisk.experiments.training_tasks`   - per-run training tasks
- :mod:`mprisk.experiments.resource_gates`   - GPU resource gates + queue status
- :mod:`mprisk.experiments._io_utils`        - hashing/yaml/csv helpers

``run_queue``, ``_gpu_available`` and ``_cache_producer_can_launch_gpu_work`` stay here so
that the contract test (``tests/test_downstream/test_downstream_contract.py``) can keep
monkey-patching ``downstream.subprocess`` and ``downstream._cache_producer_can_launch_gpu_work``
as a single, in-module unit.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from mprisk.evaluation.misread_probe import write_pending_conflict_misread_probe
from mprisk.experiments._io_utils import (
    _json_sha256,
    _load_yaml,
    _one,
    _resolve,
    _sha256,
    _training_config_path,
    _write_csv,
)
from mprisk.experiments.cache_validation import (
    _verify_cache_artifact,
    validate_completed_cache,
)
from mprisk.experiments.jobs import (
    AllowedExternalGpuContext,
    CacheJob,
    CacheNotReady,
    CONDITIONS,
    REPRESENTATIONS,
    DownstreamPlan,
    OFFICIAL_TEST,
    PLAN_SCHEMA,
    TRAINING_SPLITS,
    CALIBRATION,
)
from mprisk.experiments.plan_loader import load_plan
from mprisk.experiments.relation_dataset import (
    build_relation_dataset_from_cache,
    official_test_rows,
)
from mprisk.experiments.resource_gates import (
    _aggregate_ready_models,
    _all_runs_complete,
    _configure_resources,
    _write_runtime_status,
)
from mprisk.experiments.training_tasks import (
    _export_tme_state_outputs,
    _retained_conflict_rows,
    _run_model_seed,
    _run_retention_sensitivity,
    _train_until_converged,
)
from mprisk.representation.relation_models import TME_PROXY_ANCHOR_V1
from mprisk.utils.io import write_json

__all__ = [
    # Public orchestrator
    "run_queue",
    "load_plan",
    # Jobs + constants
    "CacheJob",
    "CacheNotReady",
    "DownstreamPlan",
    "AllowedExternalGpuContext",
    "PLAN_SCHEMA",
    "CONDITIONS",
    "OFFICIAL_TEST",
    "CALIBRATION",
    "TRAINING_SPLITS",
    "REPRESENTATIONS",
    # Cache + relation dataset
    "validate_completed_cache",
    "build_relation_dataset_from_cache",
    "official_test_rows",
    # Training + resource gates (kept exported because run_queue uses them, and tests poke
    # the train pipeline via these symbols on the downstream module)
    "_run_model_seed",
    "_run_retention_sensitivity",
    "_retained_conflict_rows",
    "_train_until_converged",
    "_export_tme_state_outputs",
    "_verify_cache_artifact",
    "_configure_resources",
    "_aggregate_ready_models",
    "_all_runs_complete",
    "_write_runtime_status",
    "_gpu_available",
    "_cache_producer_can_launch_gpu_work",
    # IO helpers (kept re-exported so existing callers continue to work)
    "_training_config_path",
    "_load_yaml",
    "_resolve",
    "_sha256",
    "_json_sha256",
    "_write_csv",
    "_one",
]


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


def _gpu_available(plan: DownstreamPlan) -> bool:
    if _cache_producer_can_launch_gpu_work(plan):
        return False
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
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    external_processes: list[tuple[int, str, float]] = []
    for line in process_output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            raise RuntimeError(f"invalid nvidia-smi compute-app row: {line!r}")
        try:
            process_memory_mib = float(fields[2])
        except ValueError as exc:
            raise RuntimeError(f"invalid nvidia-smi process memory: {line!r}") from exc
        pid = int(fields[0])
        if pid != os.getpid():
            external_processes.append((pid, fields[1], process_memory_mib))
    if not external_processes:
        return True

    command_output = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    command_by_pid: dict[int, str] = {}
    for row in command_output:
        fields = row.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].isdigit():
            command_by_pid[int(fields[0])] = fields[1]

    process_counts: Counter[AllowedExternalGpuContext] = Counter()
    external_memory_mib = 0.0
    for pid, process_name, process_memory_mib in external_processes:
        command = command_by_pid.get(pid)
        if command is None:
            return False
        matching_rules = [
            rule
            for rule in plan.allowed_external_gpu_contexts
            if rule.process_name == process_name and rule.command_substring in command
        ]
        if len(matching_rules) != 1:
            return False
        rule = matching_rules[0]
        if process_memory_mib > rule.max_gpu_memory_mib_per_process:
            return False
        process_counts[rule] += 1
        if process_counts[rule] > rule.max_process_count:
            return False
        external_memory_mib += process_memory_mib
    return external_memory_mib <= plan.max_external_gpu_context_memory_mib


def _cache_producer_can_launch_gpu_work(plan: DownstreamPlan) -> bool:
    for session in plan.producer_tmux_sessions:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
    process_rows = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for row in process_rows:
        fields = row.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) == os.getpid():
            continue
        if any(token in fields[1] for token in plan.producer_command_substrings):
            return True
    return False
