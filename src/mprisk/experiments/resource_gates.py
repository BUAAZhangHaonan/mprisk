"""Resource gates: GPU setup, aggregation, completion check, runtime status."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch

from mprisk.evaluation.downstream_metrics import aggregate_three_seeds
from mprisk.experiments.jobs import (
    REPRESENTATIONS,
    CacheJob,
    DownstreamPlan,
)
from mprisk.representation.relation_models import TME_PROXY_ANCHOR_V1
from mprisk.utils.io import write_json


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
            "waiting_reason": (
                None if completed == 27 else "cache_or_registered_gpu_resource_gate"
            ),
            "updated_unix": time.time(),
        },
    )
