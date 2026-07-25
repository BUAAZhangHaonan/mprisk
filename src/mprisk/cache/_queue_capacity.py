"""Disk/inode capacity gate evaluation."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mprisk.cache._queue_manifest import (
    CapacityFailure,
    CapacityGate,
    CapacityModel,
    CapacityStatus,
    QueueManifest,
)
from mprisk.viz.runtime_records import utc_now

_artifact_stats_cache: dict[tuple[str, float], tuple[int, int, int]] = {}


def evaluate_capacity(
    queue: QueueManifest,
    *,
    statvfs_fn: Callable[[Path], Any] = os.statvfs,
) -> CapacityStatus:
    gate = queue.capacity_gate
    filesystem = statvfs_fn(gate.filesystem_path)
    block_size = int(filesystem.f_frsize)
    total_bytes = int(filesystem.f_blocks) * block_size
    # f_bavail consistent across used/free
    used_bytes = (int(filesystem.f_blocks) - int(filesystem.f_bavail)) * block_size
    free_bytes = int(filesystem.f_bavail) * block_size
    total_inodes = int(filesystem.f_files)
    # f_favail consistent across used/free, matching bytes fix
    free_inodes = int(filesystem.f_favail)
    used_inodes = total_inodes - free_inodes
    projected_bytes = 0
    projected_inodes = 0
    model_records: list[dict[str, Any]] = []
    for model in gate.models:
        calibration_bytes, _, calibration_tasks = _artifact_stats(model.calibration_root)
        if calibration_tasks <= 0:
            raise CapacityFailure(f"{model.model_key} calibration has no cache artifacts")
        average_bytes = calibration_bytes / calibration_tasks
        expected_tasks = sum(output.expected_tasks for output in model.outputs)
        current_bytes = 0
        current_files = 0
        current_tasks = 0
        for output in model.outputs:
            artifact_bytes, artifact_files, artifact_tasks = _artifact_stats(output.output_root)
            current_bytes += artifact_bytes
            current_files += artifact_files
            current_tasks += artifact_tasks
        projected_final_bytes = math.ceil(average_bytes * expected_tasks)
        additional_bytes = max(projected_final_bytes - current_bytes, 0)
        additional_files = max(expected_tasks * 2 - current_files, 0)
        projected_bytes += additional_bytes
        projected_inodes += additional_files
        model_records.append(
            {
                "model_key": model.model_key,
                "calibration_root": str(model.calibration_root),
                "calibration_tasks": calibration_tasks,
                "average_artifact_bytes_per_task": average_bytes,
                "expected_tasks": expected_tasks,
                "current_tasks": current_tasks,
                "current_artifact_bytes": current_bytes,
                "projected_additional_bytes": additional_bytes,
                "projected_additional_inodes": additional_files,
            }
        )
    projected_used_bytes = used_bytes + projected_bytes
    df_capacity_bytes = used_bytes + free_bytes
    projected_utilization = (
        projected_used_bytes / df_capacity_bytes if df_capacity_bytes else 1.0
    )
    df_capacity_inodes = used_inodes + free_inodes
    projected_inode_utilization = (
        (used_inodes + projected_inodes) / df_capacity_inodes
        if df_capacity_inodes
        else 1.0
    )
    safe = (
        projected_utilization < gate.max_projected_utilization
        and projected_inode_utilization < gate.max_projected_utilization
    )
    return CapacityStatus(
        safe=safe,
        filesystem_path=gate.filesystem_path,
        total_bytes=total_bytes,
        used_bytes=used_bytes,
        free_bytes=free_bytes,
        projected_bytes=projected_bytes,
        projected_used_bytes=projected_used_bytes,
        projected_utilization=projected_utilization,
        total_inodes=total_inodes,
        free_inodes=free_inodes,
        projected_inodes=projected_inodes,
        projected_inode_utilization=projected_inode_utilization,
        max_projected_utilization=gate.max_projected_utilization,
        models=tuple(model_records),
    )


def _artifact_stats(root: Path) -> tuple[int, int, int]:
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        return 0, 0, 0
    cache_key = (str(manifest.resolve()), manifest.stat().st_mtime)
    cached = _artifact_stats_cache.get(cache_key)
    if cached is not None:
        return cached
    total_bytes = 0
    file_count = 0
    task_count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cache_root = Path(str(entry["cache_root"]))
        shard = cache_root / str(entry["shard_path"])
        sidecar = cache_root / str(entry["metadata"]["sidecar_path"])
        for artifact in (shard, sidecar):
            if not artifact.is_file():
                raise CapacityFailure(f"Capacity artifact is missing: {artifact}")
            total_bytes += artifact.stat().st_size
            file_count += 1
        task_count += 1
    result = (total_bytes, file_count, task_count)
    _artifact_stats_cache[cache_key] = result
    return result


def _capacity_payload(status: CapacityStatus) -> dict[str, Any]:
    return {
        "safe": status.safe,
        "filesystem_path": str(status.filesystem_path),
        "total_bytes": status.total_bytes,
        "used_bytes": status.used_bytes,
        "free_bytes": status.free_bytes,
        "projected_bytes": status.projected_bytes,
        "projected_used_bytes": status.projected_used_bytes,
        "projected_utilization": status.projected_utilization,
        "total_inodes": status.total_inodes,
        "free_inodes": status.free_inodes,
        "projected_inodes": status.projected_inodes,
        "projected_inode_utilization": status.projected_inode_utilization,
        "max_projected_utilization": status.max_projected_utilization,
        "models": list(status.models),
        "recorded_at": utc_now(),
    }
