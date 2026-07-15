"""Atomic machine-readable runtime records for RUN_STATUS aggregation."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_RECORDS_SCHEMA = "mprisk_run_records_v1"


def append_command_record(
    path: str | Path,
    *,
    command_id: str,
    argv: list[str],
    status: str,
    pid: int,
    started_at: str,
    reason: str = "",
    gpu: dict[str, Any] | None = None,
) -> None:
    payload = load_run_records(path)
    payload["commands"].append(
        {
            "command_id": command_id,
            "argv": list(argv),
            "status": status,
            "pid": pid,
            "started_at": started_at,
            "ended_at": utc_now(),
            "reason": reason,
            "gpu": gpu,
        }
    )
    write_run_records(path, payload)


def snapshot_gpu_records(path: str | Path) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, name, total, used, utilization = [part.strip() for part in line.split(",", 4)]
        rows.append(
            {
                "physical_index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": int(utilization),
                "recorded_at": utc_now(),
            }
        )
    payload = load_run_records(path)
    payload["gpus"] = rows
    write_run_records(path, payload)
    return rows


def snapshot_cache_manifest(
    path: str | Path,
    *,
    cache_key: str,
    manifest_path: str | Path,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    if not manifest.is_file():
        row = {
            "cache_key": cache_key,
            "status": "missing",
            "complete": 0,
            "failed": None,
            "missing": None,
            "source": str(manifest),
        }
    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ValueError("cache manifest entries must be a list")
        row = {
            "cache_key": cache_key,
            "status": "complete" if entries else "missing",
            "complete": len(entries),
            "failed": None,
            "missing": None,
            "source": str(manifest),
        }
    _replace_keyed_record(path, "caches", "cache_key", row)
    return row


def snapshot_cache_summary(
    path: str | Path,
    *,
    cache_key: str,
    summary_path: str | Path,
) -> dict[str, Any]:
    source = Path(summary_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    complete = int(payload["completed"])
    failed = int(payload["failed"])
    pending = int(payload.get("pending", 0)) + int(payload.get("running", 0))
    total = int(payload["total"])
    status = (
        "failure"
        if failed
        else "complete"
        if complete == total and not pending
        else "incomplete"
    )
    row = {
        "cache_key": cache_key,
        "status": status,
        "complete": complete,
        "failed": failed,
        "missing": pending,
        "source": str(source),
    }
    _replace_keyed_record(path, "caches", "cache_key", row)
    return row


def load_run_records(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {
            "schema": RUN_RECORDS_SCHEMA,
            "commands": [],
            "gpus": [],
            "caches": [],
            "experiments": [],
        }
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != RUN_RECORDS_SCHEMA:
        raise ValueError(f"run records schema must be {RUN_RECORDS_SCHEMA}")
    for key in ("commands", "gpus", "caches", "experiments"):
        payload.setdefault(key, [])
        if not isinstance(payload[key], list):
            raise ValueError(f"run records {key} must be a list")
    return payload


def write_run_records(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _replace_keyed_record(
    path: str | Path,
    collection: str,
    key: str,
    row: dict[str, Any],
) -> None:
    payload = load_run_records(path)
    payload[collection] = [item for item in payload[collection] if item.get(key) != row[key]]
    payload[collection].append(row)
    write_run_records(path, payload)
