"""Validation of completed prefill caches (ledger + manifest + artifact checksums)."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from mprisk.data.manifests import read_jsonl
from mprisk.experiments._io_utils import _load_yaml, _sha256
from mprisk.experiments.jobs import CONDITIONS, CacheJob, CacheNotReady


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
