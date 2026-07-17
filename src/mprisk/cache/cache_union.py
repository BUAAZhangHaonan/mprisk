"""Immutable, fail-closed views over disjoint prefill-cache roots."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from safetensors import safe_open

from mprisk.cache.prefill_batch import BatchPlan, _request_for_task
from mprisk.models.base_wrapper import PrefillRequest

EVIDENCE_SCHEMA = "mprisk_prefill_extractor_evidence_v1"
SOURCE_SCHEMA = "mprisk_prefill_cache_source_v1"
UNION_SCHEMA = "mprisk_prefill_cache_union_v2"
UNION_VERSION = "v2"
SIDECAR_SCHEMA = "mprisk_prefill_cache_sidecar_v1"
FULL_PREFILL_STRATEGY = "full_prefill"
FULL_PREFILL_STRATEGY_VERSION = "v1"

SEMANTIC_CODE_PATHS = (
    "src/mprisk/assets/registry.py",
    "src/mprisk/cache/prefill_batch.py",
    "src/mprisk/cache/prefill_writer.py",
    "src/mprisk/models/base_wrapper.py",
    "src/mprisk/models/qwen_omni.py",
    "src/mprisk/models/wrapper_registry.py",
    "src/mprisk/prompts/compiler.py",
    "src/mprisk/prompts/template_bank.py",
)
WRAPPER_PATHS = {
    "qwen_vl": "src/mprisk/models/qwen_vl.py",
    "qwen_omni": "src/mprisk/models/qwen_omni.py",
    "internvl": "src/mprisk/models/internvl.py",
}

SIGNATURE_IGNORED_FIELDS = frozenset(
    {
        "schema",
        "manifest_sha256",
        "prefill_strategy",
        "prefill_strategy_version",
    }
)
RUNTIME_PROVENANCE_FIELDS = (
    "schema",
    "model_path",
    "model_class",
    "processor_class",
    "talker_loaded",
    "transformers_version",
    "qwen_omni_utils_version",
    "torch_version",
    "source_dtype",
    "stored_dtype",
    "attn_implementation",
    "num_hidden_layers",
    "hidden_size",
    "hidden_state_index_offset",
    "model_config_sha256",
    "weight_index_sha256",
    "video_fps",
    "video_num_segments",
    "internvl_max_num",
)
MODEL_RUNTIME_ASSET_SUFFIXES = frozenset({".json", ".model", ".py", ".tiktoken", ".txt"})
WEIGHT_INDEX_FILENAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


class CacheUnionError(ValueError):
    """Raised when a cache union cannot be proven complete and exact."""


@dataclass(frozen=True)
class ExpectedCacheTask:
    task_id: str
    request: PrefillRequest
    sample_type: str
    split: str
    source_dataset: str


@dataclass(frozen=True)
class BlockedCacheTask:
    task_id: str
    sample_id: str
    prompt_id: str
    condition: str
    reason: str


@dataclass(frozen=True)
class CacheSource:
    source_id: str
    cache_root: Path
    ledger_path: Path
    evidence_path: Path


@dataclass(frozen=True)
class SourceTask:
    source: CacheSource
    status: str
    task_id: str
    entry: dict[str, Any] | None
    model_asset_fingerprint: str
    model_asset_inventory: dict[str, Any]


@dataclass(frozen=True)
class CacheUnionResult:
    output_path: Path
    resolved_tasks: int
    blocked_tasks: int
    source_counts: dict[str, int]
    extractor_semantic_fingerprint: str


def expected_tasks_from_plan(args: Any, plan: BatchPlan) -> list[ExpectedCacheTask]:
    """Compile the exact request identity for every valid delivery task."""
    expected: list[ExpectedCacheTask] = []
    for task in plan.tasks:
        expected.append(
            ExpectedCacheTask(
                task_id=task.task_id,
                request=_request_for_task(args, task),
                sample_type=str(task.row["sample_type"]),
                split=str(task.row["split"]),
                source_dataset=str(task.row["source_dataset"]),
            )
        )
    _require_unique_task_ids(expected)
    return expected


def blocked_tasks_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    model_key: str,
    protocol: str,
    prompt_ids: Sequence[str],
    conditions: Sequence[str] = ("M1", "M2", "M12"),
) -> list[BlockedCacheTask]:
    """Account for invalid input rows without exposing them as cache entries."""
    blocked: list[BlockedCacheTask] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        reason = str(row.get("cache_block_reason") or row.get("reason") or "").strip()
        if not reason:
            raise CacheUnionError(f"Blocked sample has no reason: {sample_id}")
        for prompt_id in prompt_ids:
            for condition in conditions:
                identity = {
                    "sample_id": sample_id,
                    "prompt_id": str(prompt_id),
                    "condition": str(condition).upper(),
                    "protocol": str(protocol).lower(),
                    "model_key": str(model_key),
                }
                blocked.append(
                    BlockedCacheTask(
                        task_id=_sha256_bytes(_canonical_json(identity).encode()),
                        sample_id=sample_id,
                        prompt_id=str(prompt_id),
                        condition=str(condition).upper(),
                        reason=reason,
                    )
                )
    task_ids = [task.task_id for task in blocked]
    if len(task_ids) != len(set(task_ids)):
        raise CacheUnionError("Blocked cache task identities are not unique")
    return blocked


def load_cache_source(path: str | Path) -> CacheSource:
    config_path = Path(path).expanduser().resolve()
    payload = _read_json(config_path)
    if payload.get("schema") != SOURCE_SCHEMA:
        raise CacheUnionError(f"Unsupported cache source schema: {config_path}")
    required = ("source_id", "cache_root", "ledger_path", "evidence_path")
    missing = [field for field in required if not payload.get(field)]
    if missing:
        raise CacheUnionError(f"Cache source is missing fields {missing}: {config_path}")
    return CacheSource(
        source_id=str(payload["source_id"]),
        cache_root=_resolve_from(config_path, payload["cache_root"]),
        ledger_path=_resolve_from(config_path, payload["ledger_path"]),
        evidence_path=_resolve_from(config_path, payload["evidence_path"]),
    )


def write_extractor_evidence(
    *,
    source_id: str,
    ledger_path: str | Path,
    cache_root: str | Path,
    code_root: str | Path,
    output_path: str | Path,
) -> Path:
    """Record exact full-prefill code/config evidence after a ledger completes."""
    ledger = Path(ledger_path).expanduser().resolve()
    root = Path(cache_root).expanduser().resolve()
    repository = Path(code_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    signature = _read_ledger_signature(ledger)
    _require_completed_source_ledger(ledger)
    family = str(signature.get("family"))
    try:
        wrapper_path = WRAPPER_PATHS[family]
    except KeyError as exc:
        raise CacheUnionError(f"Unsupported extractor family in ledger: {family!r}") from exc
    code_paths = tuple(dict.fromkeys((*SEMANTIC_CODE_PATHS, wrapper_path)))
    code_hashes = _semantic_code_hashes(repository, code_paths)
    model_asset_inventory = _model_asset_inventory(str(signature["model_path"]))
    model_asset_fingerprint = _fingerprint(model_asset_inventory)
    git_commit = _git_output(repository, "rev-parse", "HEAD")
    dirty = _git_output(repository, "status", "--porcelain", "--", *code_paths)
    if dirty:
        raise CacheUnionError("Extractor semantic files are dirty; evidence would be ambiguous")
    normalized_signature = _normalized_signature(signature)
    fingerprint_input = {
        "strategy": FULL_PREFILL_STRATEGY,
        "strategy_version": FULL_PREFILL_STRATEGY_VERSION,
        "code_files_sha256": code_hashes,
        "extraction_signature": normalized_signature,
        "model_asset_fingerprint": model_asset_fingerprint,
    }
    payload = {
        "schema": EVIDENCE_SCHEMA,
        "source_id": source_id,
        "ledger_path": str(ledger),
        "cache_root": str(root),
        "ledger_content_sha256": _ledger_content_fingerprint(ledger),
        "ledger_file_sha256_at_recording": _sha256_file(ledger),
        "ledger_signature_sha256": _fingerprint(signature),
        "extractor_semantic_fingerprint": _fingerprint(fingerprint_input),
        "prefill_strategy": FULL_PREFILL_STRATEGY,
        "prefill_strategy_version": FULL_PREFILL_STRATEGY_VERSION,
        "code_root": str(repository),
        "git_commit": git_commit,
        "semantic_files_clean": True,
        "code_files_sha256": code_hashes,
        "extraction_signature": normalized_signature,
        "model_asset_fingerprint": model_asset_fingerprint,
        "model_asset_inventory": model_asset_inventory,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output, payload)
    return output


def write_cache_source(
    *,
    source_id: str,
    cache_root: str | Path,
    ledger_path: str | Path,
    evidence_path: str | Path,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    payload = {
        "schema": SOURCE_SCHEMA,
        "source_id": source_id,
        "cache_root": str(Path(cache_root).expanduser().resolve()),
        "ledger_path": str(Path(ledger_path).expanduser().resolve()),
        "evidence_path": str(Path(evidence_path).expanduser().resolve()),
    }
    _atomic_json(output, payload)
    return output


def build_cache_union(
    *,
    expected_tasks: Sequence[ExpectedCacheTask],
    expected_signature: dict[str, Any],
    sources: Sequence[CacheSource],
    output_path: str | Path,
    blocked_tasks: Sequence[BlockedCacheTask] = (),
    expected_resolved_tasks: int,
    expected_blocked_tasks: int = 0,
    expected_raw_tasks: int | None = None,
    checksum_workers: int = 8,
) -> CacheUnionResult:
    """Validate and atomically write an immutable multi-root cache view."""
    if checksum_workers <= 0:
        raise CacheUnionError("checksum_workers must be positive")
    _require_unique_task_ids(expected_tasks)
    if len(expected_tasks) != expected_resolved_tasks:
        raise CacheUnionError(
            f"Expected {expected_resolved_tasks} resolved tasks, got {len(expected_tasks)}"
        )
    if len(blocked_tasks) != expected_blocked_tasks:
        raise CacheUnionError(
            f"Expected {expected_blocked_tasks} blocked tasks, got {len(blocked_tasks)}"
        )
    raw_tasks = len(expected_tasks) + len(blocked_tasks)
    if expected_raw_tasks is not None and raw_tasks != expected_raw_tasks:
        raise CacheUnionError(f"Expected {expected_raw_tasks} raw tasks, got {raw_tasks}")
    overlap = {task.task_id for task in expected_tasks} & {
        task.task_id for task in blocked_tasks
    }
    if overlap:
        raise CacheUnionError("Usable and blocked task identities overlap")
    if not sources:
        raise CacheUnionError("At least one cache source is required")
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise CacheUnionError("Cache source IDs must be unique")

    source_rows: defaultdict[str, list[SourceTask]] = defaultdict(list)
    evidence_records = []
    for source in sources:
        evidence = _validate_source_evidence(source)
        _require_signature_compatible(expected_signature, evidence["extraction_signature"])
        evidence_records.append(evidence)
        for row in _read_source_tasks(source, evidence):
            source_rows[row.task_id].append(row)
    strategy_identities = {
        _source_prefill_identity(record, context="cache source evidence")
        for record in evidence_records
    }
    if len(strategy_identities) != 1:
        raise CacheUnionError("Cache sources have different prefill strategy identities")
    prefill_strategy, prefill_strategy_version = next(iter(strategy_identities))
    fingerprints = {
        str(record["extractor_semantic_fingerprint"]) for record in evidence_records
    }
    if len(fingerprints) != 1:
        raise CacheUnionError(
            "Cache sources have different extractor semantic fingerprints; exact reuse denied"
        )
    fingerprint = next(iter(fingerprints))
    model_asset_fingerprints = {
        str(record["model_asset_fingerprint"]) for record in evidence_records
    }
    if len(model_asset_fingerprints) != 1:
        raise CacheUnionError("Cache sources use different model assets")
    model_asset_fingerprint = next(iter(model_asset_fingerprints))

    blocked_payload = _blocked_payload(blocked_tasks, source_rows)
    jobs = []
    for expected in expected_tasks:
        matches = source_rows.get(expected.task_id, [])
        if len(matches) != 1:
            state = "missing" if not matches else "duplicated"
            raise CacheUnionError(f"Expected task is {state}: {expected.task_id}")
        source_task = matches[0]
        if source_task.status != "completed":
            raise CacheUnionError(
                f"Expected task is not completed ({source_task.status}): {expected.task_id}"
            )
        if source_task.entry is None:
            raise CacheUnionError(f"Completed task has no entry_json: {expected.task_id}")
        jobs.append((expected, source_task))

    def validate(item: tuple[ExpectedCacheTask, SourceTask]) -> dict[str, Any]:
        return _validate_and_materialize_entry(
            *item,
            expected_signature=expected_signature,
        )

    if checksum_workers == 1:
        entries = [validate(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=checksum_workers) as executor:
            entries = list(executor.map(validate, jobs))
    entries.sort(key=_union_entry_sort_key)
    source_counts = dict(Counter(entry["source_provenance"]["source_id"] for entry in entries))
    runtime_fingerprints = {
        str(entry["source_provenance"]["runtime_provenance_fingerprint"])
        for entry in entries
    }
    if len(runtime_fingerprints) != 1:
        raise CacheUnionError("Cache entries have incompatible stable runtime provenance")
    payload = {
        "schema": UNION_SCHEMA,
        "version": UNION_VERSION,
        "prefill_strategy": prefill_strategy,
        "prefill_strategy_version": prefill_strategy_version,
        "entries": entries,
        "blocked_tasks": blocked_payload,
        "provenance": {
            "created_at": datetime.now(UTC).isoformat(),
            "prefill_strategy": prefill_strategy,
            "prefill_strategy_version": prefill_strategy_version,
            "selection": "exact delivery task identities; no shard copying",
            "source_roots_immutable": True,
            "new_split_attached_only_in_union": True,
            "expected_signature": expected_signature,
            "expected_signature_sha256": _fingerprint(expected_signature),
            "extractor_semantic_fingerprint": fingerprint,
            "model_asset_fingerprint": model_asset_fingerprint,
            "runtime_provenance_fingerprint": next(iter(runtime_fingerprints)),
            "sources": [
                {
                    "source_id": source.source_id,
                    "cache_root": str(source.cache_root),
                    "ledger_path": str(source.ledger_path),
                    "evidence_path": str(source.evidence_path),
                    "evidence_sha256": _sha256_file(source.evidence_path),
                    "resolved_tasks": source_counts.get(source.source_id, 0),
                }
                for source in sources
            ],
            "counts": {
                "resolved_tasks": len(entries),
                "blocked_tasks": len(blocked_payload),
                "raw_tasks": len(entries) + len(blocked_payload),
            },
        },
    }
    output = Path(output_path).expanduser().resolve()
    _atomic_json(output, payload)
    return CacheUnionResult(
        output_path=output,
        resolved_tasks=len(entries),
        blocked_tasks=len(blocked_payload),
        source_counts=source_counts,
        extractor_semantic_fingerprint=fingerprint,
    )


def _validate_source_evidence(source: CacheSource) -> dict[str, Any]:
    if not source.cache_root.is_dir():
        raise CacheUnionError(f"Cache source root does not exist: {source.cache_root}")
    if not source.ledger_path.is_file():
        raise CacheUnionError(f"Cache source ledger does not exist: {source.ledger_path}")
    evidence = _read_json(source.evidence_path)
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise CacheUnionError(f"Unsupported extractor evidence: {source.evidence_path}")
    _source_prefill_identity(evidence, context=str(source.evidence_path))
    for field, expected in (
        ("source_id", source.source_id),
        ("cache_root", str(source.cache_root)),
        ("ledger_path", str(source.ledger_path)),
        ("ledger_content_sha256", _ledger_content_fingerprint(source.ledger_path)),
    ):
        if evidence.get(field) != expected:
            raise CacheUnionError(f"Extractor evidence {field} mismatch: {source.evidence_path}")
    signature = _read_ledger_signature(source.ledger_path)
    if evidence.get("ledger_signature_sha256") != _fingerprint(signature):
        raise CacheUnionError(f"Extractor evidence signature mismatch: {source.evidence_path}")
    fingerprint_input = {
        "strategy": evidence.get("prefill_strategy"),
        "strategy_version": evidence.get("prefill_strategy_version"),
        "code_files_sha256": evidence.get("code_files_sha256"),
        "extraction_signature": evidence.get("extraction_signature"),
        "model_asset_fingerprint": evidence.get("model_asset_fingerprint"),
    }
    if evidence.get("extractor_semantic_fingerprint") != _fingerprint(fingerprint_input):
        raise CacheUnionError(f"Extractor evidence fingerprint mismatch: {source.evidence_path}")
    if evidence.get("extraction_signature") != _normalized_signature(signature):
        raise CacheUnionError(f"Extractor evidence config mismatch: {source.evidence_path}")
    model_asset_inventory = evidence.get("model_asset_inventory")
    if not isinstance(model_asset_inventory, dict):
        raise CacheUnionError(f"Model asset inventory is missing: {source.evidence_path}")
    if evidence.get("model_asset_fingerprint") != _fingerprint(model_asset_inventory):
        raise CacheUnionError(f"Model asset fingerprint mismatch: {source.evidence_path}")
    current_inventory = _validated_model_asset_inventory(
        str(signature["model_path"]),
        str(evidence["model_asset_fingerprint"]),
    )
    if current_inventory != model_asset_inventory:
        raise CacheUnionError(f"Model assets changed after evidence: {source.evidence_path}")
    _validate_evidence_code(evidence, source.evidence_path)
    return evidence


def _source_prefill_identity(
    evidence: dict[str, Any],
    *,
    context: str,
) -> tuple[str, str]:
    strategy = evidence.get("prefill_strategy")
    version = evidence.get("prefill_strategy_version")
    if not isinstance(strategy, str) or not strategy.strip():
        raise CacheUnionError(f"Prefill strategy is missing from {context}")
    if not isinstance(version, str) or not version.strip():
        raise CacheUnionError(f"Prefill strategy version is missing from {context}")
    return strategy, version


def _read_source_tasks(
    source: CacheSource,
    evidence: dict[str, Any],
) -> list[SourceTask]:
    connection = sqlite3.connect(f"file:{source.ledger_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT task_id,status,entry_json FROM tasks ORDER BY rowid"
        ).fetchall()
    finally:
        connection.close()
    return [
        SourceTask(
            source=source,
            status=str(row["status"]),
            task_id=str(row["task_id"]),
            entry=json.loads(row["entry_json"]) if row["entry_json"] else None,
            model_asset_fingerprint=str(evidence["model_asset_fingerprint"]),
            model_asset_inventory=dict(evidence["model_asset_inventory"]),
        )
        for row in rows
    ]


def _validate_and_materialize_entry(
    expected: ExpectedCacheTask,
    source_task: SourceTask,
    *,
    expected_signature: dict[str, Any],
) -> dict[str, Any]:
    source_entry = source_task.entry
    if source_entry is None:
        raise CacheUnionError(f"Completed task has no entry: {expected.task_id}")
    cache_root = Path(str(source_entry.get("cache_root", ""))).expanduser().resolve()
    if not cache_root.is_relative_to(source_task.source.cache_root):
        raise CacheUnionError(f"Entry escapes declared cache root: {expected.task_id}")
    shard = _resolve_artifact(cache_root, source_entry.get("shard_path"))
    metadata = source_entry.get("metadata")
    if not isinstance(metadata, dict):
        raise CacheUnionError(f"Cache entry metadata is invalid: {expected.task_id}")
    sidecar = _resolve_artifact(cache_root, metadata.get("sidecar_path"))
    if not shard.is_file() or not sidecar.is_file():
        raise CacheUnionError(f"Cache artifact pair is incomplete: {expected.task_id}")
    sidecar_payload = _read_json(sidecar)
    if sidecar_payload.get("schema") != SIDECAR_SCHEMA:
        raise CacheUnionError(f"Unsupported cache sidecar schema: {sidecar}")
    if sidecar_payload.get("entry") != source_entry:
        raise CacheUnionError(f"Ledger/sidecar entry mismatch: {expected.task_id}")
    source_request = sidecar_payload.get("request")
    if not isinstance(source_request, dict):
        raise CacheUnionError(f"Cache sidecar request is invalid: {expected.task_id}")
    expected_request = _request_payload(expected.request)
    if _semantic_request(source_request) != _semantic_request(expected_request):
        raise CacheUnionError(f"Semantic request fingerprint mismatch: {expected.task_id}")
    for field in (
        "sample_id",
        "model_key",
        "protocol",
        "condition",
        "prompt_set_key",
        "prompt_id",
    ):
        if source_entry.get(field) != source_request.get(field):
            raise CacheUnionError(f"Entry/request {field} mismatch: {expected.task_id}")
    semantic_request_fingerprint = _fingerprint(_semantic_request(expected_request))
    checksum = _sha256_file(shard)
    if checksum != source_entry.get("checksum"):
        raise CacheUnionError(f"Cache shard checksum mismatch: {expected.task_id}")
    tensor_key = metadata.get("tensor_key")
    if tensor_key != "hidden_states":
        raise CacheUnionError(f"Unexpected cache tensor key: {expected.task_id}")
    with safe_open(str(shard), framework="np") as tensors:
        if list(tensors.keys()) != [tensor_key]:
            raise CacheUnionError(f"Unexpected cache shard tensors: {expected.task_id}")
        shape = tuple(int(value) for value in tensors.get_slice(tensor_key).get_shape())
    expected_shape = (
        int(source_entry.get("layer_count", -1)),
        int(source_entry.get("hidden_dim", -1)),
    )
    if shape != expected_shape or any(value <= 0 for value in shape):
        raise CacheUnionError(f"Cache tensor shape mismatch: {expected.task_id}")
    token_count = int(source_entry.get("token_count", 0))
    t0_token_index = int(source_entry.get("t0_token_index", -1))
    if token_count <= 0 or t0_token_index != token_count - 1:
        raise CacheUnionError(f"Invalid t0/token count contract: {expected.task_id}")
    provenance = sidecar_payload.get("provenance")
    if not isinstance(provenance, dict):
        raise CacheUnionError(f"Cache sidecar provenance is invalid: {expected.task_id}")
    if provenance.get("model_path") is None:
        raise CacheUnionError(f"Cache provenance has no model path: {expected.task_id}")
    for provenance_field, signature_field in (
        ("model_path", "model_path"),
        ("attn_implementation", "attn_implementation"),
        ("source_dtype", "dtype"),
    ):
        if provenance.get(provenance_field) != expected_signature.get(signature_field):
            raise CacheUnionError(
                f"Provenance {provenance_field} differs from plan: {expected.task_id}"
            )
    if int(provenance.get("num_hidden_layers", -1)) != shape[0]:
        raise CacheUnionError(f"Provenance layer count mismatch: {expected.task_id}")
    if int(provenance.get("hidden_size", -1)) != shape[1]:
        raise CacheUnionError(f"Provenance hidden size mismatch: {expected.task_id}")
    if provenance.get("stored_dtype") != "float32":
        raise CacheUnionError(f"Unexpected stored dtype: {expected.task_id}")
    model_config_sha256 = provenance.get("model_config_sha256")
    weight_index_sha256 = provenance.get("weight_index_sha256")
    if not isinstance(model_config_sha256, str) or not isinstance(weight_index_sha256, str):
        raise CacheUnionError(f"Model artifact hashes are missing: {expected.task_id}")
    asset_files = {
        str(item["path"]): item
        for item in source_task.model_asset_inventory.get("files", [])
        if isinstance(item, dict) and "path" in item
    }
    config_asset = asset_files.get("config.json")
    weight_index_path = source_task.model_asset_inventory.get("weight_index_path")
    weight_index_asset = asset_files.get(str(weight_index_path))
    if config_asset is None or config_asset.get("sha256") != model_config_sha256:
        raise CacheUnionError(f"Provenance model config is not in evidence: {expected.task_id}")
    if weight_index_asset is None or weight_index_asset.get("sha256") != weight_index_sha256:
        raise CacheUnionError(f"Provenance weight index is not in evidence: {expected.task_id}")
    runtime_payload = {
        field: provenance.get(field)
        for field in RUNTIME_PROVENANCE_FIELDS
        if field in provenance
    }
    runtime_fingerprint = _fingerprint(runtime_payload)
    union_entry = dict(source_entry)
    union_metadata = dict(metadata)
    union_metadata["sidecar_path"] = str(sidecar)
    union_metadata["semantic_request_fingerprint"] = semantic_request_fingerprint
    union_entry.update(
        {
            "dataset_key": expected.source_dataset,
            "split": expected.split,
            "shard_path": str(shard),
            "cache_root": str(cache_root),
            "metadata": union_metadata,
            "source_provenance": {
                "source_id": source_task.source.source_id,
                "task_id": source_task.task_id,
                "ledger_path": str(source_task.source.ledger_path),
                "source_cache_root": str(source_task.source.cache_root),
                "source_dataset_key": source_request.get("dataset_key"),
                "source_split": source_request.get("split"),
                "sidecar_path": str(sidecar),
                "semantic_request_fingerprint": semantic_request_fingerprint,
                "runtime_provenance_fingerprint": runtime_fingerprint,
                "model_asset_fingerprint": source_task.model_asset_fingerprint,
                "runtime_provenance": runtime_payload,
            },
        }
    )
    return union_entry


def _blocked_payload(
    blocked_tasks: Sequence[BlockedCacheTask],
    source_rows: dict[str, list[SourceTask]],
) -> list[dict[str, Any]]:
    payload = []
    for task in blocked_tasks:
        matches = source_rows.get(task.task_id, [])
        if len(matches) > 1:
            raise CacheUnionError(f"Blocked task is duplicated across sources: {task.task_id}")
        status = matches[0].status if matches else "not_scheduled"
        payload.append(
            {
                "task_id": task.task_id,
                "sample_id": task.sample_id,
                "prompt_id": task.prompt_id,
                "condition": task.condition,
                "reason": task.reason,
                "source_status": status,
                "exposed_as_cache_entry": False,
            }
        )
    return sorted(payload, key=lambda item: item["task_id"])


def _require_signature_compatible(
    expected_signature: dict[str, Any],
    source_signature: dict[str, Any],
) -> None:
    normalized_expected = _normalized_signature(expected_signature)
    if normalized_expected != source_signature:
        raise CacheUnionError("Source extraction signature differs from the expected plan")


def _normalized_signature(signature: dict[str, Any]) -> dict[str, Any]:
    filtered = {
        key: value
        for key, value in signature.items()
        if key not in SIGNATURE_IGNORED_FIELDS
    }
    normalized = json.loads(_canonical_json(filtered))
    if not isinstance(normalized, dict):
        raise CacheUnionError("Normalized extraction signature must be an object")
    return normalized


def _require_completed_source_ledger(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        counts = dict(connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status"))
    finally:
        connection.close()
    incomplete = sum(counts.get(status, 0) for status in ("pending", "running"))
    if incomplete:
        raise CacheUnionError(f"Source ledger is still running: {path}")


def _read_ledger_signature(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CacheUnionError(f"Source ledger does not exist: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='signature'"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise CacheUnionError(f"Source ledger has no extraction signature: {path}")
    payload = json.loads(row[0])
    if not isinstance(payload, dict):
        raise CacheUnionError(f"Source ledger signature is invalid: {path}")
    return payload


def _ledger_content_fingerprint(path: Path) -> str:
    """Hash logical SQLite content so a WAL checkpoint cannot change evidence."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = [
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        ]
        if not columns:
            raise CacheUnionError(f"Source ledger has no tasks table: {path}")
        digest = hashlib.sha256()
        digest.update(_canonical_json({"task_columns": columns}).encode())
        for row in connection.execute("SELECT key,value FROM metadata ORDER BY key"):
            digest.update(_canonical_json({"metadata": dict(row)}).encode())
        selected = ",".join(f'"{column}"' for column in columns)
        for row in connection.execute(
            f"SELECT {selected} FROM tasks ORDER BY task_id"  # noqa: S608
        ):
            digest.update(_canonical_json({"task": dict(row)}).encode())
    finally:
        connection.close()
    return digest.hexdigest()


def _semantic_code_hashes(repository: Path, code_paths: Sequence[str]) -> dict[str, str]:
    hashes = {}
    for relative in code_paths:
        path = repository / relative
        if not path.is_file():
            raise CacheUnionError(f"Extractor semantic file does not exist: {path}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _validate_evidence_code(evidence: dict[str, Any], evidence_path: Path) -> None:
    code_root = Path(str(evidence.get("code_root", ""))).expanduser().resolve()
    commit = str(evidence.get("git_commit", ""))
    code_hashes = evidence.get("code_files_sha256")
    if not code_root.is_dir() or not commit or not isinstance(code_hashes, dict):
        raise CacheUnionError(f"Extractor code evidence is incomplete: {evidence_path}")
    for relative, expected_sha256 in code_hashes.items():
        process = subprocess.run(
            ["git", "-C", str(code_root), "show", f"{commit}:{relative}"],
            check=False,
            capture_output=True,
        )
        if process.returncode != 0:
            raise CacheUnionError(f"Cannot resolve extractor code evidence: {relative}")
        if _sha256_bytes(process.stdout) != expected_sha256:
            raise CacheUnionError(f"Extractor code evidence hash mismatch: {relative}")


def _model_asset_inventory(model_path: str) -> dict[str, Any]:
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise CacheUnionError(f"Model asset directory does not exist: {root}")
    weight_index = next(
        (root / name for name in WEIGHT_INDEX_FILENAMES if (root / name).is_file()),
        None,
    )
    if weight_index is None:
        raise CacheUnionError(f"Model weight index does not exist: {root}")
    index_payload = _read_json(weight_index)
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CacheUnionError(f"Model weight index has no weight_map: {weight_index}")
    shard_names = sorted({str(value) for value in weight_map.values()})
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in shard_names):
        raise CacheUnionError(f"Model weight index contains unsafe shard paths: {weight_index}")
    runtime_assets = {
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in MODEL_RUNTIME_ASSET_SUFFIXES
        and ".git" not in path.relative_to(root).parts
    }
    referenced_shards = {root / name for name in shard_names}
    missing_shards = sorted(str(path) for path in referenced_shards if not path.is_file())
    if missing_shards:
        raise CacheUnionError(f"Model weight shards are missing: {missing_shards[:3]}")
    files = []
    for path in sorted(runtime_assets | referenced_shards):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "role": "weight_shard" if path in referenced_shards else "runtime_asset",
            }
        )
    if "config.json" not in {item["path"] for item in files}:
        raise CacheUnionError(f"Model config.json is missing: {root}")
    return {
        "model_path": str(root),
        "weight_index_path": weight_index.relative_to(root).as_posix(),
        "referenced_weight_shards": shard_names,
        "files": files,
    }


@lru_cache(maxsize=16)
def _validated_model_asset_inventory(
    model_path: str,
    expected_fingerprint: str,
) -> dict[str, Any]:
    inventory = _model_asset_inventory(model_path)
    if _fingerprint(inventory) != expected_fingerprint:
        raise CacheUnionError(f"Model asset fingerprint changed: {model_path}")
    return inventory


def _git_output(repository: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise CacheUnionError(process.stderr.strip() or "Git evidence command failed")
    return process.stdout.strip()


def _request_payload(request: PrefillRequest) -> dict[str, Any]:
    return {
        "sample_id": request.sample_id,
        "model_key": request.model_key,
        "protocol": request.protocol,
        "condition": request.condition,
        "prompt_set_key": request.prompt_set_key,
        "prompt_id": request.prompt_id,
        "dataset_key": request.dataset_key,
        "split": request.split,
        "messages": list(request.messages),
        "media_paths": dict(request.media_paths),
        "use_audio_in_video": request.use_audio_in_video,
    }


def _semantic_request(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "sample_id",
            "model_key",
            "protocol",
            "condition",
            "prompt_set_key",
            "prompt_id",
            "messages",
            "media_paths",
            "use_audio_in_video",
        )
    }


def _require_unique_task_ids(tasks: Sequence[ExpectedCacheTask]) -> None:
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise CacheUnionError("Expected cache task identities are not unique")


def _resolve_artifact(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise CacheUnionError("Cache artifact path is missing")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_from(config_path: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _union_entry_sort_key(entry: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(entry["sample_id"]),
        str(entry["prompt_id"]),
        str(entry["condition"]),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CacheUnionError(f"JSON artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CacheUnionError(f"JSON artifact must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise CacheUnionError(f"Stale union temporary file exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fingerprint(payload: Any) -> str:
    return _sha256_bytes(_canonical_json(payload).encode())


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
