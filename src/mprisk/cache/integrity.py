"""Deterministic, fail-closed integrity contracts for prefill cache production."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from safetensors import safe_open

CHECKPOINT_RECEIPT_SCHEMA = "mprisk_checkpoint_digest_receipt_v1"
COMPLETION_RECEIPT_SCHEMA = "mprisk_cache_completion_receipt_v1"
COMPLETION_POINTER_SCHEMA = "mprisk_cache_completion_pointer_v1"
EQUIVALENCE_WAIVER_SCHEMA = "mprisk_cache_equivalence_waiver_v1"
SIDECAR_SCHEMA = "mprisk_prefill_cache_sidecar_v1"
WEIGHT_INDEX_FILENAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
MODEL_RUNTIME_ASSET_SUFFIXES = frozenset(
    {".json", ".model", ".py", ".tiktoken", ".txt"}
)
COMMON_EXTRACTOR_PATHS = (
    "src/mprisk/assets/registry.py",
    "src/mprisk/cache/hidden_state_cache.py",
    "src/mprisk/cache/kv_prefill.py",
    "src/mprisk/cache/prefill_batch.py",
    "src/mprisk/cache/prefill_extract.py",
    "src/mprisk/cache/prefill_strategy_registry.py",
    "src/mprisk/cache/prefill_writer.py",
    "src/mprisk/models/base_wrapper.py",
    "src/mprisk/models/hf_visual_prefill.py",
    "src/mprisk/models/video_frame_utils.py",
    "src/mprisk/models/wrapper_registry.py",
    "src/mprisk/prompts/compiler.py",
    "src/mprisk/prompts/template_bank.py",
    "scripts/extract_prefill_batch.py",
)
WRAPPER_PATHS = {
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


class CacheIntegrityError(ValueError):
    """Raised when exact cache identity or content cannot be proven."""


def build_checkpoint_digest(
    model_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    write_receipt: bool = False,
) -> dict[str, Any]:
    """Hash the checkpoint index and every referenced shard.

    Existing per-file hashes are reused only when the complete stat identity is
    unchanged.  When receipt writing is enabled, progress is committed after
    each file so an interrupted audit resumes without rereading finished shards.
    """
    root = Path(model_path).expanduser().resolve()
    index_path, shard_names = _checkpoint_files(root)
    paths = [index_path, *(root / name for name in shard_names)]
    receipt = None if receipt_path is None else Path(receipt_path).expanduser().resolve()
    previous = _load_optional_json(receipt)
    reusable = {
        str(item.get("path")): item
        for item in (previous or {}).get("files", [])
        if isinstance(item, dict)
    }
    records: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        stat = _stat_identity(path)
        old = reusable.get(relative)
        sha256 = (
            str(old["sha256"])
            if old is not None
            and old.get("stat") == stat
            and _is_sha256(old.get("sha256"))
            else _sha256_file(path)
        )
        records.append(
            {
                "path": relative,
                "bytes": int(stat["size"]),
                "sha256": sha256,
                "role": "weight_index" if path == index_path else "weight_shard",
                "stat": stat,
            }
        )
        if write_receipt:
            if receipt is None:
                raise CacheIntegrityError("write_receipt requires receipt_path")
            _write_checkpoint_receipt(
                receipt,
                root=root,
                index_path=index_path,
                shard_names=shard_names,
                records=records,
                complete=False,
            )
    payload = _checkpoint_payload(
        root=root,
        index_path=index_path,
        shard_names=shard_names,
        records=records,
        complete=True,
    )
    if write_receipt:
        assert receipt is not None
        _atomic_json(receipt, payload)
    return payload


def build_extractor_semantic_digest(
    repository: str | Path,
    *,
    family: str,
    model_path: str | Path,
) -> dict[str, Any]:
    """Hash all repository and checkpoint-local Python that can affect extraction."""
    root = Path(repository).expanduser().resolve()
    try:
        wrapper = WRAPPER_PATHS[family]
    except KeyError as exc:
        raise CacheIntegrityError(f"Unsupported wrapper family: {family!r}") from exc
    paths = tuple(dict.fromkeys((*COMMON_EXTRACTOR_PATHS, wrapper)))
    missing = [relative for relative in paths if not (root / relative).is_file()]
    if missing:
        raise CacheIntegrityError(f"Extractor semantic files are missing: {missing}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise CacheIntegrityError(
            f"Extractor semantic files must be committed before use:\n{dirty}"
        )
    repository_files = {
        relative: _sha256_file(root / relative) for relative in sorted(paths)
    }
    checkpoint = Path(model_path).expanduser().resolve()
    remote_code_files = {
        path.relative_to(checkpoint).as_posix(): _sha256_file(path)
        for path in sorted(checkpoint.rglob("*.py"))
        if path.is_file()
        and not {".git", "__pycache__"} & set(path.relative_to(checkpoint).parts)
    }
    core = {
        "schema": "mprisk_extractor_semantic_digest_v1",
        "family": family,
        "repository_files_sha256": repository_files,
        "trust_remote_code_files_sha256": remote_code_files,
    }
    return {**core, "sha256": _fingerprint(core)}


def build_model_asset_inventory(
    model_path: str | Path,
    *,
    checkpoint_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Build the cache-union-compatible model asset inventory."""
    root = Path(model_path).expanduser().resolve()
    index_path, shard_names = _checkpoint_files(root)
    checkpoint_hashes = {
        str(item["path"]): str(item["sha256"])
        for item in checkpoint_receipt.get("files", [])
        if isinstance(item, dict)
    }
    referenced = {root / name for name in shard_names}
    runtime_assets = {
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in MODEL_RUNTIME_ASSET_SUFFIXES
        and ".git" not in path.relative_to(root).parts
    }
    files = []
    for path in sorted(runtime_assets | referenced):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": checkpoint_hashes.get(relative) or _sha256_file(path),
                "role": "weight_shard" if path in referenced else "runtime_asset",
            }
        )
    if "config.json" not in {item["path"] for item in files}:
        raise CacheIntegrityError(f"Model config.json is missing: {root}")
    inventory = {
        "model_path": str(root),
        "weight_index_path": index_path.relative_to(root).as_posix(),
        "referenced_weight_shards": shard_names,
        "files": files,
    }
    return {
        "inventory": inventory,
        "sha256": _fingerprint(inventory),
    }


def completion_receipt_status(
    output_root: str | Path,
    *,
    expected_signature: dict[str, Any],
    expected_tasks: int,
) -> dict[str, Any]:
    """Validate an existing completion receipt without rereading cache contents."""
    root = Path(output_root).expanduser().resolve()
    pointer_path = root / "COMPLETION_RECEIPT.json"
    if not pointer_path.is_file():
        return {"passed": False, "reason": "missing", "path": str(pointer_path)}
    try:
        pointer = _read_json(pointer_path)
        if pointer.get("schema") != COMPLETION_POINTER_SCHEMA:
            raise CacheIntegrityError("Unsupported completion pointer schema")
        receipt_path = _contained_path(root, pointer.get("receipt_path"))
        receipt = _read_json(receipt_path)
        if receipt.get("schema") != COMPLETION_RECEIPT_SCHEMA:
            raise CacheIntegrityError("Unsupported completion receipt schema")
        if _fingerprint(receipt["content"]) != receipt.get("content_sha256"):
            raise CacheIntegrityError("Completion receipt content hash mismatch")
        if receipt.get("content_sha256") != pointer.get("content_sha256"):
            raise CacheIntegrityError("Completion pointer hash mismatch")
        content = receipt["content"]
        if content.get("expected_tasks") != expected_tasks:
            raise CacheIntegrityError("Completion receipt task count mismatch")
        if content.get("expected_signature_sha256") != _fingerprint(expected_signature):
            raise CacheIntegrityError("Completion receipt signature mismatch")
        ledger = root / "batch_state.sqlite3"
        if content.get("ledger_content_sha256") != _ledger_content_fingerprint(ledger):
            raise CacheIntegrityError("Completion receipt ledger content changed")
        for item in content.get("artifacts", []):
            path = _contained_path(root, item.get("path"))
            if _stat_identity(path) != item.get("stat"):
                raise CacheIntegrityError(f"Completion artifact changed: {path}")
    except (CacheIntegrityError, KeyError, OSError, sqlite3.Error, TypeError) as exc:
        return {"passed": False, "reason": "invalid", "error": str(exc)}
    return {
        "passed": True,
        "path": str(receipt_path),
        "content_sha256": receipt["content_sha256"],
        "entries": expected_tasks,
    }


def audit_completed_cache(
    output_root: str | Path,
    *,
    expected_signature: dict[str, Any],
    expected_tasks: int,
    write_receipt: bool = False,
) -> dict[str, Any]:
    """Fully validate completed rows, optionally writing a content-addressed receipt."""
    root = Path(output_root).expanduser().resolve()
    ledger = root / "batch_state.sqlite3"
    rows, ledger_signature = _completed_rows(ledger, expected_tasks)
    _require_signature_matches(ledger_signature, expected_signature)
    artifact_records: dict[str, dict[str, Any]] = {}
    entries = []
    for row in rows:
        task_id = str(row["task_id"])
        entry = json.loads(str(row["entry_json"]))
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            raise CacheIntegrityError(f"Entry metadata is invalid: {task_id}")
        cache_root = Path(str(entry.get("cache_root", root))).expanduser().resolve()
        if not cache_root.is_relative_to(root):
            raise CacheIntegrityError(f"Entry cache root escapes output root: {task_id}")
        shard = _contained_path(cache_root, entry.get("shard_path"))
        sidecar = _contained_path(cache_root, metadata.get("sidecar_path"))
        sidecar_payload = _read_json(sidecar)
        if sidecar_payload.get("schema") != SIDECAR_SCHEMA:
            raise CacheIntegrityError(f"Unsupported sidecar schema: {task_id}")
        if sidecar_payload.get("entry") != entry:
            raise CacheIntegrityError(f"Ledger/sidecar entry mismatch: {task_id}")
        request = sidecar_payload.get("request")
        if not isinstance(request, dict):
            raise CacheIntegrityError(f"Sidecar request is invalid: {task_id}")
        for field in (
            "sample_id",
            "model_key",
            "protocol",
            "condition",
            "prompt_set_key",
            "prompt_id",
        ):
            if entry.get(field) != request.get(field) or row[field] != request.get(field):
                raise CacheIntegrityError(f"Entry/request {field} mismatch: {task_id}")
        identity = {
            "sample_id": request["sample_id"],
            "prompt_id": request["prompt_id"],
            "condition": request["condition"],
            "protocol": request["protocol"],
            "model_key": request["model_key"],
            "runtime_contracts": request.get("runtime_contracts", {}),
        }
        if _fingerprint(identity) != task_id:
            raise CacheIntegrityError(f"Task identity mismatch: {task_id}")
        shard_sha = _sha256_file(shard)
        if shard_sha != entry.get("checksum") or shard_sha != row["checksum"]:
            raise CacheIntegrityError(f"Cache shard checksum mismatch: {task_id}")
        tensor_key = metadata.get("tensor_key")
        if tensor_key != "hidden_states":
            raise CacheIntegrityError(f"Unexpected tensor key: {task_id}")
        with safe_open(str(shard), framework="np") as tensors:
            if list(tensors.keys()) != [tensor_key]:
                raise CacheIntegrityError(f"Unexpected tensors: {task_id}")
            shape = list(tensors.get_slice(tensor_key).get_shape())
        if shape != [entry.get("layer_count"), entry.get("hidden_dim")]:
            raise CacheIntegrityError(f"Tensor shape mismatch: {task_id}")
        if int(entry.get("t0_token_index", -1)) != int(entry.get("token_count", 0)) - 1:
            raise CacheIntegrityError(f"t0 token contract mismatch: {task_id}")
        sidecar_sha = _sha256_file(sidecar)
        entries.append(
            {
                "task_id": task_id,
                "entry_sha256": _fingerprint(entry),
                "request_sha256": _fingerprint(request),
                "shard_sha256": shard_sha,
                "sidecar_sha256": sidecar_sha,
            }
        )
        for path, sha256 in ((shard, shard_sha), (sidecar, sidecar_sha)):
            relative = path.relative_to(root).as_posix()
            artifact_records[relative] = {
                "path": relative,
                "sha256": sha256,
                "stat": _stat_identity(path),
            }
    task_set = [item["task_id"] for item in sorted(entries, key=lambda item: item["task_id"])]
    content = {
        "expected_tasks": expected_tasks,
        "expected_signature_sha256": _fingerprint(expected_signature),
        "ledger_signature_sha256": _fingerprint(ledger_signature),
        "ledger_content_sha256": _ledger_content_fingerprint(ledger),
        "task_set_sha256": _fingerprint(task_set),
        "entries_sha256": _fingerprint(
            sorted(entries, key=lambda item: item["task_id"])
        ),
        "artifacts": sorted(artifact_records.values(), key=lambda item: item["path"]),
    }
    content_sha256 = _fingerprint(content)
    receipt = {
        "schema": COMPLETION_RECEIPT_SCHEMA,
        "content_sha256": content_sha256,
        "content": content,
    }
    receipt_path = root / "receipts" / "completion" / f"{content_sha256}.json"
    pointer = {
        "schema": COMPLETION_POINTER_SCHEMA,
        "content_sha256": content_sha256,
        "receipt_path": receipt_path.relative_to(root).as_posix(),
    }
    if write_receipt:
        _atomic_json(receipt_path, receipt)
        _atomic_json(root / "COMPLETION_RECEIPT.json", pointer)
    return {
        "passed": True,
        "path": str(receipt_path),
        "content_sha256": content_sha256,
        "entries": len(entries),
        "receipt_written": write_receipt,
    }


def write_completion_receipt(
    output_root: str | Path,
    *,
    expected_signature: dict[str, Any],
    expected_tasks: int,
) -> dict[str, Any]:
    """Fully validate completed rows and write their completion receipt."""
    return audit_completed_cache(
        output_root,
        expected_signature=expected_signature,
        expected_tasks=expected_tasks,
        write_receipt=True,
    )


def validate_accepted_bundle(
    index_path: str | Path,
    *,
    expected_identity: dict[str, Any],
    equivalence_waiver: str | Path | None = None,
) -> dict[str, Any]:
    """Require exact bundle identity or a signed, field-exact equivalence waiver."""
    path = Path(index_path).expanduser().resolve()
    package = _read_json(path)
    if package.get("schema") != "mprisk_prefill_cache_union_v2":
        raise CacheIntegrityError("Unsupported accepted bundle schema")
    provenance = package.get("provenance")
    if not isinstance(provenance, dict):
        raise CacheIntegrityError("Accepted bundle provenance is missing")
    signature = provenance.get("expected_signature")
    if not isinstance(signature, dict):
        raise CacheIntegrityError("Accepted bundle expected signature is missing")
    if provenance.get("expected_signature_sha256") != _fingerprint(signature):
        raise CacheIntegrityError("Accepted bundle signature hash mismatch")
    entries = package.get("entries")
    if not isinstance(entries, list):
        raise CacheIntegrityError("Accepted bundle entries are missing")
    keys = [
        [
            entry.get("sample_id"),
            entry.get("prompt_id"),
            entry.get("condition"),
            entry.get("model_key"),
            str(entry.get("protocol", "")).lower(),
        ]
        for entry in entries
        if isinstance(entry, dict)
    ]
    if len(keys) != len(entries) or len({tuple(key) for key in keys}) != len(keys):
        raise CacheIntegrityError("Accepted bundle task keys are invalid or duplicated")
    actual = {
        "model_key": signature.get("model_key"),
        "family": signature.get("family"),
        "protocol": str(signature.get("protocol", "")).lower(),
        "dtype": signature.get("dtype"),
        "manifest_sha256": signature.get("manifest_sha256"),
        "prompt_set_sha256": signature.get("prompt_set_sha256"),
        "prompt_ids": signature.get("prompt_ids"),
        "conditions": signature.get("conditions"),
        "model_path": signature.get("model_path"),
        "prefill_strategy": provenance.get("prefill_strategy"),
        "prefill_strategy_version": provenance.get("prefill_strategy_version"),
        "expected_tasks": len(entries),
        "task_set_sha256": _fingerprint(sorted(keys)),
        "model_asset_fingerprint": provenance.get("model_asset_fingerprint"),
        "extractor_semantic_fingerprint": provenance.get(
            "extractor_semantic_fingerprint"
        ),
    }
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected_identity.items()
        if actual.get(key) != value
    }
    if mismatches:
        if equivalence_waiver is None:
            raise CacheIntegrityError(
                f"Accepted bundle identity mismatch: {sorted(mismatches)}"
            )
        _validate_equivalence_waiver(
            Path(equivalence_waiver).expanduser().resolve(),
            index_path=path,
            expected_identity=expected_identity,
            actual_identity=actual,
            mismatches=mismatches,
        )
    return {
        "passed": True,
        "index_path": str(path),
        "index_sha256": _sha256_file(path),
        "task_count": len(entries),
        "waived_fields": sorted(mismatches),
    }


def _validate_equivalence_waiver(
    path: Path,
    *,
    index_path: Path,
    expected_identity: dict[str, Any],
    actual_identity: dict[str, Any],
    mismatches: dict[str, Any],
) -> None:
    waiver = _read_json(path)
    if waiver.get("schema") != EQUIVALENCE_WAIVER_SCHEMA:
        raise CacheIntegrityError("Unsupported equivalence waiver schema")
    payload = {key: value for key, value in waiver.items() if key != "payload_sha256"}
    if waiver.get("payload_sha256") != _fingerprint(payload):
        raise CacheIntegrityError("Equivalence waiver signature is invalid")
    expected = {
        "accepted_index_sha256": _sha256_file(index_path),
        "accepted_identity_sha256": _fingerprint(actual_identity),
        "expected_identity_sha256": _fingerprint(expected_identity),
        "waived_fields": sorted(mismatches),
    }
    for key, value in expected.items():
        if waiver.get(key) != value:
            raise CacheIntegrityError(f"Equivalence waiver {key} mismatch")
    if not isinstance(waiver.get("reason"), str) or not waiver["reason"].strip():
        raise CacheIntegrityError("Equivalence waiver reason is required")
    if not isinstance(waiver.get("approved_by"), str) or not waiver["approved_by"].strip():
        raise CacheIntegrityError("Equivalence waiver approved_by is required")


def _checkpoint_files(root: Path) -> tuple[Path, list[str]]:
    if not root.is_dir():
        raise CacheIntegrityError(f"Model directory does not exist: {root}")
    index_path = next(
        (root / name for name in WEIGHT_INDEX_FILENAMES if (root / name).is_file()),
        None,
    )
    if index_path is None:
        raise CacheIntegrityError(f"Model weight index does not exist: {root}")
    payload = _read_json(index_path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CacheIntegrityError(f"Weight index has no weight_map: {index_path}")
    shard_names = sorted({str(value) for value in weight_map.values()})
    for name in shard_names:
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise CacheIntegrityError(f"Unsafe checkpoint shard path: {name}")
        if not (root / candidate).is_file():
            raise CacheIntegrityError(f"Checkpoint shard is missing: {name}")
    return index_path, shard_names


def _checkpoint_payload(
    *,
    root: Path,
    index_path: Path,
    shard_names: list[str],
    records: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    deterministic_files = [
        {key: item[key] for key in ("path", "bytes", "sha256", "role")}
        for item in records
    ]
    core = {
        "weight_index_path": index_path.relative_to(root).as_posix(),
        "referenced_weight_shards": shard_names,
        "files": deterministic_files,
    }
    return {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "model_path": str(root),
        "complete": complete and len(records) == len(shard_names) + 1,
        "checkpoint_sha256": _fingerprint(core),
        "files": records,
    }


def _write_checkpoint_receipt(
    path: Path,
    *,
    root: Path,
    index_path: Path,
    shard_names: list[str],
    records: list[dict[str, Any]],
    complete: bool,
) -> None:
    _atomic_json(
        path,
        _checkpoint_payload(
            root=root,
            index_path=index_path,
            shard_names=shard_names,
            records=records,
            complete=complete,
        ),
    )


def _completed_rows(
    ledger: Path, expected_tasks: int
) -> tuple[list[sqlite3.Row], dict[str, Any]]:
    if not ledger.is_file():
        raise CacheIntegrityError(f"Cache ledger does not exist: {ledger}")
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM tasks ORDER BY task_id"
        ).fetchall()
        metadata = connection.execute(
            "SELECT value FROM metadata WHERE key='signature'"
        ).fetchone()
    finally:
        connection.close()
    if len(rows) != expected_tasks:
        raise CacheIntegrityError(
            f"Cache ledger has {len(rows)} tasks; expected {expected_tasks}"
        )
    invalid = [
        str(row["task_id"])
        for row in rows
        if row["status"] != "completed" or row["entry_json"] is None
    ]
    if invalid:
        raise CacheIntegrityError(f"Cache ledger has incomplete tasks: {invalid[:3]}")
    if metadata is None:
        raise CacheIntegrityError("Cache ledger signature is missing")
    signature = json.loads(str(metadata["value"]))
    if not isinstance(signature, dict):
        raise CacheIntegrityError("Cache ledger signature is invalid")
    return rows, signature


def _require_signature_matches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise CacheIntegrityError(
            f"Cache ledger signature mismatch: {sorted(mismatches)}"
        )


def _ledger_content_fingerprint(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = [
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        ]
        if not columns:
            raise CacheIntegrityError(f"Ledger has no tasks table: {path}")
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


def _contained_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise CacheIntegrityError("Cache artifact path is missing")
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise CacheIntegrityError(f"Cache artifact escapes root: {path}")
    if not path.is_file():
        raise CacheIntegrityError(f"Cache artifact does not exist: {path}")
    return path


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CacheIntegrityError(f"JSON artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CacheIntegrityError(f"Expected JSON object: {path}")
    return payload


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        return _read_json(path)
    except (CacheIntegrityError, json.JSONDecodeError, OSError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
