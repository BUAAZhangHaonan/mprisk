"""Atomic safetensors, sidecar, and manifest output for prefill trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from mprisk.models.base_wrapper import PrefillRequest, PrefillResult

DEFAULT_PREFILL_MANIFEST = Path("manifests/unified_full_cache_manifest.json")


@dataclass(frozen=True)
class PrefillCacheArtifact:
    shard_path: Path
    sidecar_path: Path
    manifest_path: Path
    checksum: str
    entry: dict[str, Any]


@dataclass(frozen=True)
class PrefillCachePaths:
    shard_path: Path
    sidecar_path: Path
    manifest_path: Path


def write_prefill_result(
    result: PrefillResult,
    *,
    output_root: str | Path,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
    update_manifest: bool = True,
) -> PrefillCacheArtifact:
    """Persist one trajectory and optionally update the unified manifest."""
    root = Path(output_root).expanduser().resolve()
    paths = prefill_artifact_paths(
        result.request,
        output_root=root,
        manifest_path=manifest_path,
    )
    manifest = paths.manifest_path
    stem = _artifact_stem(result.request.sample_id)
    relative_dir = Path("shards") / result.request.model_key / result.request.protocol
    relative_dir = relative_dir / result.request.condition
    relative_shard = relative_dir / f"{stem}.safetensors"
    relative_sidecar = relative_dir / f"{stem}.json"
    shard = paths.shard_path
    sidecar = paths.sidecar_path

    manifest_payload = _load_manifest(manifest) if update_manifest else None
    key = (
        result.request.sample_id,
        result.request.model_key,
        result.request.protocol,
        result.request.condition,
        result.request.prompt_set_key,
        result.request.prompt_id,
    )
    existing_indices = (
        [
            index
            for index, entry in enumerate(manifest_payload["entries"])
            if _entry_key(entry) == key
        ]
        if manifest_payload is not None
        else []
    )
    if len(existing_indices) > 1:
        raise ValueError(f"Manifest contains duplicate cache entries for {key!r}")
    if existing_indices and not overwrite:
        raise FileExistsError(f"Manifest already contains cache entry {key!r}")
    if not overwrite and (shard.exists() or sidecar.exists()):
        raise FileExistsError(f"Cache artifact already exists for {key!r}")

    shard.parent.mkdir(parents=True, exist_ok=True)
    if update_manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_shard = shard.with_name(f".{shard.name}.tmp")
    tmp_sidecar = sidecar.with_name(f".{sidecar.name}.tmp")
    tmp_manifest = manifest.with_name(f".{manifest.name}.tmp") if update_manifest else None
    temporary_paths = [tmp_shard, tmp_sidecar]
    if tmp_manifest is not None:
        temporary_paths.append(tmp_manifest)
    for path in temporary_paths:
        if path.exists():
            raise FileExistsError(f"Stale temporary cache file exists: {path}")

    save_file(
        {"hidden_states": np.ascontiguousarray(result.trajectory, dtype=np.float32)},
        str(tmp_shard),
    )
    checksum = _sha256(tmp_shard)
    entry = _manifest_entry(
        result,
        root=root,
        relative_shard=relative_shard,
        relative_sidecar=relative_sidecar,
        checksum=checksum,
    )
    sidecar_payload = {
        "schema": "mprisk_prefill_cache_sidecar_v1",
        "entry": entry,
        "request": {
            "sample_id": result.request.sample_id,
            "model_key": result.request.model_key,
            "protocol": result.request.protocol,
            "condition": result.request.condition,
            "prompt_set_key": result.request.prompt_set_key,
            "prompt_id": result.request.prompt_id,
            "dataset_key": result.request.dataset_key,
            "split": result.request.split,
            "messages": list(result.request.messages),
            "media_paths": dict(result.request.media_paths),
            "use_audio_in_video": result.request.use_audio_in_video,
        },
        "provenance": dict(result.provenance),
    }
    tmp_sidecar.write_text(
        json.dumps(sidecar_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if manifest_payload is not None and tmp_manifest is not None:
        if existing_indices:
            manifest_payload["entries"][existing_indices[0]] = entry
        else:
            manifest_payload["entries"].append(entry)
        tmp_manifest.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    os.replace(tmp_shard, shard)
    os.replace(tmp_sidecar, sidecar)
    if tmp_manifest is not None:
        os.replace(tmp_manifest, manifest)
    return PrefillCacheArtifact(
        shard_path=shard,
        sidecar_path=sidecar,
        manifest_path=manifest,
        checksum=checksum,
        entry=entry,
    )


def prefill_artifact_paths(
    request: PrefillRequest,
    *,
    output_root: str | Path,
    manifest_path: str | Path | None = None,
) -> PrefillCachePaths:
    """Return deterministic artifact paths for one request without writing files."""
    root = Path(output_root).expanduser().resolve()
    manifest = _resolve_manifest(root, manifest_path)
    stem = _artifact_stem(str(request.sample_id))
    relative_dir = Path("shards") / str(request.model_key) / str(request.protocol).lower()
    relative_dir = relative_dir / str(request.condition).upper()
    return PrefillCachePaths(
        shard_path=root / relative_dir / f"{stem}.safetensors",
        sidecar_path=root / relative_dir / f"{stem}.json",
        manifest_path=manifest,
    )


def write_full_cache_manifest(entries: list[dict[str, Any]], path: str | Path) -> Path:
    """Atomically materialize a full cache manifest from validated entries."""
    destination = Path(path).expanduser().resolve()
    keys = [_entry_key(entry) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Cache manifest entries contain duplicate cache keys")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = {"schema": "mprisk_full_cache_manifest_v1", "entries": entries}
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _manifest_entry(
    result: PrefillResult,
    *,
    root: Path,
    relative_shard: Path,
    relative_sidecar: Path,
    checksum: str,
) -> dict[str, Any]:
    return {
        "sample_id": result.request.sample_id,
        "model_key": result.request.model_key,
        "protocol": result.request.protocol,
        "condition": result.request.condition,
        "prompt_set_key": result.request.prompt_set_key,
        "prompt_id": result.request.prompt_id,
        "dataset_key": result.request.dataset_key,
        "split": result.request.split,
        "shard_path": relative_shard.as_posix(),
        "index_in_shard": 0,
        "layer_count": result.layer_count,
        "hidden_dim": result.hidden_dim,
        "token_count": result.token_count,
        "t0_token_index": result.t0_token_index,
        "elapsed_seconds": result.provenance.get("elapsed_seconds"),
        "peak_gpu_memory_bytes": result.provenance.get("peak_gpu_memory_bytes"),
        "cache_root": str(root),
        "checksum": checksum,
        "metadata": {
            "tensor_key": "hidden_states",
            "t0_token_index": result.t0_token_index,
            "hidden_state_index_offset": 1,
            "sidecar_path": relative_sidecar.as_posix(),
            "use_audio_in_video": result.request.use_audio_in_video,
            "prefill_strategy": result.provenance.get("prefill_strategy"),
            "prefill_strategy_version": result.provenance.get("prefill_strategy_version"),
            "prefix_identity": result.provenance.get("prefix_identity"),
            "created_at": datetime.now(UTC).isoformat(),
        },
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": "mprisk_full_cache_manifest_v1", "entries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Cache manifest must be a JSON object: {path}")
    if payload.get("schema") != "mprisk_full_cache_manifest_v1":
        raise ValueError(f"Unsupported cache manifest schema in {path}")
    if not isinstance(payload.get("entries"), list):
        raise ValueError(f"Cache manifest entries must be a list: {path}")
    return payload


def _resolve_manifest(root: Path, manifest_path: str | Path | None) -> Path:
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_PREFILL_MANIFEST
    return path.expanduser().resolve() if path.is_absolute() else root / path


def _entry_key(entry: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(entry.get("sample_id")),
        str(entry.get("model_key")),
        str(entry.get("protocol")).lower(),
        str(entry.get("condition")).upper(),
        str(entry.get("prompt_set_key", "adhoc")),
        str(entry.get("prompt_id", "adhoc")),
    )


def _artifact_stem(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("_") or "sample"
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable[:80]}-{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
