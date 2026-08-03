"""Fail-closed migration of proven-equivalent cache asset signatures."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mprisk.cache.cache_matrix_queue import (
    _expected_batch_signature,
    _ledger_status,
    build_asset_signature,
    load_matrix_config,
)
from mprisk.cache.integrity import (
    CacheIntegrityError,
    audit_completed_cache,
    build_checkpoint_digest,
    build_model_asset_inventory,
)


MIGRATION_MANIFEST_SCHEMA = "mprisk_cache_asset_signature_migration_v1"
MIGRATION_RECORD_SCHEMA = "mprisk_cache_asset_signature_migration_record_v1"
CODE_EVIDENCE_SCHEMA = "mprisk_cache_semantic_diff_evidence_v1"
PROBE_SCHEMA = "mprisk_cache_signature_equivalence_probe_v1"
REQUIRED_PROBE_FIELDS = (
    "checksum",
    "layer_count",
    "hidden_dim",
    "token_count",
    "t0_token_index",
    "model_key",
    "protocol",
    "prompt_id",
    "condition",
    "sample_id",
)
ALLOWED_MODALITIES = frozenset({"audio", "image", "text", "video"})
LEGACY_V2_SCHEMA = "mprisk_cache_asset_signature_v2"
CURRENT_V3_SCHEMA = "mprisk_cache_asset_signature_v3"
V3_PROVENANCE_FIELDS = (
    "checkpoint_digest_receipt",
    "checkpoint_digest_schema",
    "checkpoint_sha256",
    "extractor_semantic_files",
    "extractor_semantic_schema",
    "extractor_semantic_sha256",
    "model_asset_fingerprint",
)

# A classification is accepted only when every protected prefill symbol remains
# AST-identical across the commits bound by the evidence file. Unknown changed
# files and incomplete symbol lists are rejected.
PROTECTED_SYMBOLS = {
    (
        "generation_only",
        "src/mprisk/models/base_wrapper.py",
    ): (
        "PrefillRequest",
        "PrefillResult",
        "BaseModelWrapper.extract_prefill",
    ),
    (
        "generation_only",
        "src/mprisk/models/hf_visual_prefill.py",
    ): (
        "HfVisualPrefillWrapper._forward_model",
        "HfVisualPrefillWrapper._prepare_inputs",
        "HfVisualPrefillWrapper.extract_prefill",
        "_token_position",
        "_trajectory_from_outputs",
    ),
    (
        "allocator_provenance_only",
        "src/mprisk/models/phi4_mm.py",
    ): (
        "Phi4MmWrapper._prepare_modal_inputs",
        "_token_position",
        "_trajectory_from_outputs",
    ),
    (
        "python_timezone_alias_only",
        "src/mprisk/cache/prefill_writer.py",
    ): ("write_full_cache_manifest",),
    (
        "llava_onevision_class_move_only",
        "src/mprisk/models/llava.py",
    ): (
        "LlavaV15Wrapper._prepare_inputs",
        "_validate_llava_v15_sampled_frames",
        "_validate_llava_v15_processor_tokens",
    ),
    (
        "llava_onevision_limit_boundary_only",
        "src/mprisk/models/llava_onevision.py",
    ): (
        "LlavaOneVisionWrapper._validate_request",
        "LlavaOneVisionWrapper._prepare_inputs",
    ),
    (
        "gemma4_same_media_and_provenance_only",
        "src/mprisk/models/gemma4.py",
    ): (
        "Gemma4Wrapper.load",
        "_move_inputs_to_device",
        "_require_attention_mask",
        "_media_keys",
    ),
    (
        "generation_allocator_provenance_only",
        "src/mprisk/models/phi4_mm.py",
    ): (
        "Phi4MmWrapper._prepare_modal_inputs",
        "_token_position",
        "_trajectory_from_outputs",
    ),
    (
        "generation_only",
        "src/mprisk/models/qwen_omni.py",
    ): (
        "QwenOmniWrapper.extract_prefill",
        "_require_attention_mask",
        "_move_inputs_to_device",
    ),
}


class SignatureMigrationError(CacheIntegrityError):
    """Raised when any equivalence precondition cannot be proven."""


def migrate_asset_signature(
    config_path: str | Path,
    manifest_path: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Validate one explicit migration manifest and optionally apply it."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest_bytes = manifest_file.read_bytes()
    manifest = _read_json(manifest_file)
    if manifest.get("schema") != MIGRATION_MANIFEST_SCHEMA:
        raise SignatureMigrationError("Unsupported signature migration manifest schema")
    config = load_matrix_config(Path(config_path).expanduser().resolve())
    job_id = _required_text(manifest, "job_id")
    jobs = [job for job in config.jobs if job.job_id == job_id]
    if len(jobs) != 1:
        raise SignatureMigrationError(f"Manifest job_id is not unique in config: {job_id}")
    job = jobs[0]
    if Path(_required_text(manifest, "output_root")).resolve() != job.output_root.resolve():
        raise SignatureMigrationError("Manifest output_root does not match the configured job")

    current_head = _git(config.repo_root, "rev-parse", "HEAD").strip()
    if manifest.get("current_git_sha") != current_head:
        raise SignatureMigrationError("Manifest current_git_sha does not match HEAD")
    old_signature_path = job.asset_signature_evidence
    old_signature_bytes = old_signature_path.read_bytes()
    old_signature = _read_json(old_signature_path)
    current_signature = build_asset_signature(config, job.model)
    old_signature_sha256 = _fingerprint(old_signature)
    current_signature_sha256 = _fingerprint(current_signature)
    if manifest.get("old_signature_sha256") != old_signature_sha256:
        raise SignatureMigrationError("Manifest old_signature_sha256 mismatch")
    if manifest.get("current_expected_signature_sha256") != current_signature_sha256:
        raise SignatureMigrationError("Manifest current expected signature hash mismatch")
    if old_signature == current_signature:
        raise SignatureMigrationError("Asset signature already matches; migration is unnecessary")

    signature_provenance = _verify_signature_provenance(
        config.repo_root,
        old_signature_path=old_signature_path,
        old_signature=old_signature,
        current_signature=current_signature,
        manifest=manifest,
        current_head=current_head,
    )

    evidence_bundle = _verify_file_reference(
        manifest.get("evidence_bundle"), label="evidence_bundle"
    )
    code_evidence_path = _verify_file_reference(
        manifest.get("code_diff_evidence"), label="code_diff_evidence"
    )
    code_evidence = _read_json(code_evidence_path)
    code_result = _verify_code_evidence(
        config.repo_root,
        old_signature=old_signature,
        current_signature=current_signature,
        evidence=code_evidence,
        current_head=current_head,
        historical_files=signature_provenance.get(
            "baseline_repository_files_sha256"
        ),
    )
    if (
        signature_provenance.get("baseline_git_sha") is not None
        and code_result["base_git_sha"] != signature_provenance["baseline_git_sha"]
    ):
        raise SignatureMigrationError("Code evidence base does not match legacy baseline")
    probe_results = _verify_probes(
        manifest,
        job=job,
        current_head=current_head,
    )
    _verify_inactive(job, manifest)

    ledger = _verify_complete_ledger(job)
    domain_guards = _verify_domain_guards(
        manifest,
        job=job,
        old_signature=old_signature,
        current_signature=current_signature,
        ledger=ledger,
    )
    ledger_provenance = _verify_ledger_provenance(
        manifest,
        job=job,
        current_signature=current_signature,
        ledger=ledger,
    )

    expected_batch_signature = _expected_batch_signature(config, job)
    output_root_hash = hashlib.sha256(str(job.output_root).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / (
        f"mprisk-signature-migration-{output_root_hash}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SignatureMigrationError("Another signature migration holds the job lock") from exc
        _verify_inactive(job, manifest)
        before = audit_completed_cache(
            job.output_root,
            expected_signature=expected_batch_signature,
            expected_tasks=job.domain.expected_tasks,
            write_receipt=False,
        )
        if before.get("passed") is not True:
            raise SignatureMigrationError("Official pre-migration content audit did not pass")
        expected_payload_hash = _required_text(manifest, "payload_tree_sha256")
        if before.get("payload_tree_sha256") != expected_payload_hash:
            raise SignatureMigrationError(
                "Frozen payload tree hash does not match the official audit"
            )

        base_report = {
            "schema": MIGRATION_RECORD_SCHEMA,
            "mode": "apply" if apply else "dry_run",
            "job_id": job.job_id,
            "manifest_path": str(manifest_file),
            "manifest_file_sha256": _sha256_bytes(manifest_bytes),
            "evidence_bundle_path": str(evidence_bundle),
            "code_diff": code_result,
            "signature_provenance": signature_provenance,
            "domain_guards": domain_guards,
            "ledger_provenance": ledger_provenance,
            "probes": probe_results,
            "old_signature_sha256": old_signature_sha256,
            "current_expected_signature_sha256": current_signature_sha256,
            "payload_tree_sha256_before": before["payload_tree_sha256"],
            "ledger": ledger,
        }
        if not apply:
            return {**base_report, "status": "dry_run_passed"}
        return _apply_migration(
            job=job,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
            old_signature_bytes=old_signature_bytes,
            current_signature=current_signature,
            expected_batch_signature=expected_batch_signature,
            before=before,
            base_report=base_report,
        )


def _apply_migration(
    *,
    job: Any,
    manifest: dict[str, Any],
    manifest_bytes: bytes,
    old_signature_bytes: bytes,
    current_signature: dict[str, Any],
    expected_batch_signature: dict[str, Any],
    before: dict[str, Any],
    base_report: dict[str, Any],
) -> dict[str, Any]:
    migration_id = _fingerprint(
        {
            "job_id": job.job_id,
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "old_signature_sha256": base_report["old_signature_sha256"],
            "current_expected_signature_sha256": base_report[
                "current_expected_signature_sha256"
            ],
        }
    )
    pointer_path = job.output_root / "COMPLETION_RECEIPT.json"
    old_pointer_bytes = pointer_path.read_bytes() if pointer_path.is_file() else None
    old_receipt_copy = None
    if old_pointer_bytes is not None:
        pointer = json.loads(old_pointer_bytes)
        receipt_path = (job.output_root / str(pointer["receipt_path"])).resolve()
        if not receipt_path.is_relative_to(job.output_root.resolve()) or not receipt_path.is_file():
            raise SignatureMigrationError("Existing completion receipt pointer is invalid")
        old_receipt_copy = receipt_path.read_bytes()
    record_root = job.output_root / "receipts" / "signature_migrations" / migration_id
    if record_root.exists():
        raise SignatureMigrationError(f"Migration record already exists: {record_root}")
    record_root.mkdir(parents=True)
    if old_receipt_copy is not None:
        _atomic_bytes(record_root / "PREVIOUS_COMPLETION_RECEIPT.json", old_receipt_copy)
        _atomic_bytes(record_root / "PREVIOUS_COMPLETION_POINTER.json", old_pointer_bytes)
    _atomic_bytes(record_root / "PREVIOUS_ASSET_SIGNATURE.json", old_signature_bytes)
    _atomic_bytes(record_root / "MIGRATION_MANIFEST.json", manifest_bytes)
    prepared = {**base_report, "migration_id": migration_id, "status": "prepared"}
    _atomic_json(record_root / "MIGRATION_RECORD.json", prepared)

    signature_changed = False
    try:
        _atomic_json(job.asset_signature_evidence, current_signature)
        signature_changed = True
        if _read_json(job.asset_signature_evidence) != current_signature:
            raise SignatureMigrationError("Atomic asset signature replacement did not persist")
        after = audit_completed_cache(
            job.output_root,
            expected_signature=expected_batch_signature,
            expected_tasks=job.domain.expected_tasks,
            write_receipt=True,
        )
        if after.get("passed") is not True:
            raise SignatureMigrationError("Official post-migration content audit did not pass")
        if after.get("payload_tree_sha256") != before.get("payload_tree_sha256"):
            raise SignatureMigrationError("Cache payload tree changed during migration")
        completed = {
            **base_report,
            "migration_id": migration_id,
            "status": "complete",
            "payload_tree_sha256_after": after["payload_tree_sha256"],
            "completion_receipt": after,
            "previous_completion_receipt_preserved": old_receipt_copy is not None,
            "record_root": str(record_root),
        }
        _atomic_json(record_root / "MIGRATION_RECORD.json", completed)
        return completed
    except Exception as exc:
        if signature_changed:
            _atomic_bytes(job.asset_signature_evidence, old_signature_bytes)
        if old_pointer_bytes is None:
            pointer_path.unlink(missing_ok=True)
        else:
            _atomic_bytes(pointer_path, old_pointer_bytes)
        failed = {
            **prepared,
            "status": "rolled_back",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        _atomic_json(record_root / "MIGRATION_RECORD.json", failed)
        raise


def _verify_signature_provenance(
    repo_root: Path,
    *,
    old_signature_path: Path,
    old_signature: dict[str, Any],
    current_signature: dict[str, Any],
    manifest: dict[str, Any],
    current_head: str,
) -> dict[str, Any]:
    schema = old_signature.get("schema")
    if current_signature.get("schema") != CURRENT_V3_SCHEMA:
        raise SignatureMigrationError("Current signature is not the required v3 schema")
    legacy = manifest.get("legacy_v2")
    if schema == CURRENT_V3_SCHEMA:
        if legacy is not None:
            raise SignatureMigrationError("legacy_v2 evidence is forbidden for a v3 signature")
        return {"old_schema": CURRENT_V3_SCHEMA, "mode": "field_exact_v3"}
    if schema != LEGACY_V2_SCHEMA:
        raise SignatureMigrationError(f"Unsupported old signature schema: {schema!r}")
    if not isinstance(legacy, dict):
        raise SignatureMigrationError("legacy_v2 evidence is required for a v2 signature")

    stat = old_signature_path.stat()
    actual_stat = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(old_signature_path),
    }
    if legacy.get("asset_signature_stat") != actual_stat:
        raise SignatureMigrationError("legacy_v2 ASSET_SIGNATURE stat evidence mismatch")

    old_keys = set(old_signature)
    current_keys = set(current_signature)
    missing = sorted(current_keys - old_keys)
    if missing != list(V3_PROVENANCE_FIELDS):
        raise SignatureMigrationError(f"Unexpected v2-to-v3 missing fields: {missing}")
    if legacy.get("missing_v3_fields") != missing:
        raise SignatureMigrationError("legacy_v2 missing-field evidence mismatch")
    allowed_changed = ["schema", "wrapper_file_sha256", "wrapper_git_sha"]
    if legacy.get("allowed_changed_fields") != allowed_changed:
        raise SignatureMigrationError("legacy_v2 allowed changed fields are not exact")
    changed_overlap = sorted(
        key for key in old_keys & current_keys if old_signature[key] != current_signature[key]
    )
    if set(changed_overlap) - set(allowed_changed):
        raise SignatureMigrationError(
            f"legacy_v2 changed fields exceed the allowed set: {changed_overlap}"
        )
    if legacy.get("changed_overlap_fields") != changed_overlap:
        raise SignatureMigrationError("legacy_v2 changed overlap field list mismatch")
    overlap = sorted((old_keys & current_keys) - set(allowed_changed))
    mismatches = [key for key in overlap if old_signature[key] != current_signature[key]]
    if mismatches:
        raise SignatureMigrationError(f"legacy_v2 overlapping fields changed: {mismatches}")
    if legacy.get("equal_overlap_fields") != overlap:
        raise SignatureMigrationError("legacy_v2 equal overlap field list mismatch")
    if legacy.get("equal_overlap_sha256") != _fingerprint(
        {key: old_signature[key] for key in overlap}
    ):
        raise SignatureMigrationError("legacy_v2 overlap evidence hash mismatch")

    baseline = _required_text(legacy, "baseline_git_sha")
    expected_baseline = _git(
        repo_root,
        "rev-list",
        "-1",
        f"--before=@{stat.st_mtime_ns // 1_000_000_000}",
        current_head,
    ).strip()
    if baseline != expected_baseline:
        raise SignatureMigrationError("legacy_v2 baseline is not the last commit before ASSET_SIGNATURE")
    wrapper_path = _required_text(old_signature, "wrapper_path")
    wrapper_source = _git_bytes(repo_root, "show", f"{baseline}:{wrapper_path}")
    if _sha256_bytes(wrapper_source) != old_signature.get("wrapper_file_sha256"):
        raise SignatureMigrationError("legacy_v2 baseline wrapper does not match old signature")
    wrapper_commit = _required_text(old_signature, "wrapper_git_sha")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", wrapper_commit, baseline],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise SignatureMigrationError("legacy_v2 wrapper commit is not an ancestor of baseline")

    current_files = _semantic_repository_files(current_signature)
    baseline_files = {
        path: _sha256_bytes(_git_bytes(repo_root, "show", f"{baseline}:{path}"))
        for path in current_files
    }
    if legacy.get("baseline_repository_files_sha256") != baseline_files:
        raise SignatureMigrationError("legacy_v2 baseline semantic file evidence mismatch")

    model_root = Path(_required_text(current_signature, "model_path")).resolve()
    checkpoint = build_checkpoint_digest(model_root)
    inventory = build_model_asset_inventory(model_root, checkpoint_receipt=checkpoint)
    asset_files = []
    for item in inventory["inventory"]["files"]:
        path = model_root / str(item["path"])
        file_stat = path.stat()
        if file_stat.st_mtime_ns > stat.st_mtime_ns or file_stat.st_ctime_ns > stat.st_mtime_ns:
            raise SignatureMigrationError(f"Model asset is newer than legacy signature: {path}")
        asset_files.append(
            {
                "path": str(item["path"]),
                "bytes": int(item["bytes"]),
                "sha256": str(item["sha256"]),
                "role": str(item["role"]),
                "mtime_ns": file_stat.st_mtime_ns,
                "ctime_ns": file_stat.st_ctime_ns,
            }
        )
    asset_guard = {
        "file_count": len(asset_files),
        "latest_mtime_ns": max(item["mtime_ns"] for item in asset_files),
        "latest_ctime_ns": max(item["ctime_ns"] for item in asset_files),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "model_asset_fingerprint": inventory["sha256"],
        "files_sha256": _fingerprint(asset_files),
    }
    if legacy.get("model_asset_age_guard") != asset_guard:
        raise SignatureMigrationError("legacy_v2 model asset age evidence mismatch")
    if checkpoint["checkpoint_sha256"] != current_signature.get("checkpoint_sha256"):
        raise SignatureMigrationError("Current checkpoint hash changed during legacy verification")
    if inventory["sha256"] != current_signature.get("model_asset_fingerprint"):
        raise SignatureMigrationError("Current model asset fingerprint changed during verification")
    return {
        "old_schema": LEGACY_V2_SCHEMA,
        "mode": "historical_baseline_and_asset_age_proof",
        "asset_signature_stat": actual_stat,
        "baseline_git_sha": baseline,
        "equal_overlap_fields": overlap,
        "changed_overlap_fields": changed_overlap,
        "missing_v3_fields": missing,
        "model_asset_age_guard": asset_guard,
        "baseline_repository_files_sha256": baseline_files,
    }


def _verify_code_evidence(
    repo_root: Path,
    *,
    old_signature: dict[str, Any],
    current_signature: dict[str, Any],
    evidence: dict[str, Any],
    current_head: str,
    historical_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    if evidence.get("schema") != CODE_EVIDENCE_SCHEMA:
        raise SignatureMigrationError("Unsupported code diff evidence schema")
    base = _required_text(evidence, "base_git_sha")
    head = _required_text(evidence, "head_git_sha")
    if head != current_head:
        raise SignatureMigrationError("Code evidence head does not match current HEAD")
    old_files = (
        _semantic_repository_files(old_signature)
        if historical_files is None
        else historical_files
    )
    current_files = _semantic_repository_files(current_signature)
    if set(old_files) != set(current_files):
        raise SignatureMigrationError(
            "Old signature lacks field-exact v3 semantic-file provenance"
        )
    changed_paths = sorted(
        path for path in old_files if old_files[path] != current_files[path]
    )
    items = evidence.get("changed_files")
    if not isinstance(items, list):
        raise SignatureMigrationError("Code evidence changed_files is missing")
    if not changed_paths:
        if items:
            raise SignatureMigrationError(
                "Code evidence does not cover exact semantic diff paths"
            )
        provenance_only = _verify_wrapper_provenance_only(
            repo_root,
            old_signature=old_signature,
            current_signature=current_signature,
            evidence=evidence.get("provenance_only"),
            base_git_sha=base,
            current_head=current_head,
        )
        return {
            "base_git_sha": base,
            "head_git_sha": head,
            "changed_files": [],
            "provenance_only": provenance_only,
        }
    if not items:
        raise SignatureMigrationError("Code evidence changed_files is missing")
    by_path = {item.get("path"): item for item in items if isinstance(item, dict)}
    if sorted(by_path) != changed_paths or len(by_path) != len(items):
        raise SignatureMigrationError("Code evidence does not cover exact semantic diff paths")
    verified = []
    for path in changed_paths:
        item = by_path[path]
        classification = _required_text(item, "classification")
        required_symbols = PROTECTED_SYMBOLS.get((classification, path))
        if required_symbols is None:
            raise SignatureMigrationError(
                f"Unknown semantic diff classification: {classification}:{path}"
            )
        listed_symbols = item.get("protected_symbols")
        if listed_symbols != list(required_symbols):
            raise SignatureMigrationError(f"Incomplete protected symbol list: {path}")
        old_source = _git_bytes(repo_root, "show", f"{base}:{path}")
        current_source = (repo_root / path).read_bytes()
        if _sha256_bytes(old_source) != old_files[path]:
            raise SignatureMigrationError(f"Base git blob does not match old signature: {path}")
        if _sha256_bytes(current_source) != current_files[path]:
            raise SignatureMigrationError(f"HEAD file does not match current signature: {path}")
        if item.get("old_file_sha256") != old_files[path]:
            raise SignatureMigrationError(f"Code evidence old file hash mismatch: {path}")
        if item.get("current_file_sha256") != current_files[path]:
            raise SignatureMigrationError(f"Code evidence current file hash mismatch: {path}")
        diff = _git_bytes(repo_root, "diff", "--binary", base, head, "--", path)
        if item.get("git_diff_sha256") != _sha256_bytes(diff):
            raise SignatureMigrationError(f"Code evidence diff hash mismatch: {path}")
        old_normalized = _classification_ast_sha256(
            old_source, path=path, classification=classification
        )
        current_normalized = _classification_ast_sha256(
            current_source, path=path, classification=classification
        )
        if old_normalized != current_normalized:
            raise SignatureMigrationError(
                f"Semantic diff exceeds declared classification: {classification}:{path}"
            )
        if item.get("normalized_module_ast_sha256") != old_normalized:
            raise SignatureMigrationError(f"Normalized module AST evidence mismatch: {path}")
        ast_hashes = {}
        for symbol in required_symbols:
            old_ast = _symbol_ast_sha256(old_source, symbol)
            current_ast = _symbol_ast_sha256(current_source, symbol)
            if old_ast != current_ast:
                raise SignatureMigrationError(
                    f"Protected prefill symbol changed under {classification}: {path}:{symbol}"
                )
            ast_hashes[symbol] = old_ast
        if item.get("protected_symbol_ast_sha256") != ast_hashes:
            raise SignatureMigrationError(f"Protected AST evidence mismatch: {path}")
        verified.append(
            {
                "path": path,
                "classification": classification,
                "git_diff_sha256": item["git_diff_sha256"],
                "normalized_module_ast_sha256": old_normalized,
                "protected_symbol_ast_sha256": ast_hashes,
            }
        )
    return {"base_git_sha": base, "head_git_sha": head, "changed_files": verified}


def _verify_wrapper_provenance_only(
    repo_root: Path,
    *,
    old_signature: dict[str, Any],
    current_signature: dict[str, Any],
    evidence: Any,
    base_git_sha: str,
    current_head: str,
) -> dict[str, str]:
    changed_fields = sorted(
        key
        for key in set(old_signature) | set(current_signature)
        if old_signature.get(key) != current_signature.get(key)
    )
    if changed_fields != ["wrapper_git_sha"]:
        raise SignatureMigrationError(
            "Provenance-only migration requires wrapper_git_sha as the sole signature change"
        )
    if old_signature.get("schema") != CURRENT_V3_SCHEMA:
        raise SignatureMigrationError("Provenance-only migration requires a v3 signature")
    wrapper_path = _required_text(current_signature, "wrapper_path")
    if old_signature.get("wrapper_path") != wrapper_path:
        raise SignatureMigrationError("Wrapper path changed under provenance-only migration")
    wrapper_sha256 = _required_text(current_signature, "wrapper_file_sha256")
    if old_signature.get("wrapper_file_sha256") != wrapper_sha256:
        raise SignatureMigrationError("Wrapper bytes changed under provenance-only migration")
    extractor_sha256 = _required_text(current_signature, "extractor_semantic_sha256")
    if old_signature.get("extractor_semantic_sha256") != extractor_sha256:
        raise SignatureMigrationError(
            "Extractor semantics changed under provenance-only migration"
        )
    old_wrapper_git_sha = _required_text(old_signature, "wrapper_git_sha")
    current_wrapper_git_sha = _required_text(current_signature, "wrapper_git_sha")
    expected = {
        "kind": "wrapper_git_sha_only",
        "wrapper_path": wrapper_path,
        "old_wrapper_git_sha": old_wrapper_git_sha,
        "current_wrapper_git_sha": current_wrapper_git_sha,
        "wrapper_file_sha256": wrapper_sha256,
        "extractor_semantic_sha256": extractor_sha256,
    }
    if evidence != expected:
        raise SignatureMigrationError("Wrapper provenance-only evidence mismatch")
    current_path = repo_root / wrapper_path
    if _sha256_file(current_path) != wrapper_sha256:
        raise SignatureMigrationError("Current wrapper bytes do not match signature")
    old_wrapper_bytes = _required_git_blob(
        repo_root,
        commit=old_wrapper_git_sha,
        path=wrapper_path,
        label="Old wrapper commit",
    )
    current_wrapper_bytes = _required_git_blob(
        repo_root,
        commit=current_wrapper_git_sha,
        path=wrapper_path,
        label="Current wrapper commit",
    )
    if _sha256_bytes(old_wrapper_bytes) != wrapper_sha256:
        raise SignatureMigrationError("Old wrapper commit does not reproduce wrapper bytes")
    if _sha256_bytes(current_wrapper_bytes) != wrapper_sha256:
        raise SignatureMigrationError("Current wrapper commit does not reproduce wrapper bytes")
    _require_git_ancestor(
        repo_root,
        ancestor=old_wrapper_git_sha,
        descendant=base_git_sha,
        label="Old wrapper commit",
    )
    _require_git_ancestor(
        repo_root,
        ancestor=base_git_sha,
        descendant=current_head,
        label="Code evidence base",
    )
    _require_git_ancestor(
        repo_root,
        ancestor=current_wrapper_git_sha,
        descendant=current_head,
        label="Current wrapper commit",
    )
    _require_git_ancestor(
        repo_root,
        ancestor=old_wrapper_git_sha,
        descendant=current_wrapper_git_sha,
        label="Old wrapper commit",
    )
    observed_wrapper_git_sha = _git(
        repo_root, "log", "-1", "--format=%H", "--", wrapper_path
    ).strip()
    if observed_wrapper_git_sha != current_wrapper_git_sha:
        raise SignatureMigrationError("Current wrapper provenance is stale")
    return expected


def _required_git_blob(
    repo_root: Path, *, commit: str, path: str, label: str
) -> bytes:
    try:
        return _git_bytes(repo_root, "show", f"{commit}:{path}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SignatureMigrationError(f"{label} blob is not readable") from exc


def _require_git_ancestor(
    repo_root: Path, *, ancestor: str, descendant: str, label: str
) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SignatureMigrationError(
            f"{label} is not an ancestor of required provenance"
        )


def _verify_probes(
    manifest: dict[str, Any], *, job: Any, current_head: str
) -> list[dict[str, Any]]:
    probes = manifest.get("probes")
    if not isinstance(probes, list) or not probes:
        raise SignatureMigrationError("At least one current-HEAD canary probe is required")
    results = []
    for probe in probes:
        if not isinstance(probe, dict):
            raise SignatureMigrationError("Probe must be an object")
        coverage = probe.get("coverage")
        if not isinstance(coverage, dict):
            raise SignatureMigrationError("Probe coverage is required")
        modalities = coverage.get("modalities")
        if (
            not isinstance(modalities, list)
            or not modalities
            or set(modalities) - ALLOWED_MODALITIES
        ):
            raise SignatureMigrationError("Probe modalities are missing or unknown")
        resolution = coverage.get("resolution_contract")
        if (
            not isinstance(resolution, str)
            or not resolution.strip()
            or "unknown" in resolution.lower()
        ):
            raise SignatureMigrationError("Probe resolution contract is missing or unknown")
        differences = coverage.get("known_differences")
        if not isinstance(differences, list) or any(not isinstance(v, str) for v in differences):
            raise SignatureMigrationError("Probe known_differences must be explicit")
        report_path = _verify_file_reference(probe.get("comparison_report"), label="probe report")
        report = _read_json(report_path)
        if report.get("schema") != PROBE_SCHEMA:
            raise SignatureMigrationError("Unsupported probe report schema")
        sample_id = _required_text(probe, "sample_id")
        if report.get("git_sha") != current_head or report.get("sample_id") != sample_id:
            raise SignatureMigrationError("Probe report is not bound to current HEAD/sample")
        if report.get("model_key") != job.model.model_key:
            raise SignatureMigrationError("Probe report model mismatch")
        if report.get("all_payload_checksums_identical") is not True or report.get("diffs") != []:
            raise SignatureMigrationError("Probe comparison did not pass")
        if report.get("field_contract") != list(REQUIRED_PROBE_FIELDS):
            raise SignatureMigrationError("Probe report field contract is incomplete")
        canary_root = Path(_required_text(probe, "canary_output_root")).resolve()
        comparison = _compare_probe_ledgers(
            job.output_root,
            canary_root,
            sample_id=sample_id,
            expected_prompts=tuple(probe.get("prompt_ids", ())),
            expected_conditions=tuple(probe.get("conditions", ())),
        )
        if comparison["rows"] != 24:
            raise SignatureMigrationError("Probe must cover one complete P8 x M1/M2/M12 sample")
        if report.get("canonical_rows") != 24 or report.get("probe_rows") != 24:
            raise SignatureMigrationError("Probe report does not contain 24 rows per side")
        results.append(
            {
                "sample_id": sample_id,
                "comparison_report": str(report_path),
                "contract_sha256": comparison["contract_sha256"],
                "rows": comparison["rows"],
                "coverage": coverage,
            }
        )
    return results


def _compare_probe_ledgers(
    canonical_root: Path,
    canary_root: Path,
    *,
    sample_id: str,
    expected_prompts: tuple[Any, ...],
    expected_conditions: tuple[Any, ...],
) -> dict[str, Any]:
    prompts = tuple(str(value) for value in expected_prompts)
    conditions = tuple(str(value) for value in expected_conditions)
    if len(prompts) != 8 or len(set(prompts)) != 8:
        raise SignatureMigrationError("Probe must list exactly eight unique prompts")
    if conditions != ("M1", "M2", "M12"):
        raise SignatureMigrationError("Probe conditions must be exactly M1, M2, M12")
    canonical = _sample_entries(canonical_root / "batch_state.sqlite3", sample_id)
    canary = _sample_entries(canary_root / "batch_state.sqlite3", sample_id)
    expected_keys = {(prompt, condition) for prompt in prompts for condition in conditions}
    if set(canonical) != expected_keys or set(canary) != expected_keys:
        raise SignatureMigrationError("Probe task set does not equal P8 x M1/M2/M12")
    records = []
    for key in sorted(expected_keys):
        left = canonical[key]
        right = canary[key]
        for field in REQUIRED_PROBE_FIELDS:
            if left.get(field) != right.get(field):
                raise SignatureMigrationError(f"Probe payload mismatch: {key}:{field}")
        records.append({field: left.get(field) for field in REQUIRED_PROBE_FIELDS})
    return {"rows": len(records), "contract_sha256": _fingerprint(records)}


def _sample_entries(path: Path, sample_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT entry_json FROM tasks WHERE sample_id=? AND status='completed'",
            (sample_id,),
        ).fetchall()
    finally:
        connection.close()
    result = {}
    for (value,) in rows:
        entry = json.loads(str(value))
        key = (str(entry["prompt_id"]), str(entry["condition"]))
        if key in result:
            raise SignatureMigrationError(f"Duplicate probe task key: {key}")
        result[key] = entry
    return result


def _verify_inactive(job: Any, manifest: dict[str, Any]) -> None:
    guard = manifest.get("writer_guard")
    if not isinstance(guard, dict):
        raise SignatureMigrationError("writer_guard is required")
    markers = guard.get("process_markers")
    if not isinstance(markers, list) or str(job.output_root) not in markers:
        raise SignatureMigrationError("writer_guard must include the exact job output_root")
    if any(not isinstance(marker, str) or not marker for marker in markers):
        raise SignatureMigrationError("writer_guard process markers are invalid")
    processes = _process_cmdlines()
    active_writers = [
        {"pid": pid, "command": command}
        for pid, command in processes.items()
        if any(marker in command for marker in markers)
    ]
    if active_writers:
        raise SignatureMigrationError(f"Active cache writer candidates: {active_writers}")
    gpu_markers = {job.model.model_key}
    gpu_processes = [
        {"pid": pid, "command": processes.get(pid, "")}
        for pid in _gpu_process_ids()
        if any(marker in processes.get(pid, "") for marker in gpu_markers)
    ]
    if gpu_processes:
        raise SignatureMigrationError(f"Active matching GPU processes: {gpu_processes}")


def _verify_complete_ledger(job: Any) -> dict[str, Any]:
    ledger = _ledger_status(job.output_root, job.domain.expected_tasks)
    if ledger.get("status") != "complete":
        raise SignatureMigrationError(f"Ledger is not complete: {ledger}")
    counts = ledger.get("counts", {})
    if any(int(counts.get(key, 0)) for key in ("pending", "running", "failed")):
        raise SignatureMigrationError(f"Ledger has non-completed tasks: {counts}")
    return ledger


def _completed_ledger_rows(job: Any) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{job.output_root / 'batch_state.sqlite3'}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT task_id, attempts, entry_json FROM tasks WHERE status='completed' "
            "ORDER BY task_id"
        ).fetchall()
    finally:
        connection.close()
    result = []
    for task_id, attempts, entry_json in rows:
        entry = json.loads(str(entry_json))
        result.append(
            {"task_id": str(task_id), "attempts": int(attempts), "entry": entry}
        )
    return result


def _entry_sidecar(job: Any, entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    cache_root = Path(_required_text(entry, "cache_root")).resolve()
    if not cache_root.is_relative_to(job.output_root.resolve()):
        raise SignatureMigrationError("Ledger entry cache_root escapes the job output root")
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        raise SignatureMigrationError("Ledger entry metadata is missing")
    path = (cache_root / _required_text(metadata, "sidecar_path")).resolve()
    if not path.is_relative_to(job.output_root.resolve()) or not path.is_file():
        raise SignatureMigrationError(f"Ledger sidecar is missing or outside job root: {path}")
    sidecar = _read_json(path)
    sidecar_entry = sidecar.get("entry")
    if not isinstance(sidecar_entry, dict):
        raise SignatureMigrationError(f"Sidecar entry is missing: {path}")
    for field in ("sample_id", "model_key", "protocol", "prompt_id", "condition", "checksum"):
        if sidecar_entry.get(field) != entry.get(field):
            raise SignatureMigrationError(f"Ledger/sidecar mismatch: {path}:{field}")
    return path, sidecar


def _verify_domain_guards(
    manifest: dict[str, Any],
    *,
    job: Any,
    old_signature: dict[str, Any],
    current_signature: dict[str, Any],
    ledger: dict[str, Any],
) -> list[dict[str, Any]]:
    declared = manifest.get("domain_guards", [])
    if not isinstance(declared, list) or any(not isinstance(item, dict) for item in declared):
        raise SignatureMigrationError("domain_guards must be a list of objects")
    required_kind = None
    if old_signature.get("schema") == LEGACY_V2_SCHEMA:
        path = str(current_signature.get("wrapper_path", ""))
        if path.endswith("llava_onevision.py"):
            required_kind = "llava_onevision_token_limit"
        elif path.endswith("gemma4.py"):
            required_kind = "gemma4_same_media"
    if required_kind is None:
        if declared:
            raise SignatureMigrationError("Domain guards were declared for a job that needs none")
        return []
    if len(declared) != 1 or declared[0].get("kind") != required_kind:
        raise SignatureMigrationError(f"Exactly one {required_kind} domain guard is required")
    guard = declared[0]
    rows = _completed_ledger_rows(job)
    expected_rows = int(ledger["counts"]["completed"])
    if len(rows) != expected_rows or guard.get("expected_completed_rows") != expected_rows:
        raise SignatureMigrationError("Domain guard completed-row count mismatch")
    if required_kind == "llava_onevision_token_limit":
        limit = _llava_onevision_effective_context_limit(current_signature)
        token_counts = [row["entry"].get("token_count") for row in rows]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in token_counts):
            raise SignatureMigrationError("LLaVA-OneVision ledger token counts are invalid")
        maximum = max(token_counts)
        result = {
            "kind": required_kind,
            "expected_completed_rows": expected_rows,
            "max_position_embeddings": limit,
            "maximum_completed_token_count": maximum,
        }
        if maximum >= limit:
            raise SignatureMigrationError("LLaVA-OneVision cache reaches changed limit boundary")
        if guard != result:
            raise SignatureMigrationError("LLaVA-OneVision domain guard evidence mismatch")
        return [result]

    checked = []
    for row in rows:
        entry = row["entry"]
        if entry.get("condition") != "M12":
            continue
        _, sidecar = _entry_sidecar(job, entry)
        request = sidecar.get("request")
        if not isinstance(request, dict) or request.get("use_audio_in_video") is not True:
            raise SignatureMigrationError("Gemma-4 M12 request does not use embedded audio")
        media_paths = request.get("media_paths")
        if not isinstance(media_paths, dict):
            raise SignatureMigrationError("Gemma-4 M12 request lacks media_paths")
        vision = str(media_paths.get("vision", ""))
        audio = str(media_paths.get("audio", ""))
        if not vision or vision != audio:
            raise SignatureMigrationError("Gemma-4 M12 vision/audio assets differ")
        checked.append(
            {
                "task_id": row["task_id"],
                "sample_id": str(entry.get("sample_id")),
                "prompt_id": str(entry.get("prompt_id")),
                "media_path": vision,
            }
        )
    result = {
        "kind": required_kind,
        "expected_completed_rows": expected_rows,
        "m12_rows": len(checked),
        "m12_contract_sha256": _fingerprint(checked),
    }
    if not checked or guard != result:
        raise SignatureMigrationError("Gemma-4 same-media domain guard evidence mismatch")
    return [result]


def _llava_onevision_effective_context_limit(
    current_signature: dict[str, Any],
) -> int:
    """Resolve the exact context limit through the wrapper's official config class."""
    python = _required_text(current_signature, "sys_executable")
    model_path = _required_text(current_signature, "model_path")
    runtime_library_path = _required_text(current_signature, "runtime_library_path")
    code = (
        "from transformers import LlavaOnevisionConfig; import sys; "
        "config=LlavaOnevisionConfig.from_pretrained(sys.argv[1], local_files_only=True); "
        "print(int(config.text_config.max_position_embeddings))"
    )
    env = dict(os.environ)
    inherited_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = runtime_library_path + (
        f":{inherited_library_path}" if inherited_library_path else ""
    )
    if current_signature.get("python_no_user_site") is True:
        env.update(
            {
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    else:
        env.pop("PYTHONNOUSERSITE", None)
    completed = subprocess.run(
        [python, "-c", code, model_path],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    if completed.returncode != 0:
        raise SignatureMigrationError(
            "LLaVA-OneVision effective config inspection failed: "
            + completed.stderr.strip()
        )
    try:
        limit = int(completed.stdout.strip())
    except ValueError as exc:
        raise SignatureMigrationError(
            "LLaVA-OneVision effective context limit is not an integer"
        ) from exc
    if limit <= 0:
        raise SignatureMigrationError("LLaVA-OneVision effective context limit is invalid")
    return limit


def _verify_ledger_provenance(
    manifest: dict[str, Any],
    *,
    job: Any,
    current_signature: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any] | None:
    declared = manifest.get("ledger_provenance")
    requires_mixed = (
        job.job_id == "target:phi4_multimodal"
        and current_signature.get("wrapper_path") == "src/mprisk/models/phi4_mm.py"
    )
    if not requires_mixed:
        if declared is not None:
            raise SignatureMigrationError("ledger_provenance is forbidden for this job")
        return None
    if not isinstance(declared, dict) or declared.get("kind") != "phi4_allocator_mixed_v1":
        raise SignatureMigrationError("Explicit Phi-4 mixed ledger provenance is required")
    expected_allocator = {
        "backend": "native",
        "environment": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    records = []
    counts = {"attempt_1_without_allocator": 0, "attempt_2_with_allocator": 0}
    rows = _completed_ledger_rows(job)
    if len(rows) != int(ledger["counts"]["completed"]):
        raise SignatureMigrationError("Phi-4 mixed provenance row count mismatch")
    for row in rows:
        path, sidecar = _entry_sidecar(job, row["entry"])
        provenance = sidecar.get("provenance")
        if not isinstance(provenance, dict):
            raise SignatureMigrationError(f"Phi-4 sidecar provenance is missing: {path}")
        has_allocator = "cuda_allocator" in provenance
        if row["attempts"] == 1 and not has_allocator:
            bucket = "attempt_1_without_allocator"
        elif row["attempts"] == 2 and provenance.get("cuda_allocator") == expected_allocator:
            bucket = "attempt_2_with_allocator"
        else:
            raise SignatureMigrationError(
                f"Unexpected Phi-4 attempt/allocator provenance combination: {path}"
            )
        counts[bucket] += 1
        entry = row["entry"]
        records.append(
            {
                "task_id": row["task_id"],
                "attempts": row["attempts"],
                "bucket": bucket,
                "checksum": str(entry.get("checksum")),
            }
        )
    result = {
        "kind": "phi4_allocator_mixed_v1",
        "completed_rows": len(rows),
        "counts": counts,
        "expected_allocator": expected_allocator,
        "records_sha256": _fingerprint(records),
    }
    if declared != result:
        raise SignatureMigrationError("Phi-4 mixed ledger provenance evidence mismatch")
    return result


def _process_cmdlines() -> dict[int, str]:
    result = {}
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        pid = int(path.parent.name)
        if pid == os.getpid():
            continue
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        result[pid] = command
    return result


def _gpu_process_ids() -> set[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SignatureMigrationError(f"Could not verify GPU processes: {completed.stderr.strip()}")
    return {int(line.strip()) for line in completed.stdout.splitlines() if line.strip()}


def _semantic_repository_files(signature: dict[str, Any]) -> dict[str, str]:
    value = signature.get("extractor_semantic_files")
    if not isinstance(value, dict) or not isinstance(value.get("repository"), dict):
        raise SignatureMigrationError("Signature lacks extractor semantic repository files")
    files = value["repository"]
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in files.items()):
        raise SignatureMigrationError("Signature semantic repository files are invalid")
    return dict(files)


def _symbol_ast_sha256(source: bytes, symbol: str) -> str:
    node: ast.AST = ast.parse(source.decode())
    for name in symbol.split("."):
        body = getattr(node, "body", ())
        matches = [item for item in body if getattr(item, "name", None) == name]
        if len(matches) != 1:
            raise SignatureMigrationError(f"Protected symbol is missing or ambiguous: {symbol}")
        node = matches[0]
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _classification_ast_sha256(
    source: bytes, *, path: str, classification: str
) -> str:
    tree = ast.parse(source.decode())
    if classification == "generation_only" and path.endswith("base_wrapper.py"):
        removed_assignments = {
            "OPTIONAL_GENERATION_KWARGS",
            "REQUIRED_GENERATION_KWARGS",
            "SUPPORTED_GENERATION_KWARGS",
        }
        removed_definitions = {
            "GenerationRequest",
            "generate_with_standard_kwargs",
            "validate_generation_kwargs",
        }
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in removed_assignments
                    for target in node.targets
                )
            )
            and not (
                isinstance(node, ast.FunctionDef | ast.ClassDef)
                and node.name in removed_definitions
            )
        ]
    elif classification == "generation_only" and path.endswith("hf_visual_prefill.py"):
        removed_imports = {
            "GenerationRequest",
            "GenerationResult",
            "generate_with_standard_kwargs",
        }
        removed_functions = {"_new_generation_tokens", "_tokenizer_eos_token_ids"}
        body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "mprisk.models.base_wrapper":
                node.names = [alias for alias in node.names if alias.name not in removed_imports]
            if isinstance(node, ast.FunctionDef) and node.name in removed_functions:
                continue
            if isinstance(node, ast.ClassDef) and node.name == "HfVisualPrefillWrapper":
                node.body = [
                    item
                    for item in node.body
                    if not (
                        isinstance(item, ast.FunctionDef)
                        and item.name == "generate_conditioned"
                    )
                ]
            body.append(node)
        tree.body = body
    elif classification == "python_timezone_alias_only" and path.endswith(
        "prefill_writer.py"
    ):
        class NormalizeUtcAlias(ast.NodeTransformer):
            def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
                self.generic_visit(node)
                if node.module == "datetime":
                    node.names = [
                        ast.alias(name="UTC")
                        if alias.name in {"UTC", "timezone"}
                        else alias
                        for alias in node.names
                    ]
                    node.names.sort(key=lambda alias: (alias.name, alias.asname or ""))
                return node

            def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                self.generic_visit(node)
                if (
                    node.attr == "utc"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "timezone"
                ):
                    return ast.copy_location(ast.Name(id="UTC", ctx=ast.Load()), node)
                return node

        tree = NormalizeUtcAlias().visit(tree)
    elif classification == "llava_onevision_class_move_only" and path.endswith(
        "llava.py"
    ):
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.ClassDef) and node.name == "LlavaOneVisionWrapper")
        ]
    elif classification == "llava_onevision_limit_boundary_only" and path.endswith(
        "llava_onevision.py"
    ):
        class NormalizeLimitBoundary(ast.NodeTransformer):
            def visit_If(self, node: ast.If) -> ast.AST:
                self.generic_visit(node)
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "token_count"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], (ast.Gt, ast.GtE))
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Name)
                    and test.comparators[0].id == "max_position_embeddings"
                ):
                    test.ops = [ast.Gt()]
                    node.body = [
                        ast.Raise(
                            exc=ast.Call(
                                func=ast.Name(id="ValueError", ctx=ast.Load()),
                                args=[ast.Constant(value="TOKEN_LIMIT")],
                                keywords=[],
                            ),
                            cause=None,
                        )
                    ]
                return node

        tree = NormalizeLimitBoundary().visit(tree)
    elif classification == "gemma4_same_media_and_provenance_only" and path.endswith(
        "gemma4.py"
    ):
        class NormalizeGemma4(ast.NodeTransformer):
            def _strip_docstring(self, node: ast.AST) -> ast.AST:
                body = getattr(node, "body", None)
                if (
                    isinstance(body, list)
                    and body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:]
                return node

            def visit_Module(self, node: ast.Module) -> ast.AST:
                self.generic_visit(node)
                self._strip_docstring(node)
                prefix = []
                remainder = list(node.body)
                while remainder and isinstance(remainder[0], (ast.Import, ast.ImportFrom)):
                    prefix.append(remainder.pop(0))
                node.body = sorted(
                    prefix, key=lambda item: ast.dump(item, include_attributes=False)
                ) + remainder
                return node

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                self.generic_visit(node)
                return self._strip_docstring(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                self.generic_visit(node)
                self._strip_docstring(node)
                if node.name == "__init__":
                    node.body = [
                        item
                        for item in node.body
                        if not (
                            isinstance(item, ast.Assign)
                            and any(
                                isinstance(target, ast.Attribute)
                                and target.attr in {"_weight_file", "_weight_file_sha256"}
                                for target in item.targets
                            )
                        )
                    ]
                if node.name == "_validate_request":
                    removable = {
                        "request.condition == 'M12' and (not request.use_audio_in_video)",
                        "request.condition == 'M12' and request.use_audio_in_video",
                    }
                    node.body = [
                        item
                        for item in node.body
                        if not (
                            isinstance(item, ast.If)
                            and ast.unparse(item.test) in removable
                        )
                    ]
                if node.name == "build_va_request":
                    node.body = [
                        item
                        for item in node.body
                        if not (
                            isinstance(item, ast.Assign)
                            and any(
                                isinstance(target, ast.Name)
                                and target.id == "use_embedded_audio"
                                for target in item.targets
                            )
                        )
                    ]
                    for item in ast.walk(node):
                        if isinstance(item, ast.keyword) and item.arg == "use_audio_in_video":
                            item.value = ast.Compare(
                                left=ast.Name(id="condition", ctx=ast.Load()),
                                ops=[ast.Eq()],
                                comparators=[ast.Constant(value="M12")],
                            )
                return node

            def visit_Dict(self, node: ast.Dict) -> ast.AST:
                self.generic_visit(node)
                pairs = [
                    (key, value)
                    for key, value in zip(node.keys, node.values, strict=True)
                    if not (
                        isinstance(key, ast.Constant)
                        and key.value in {"weight_file_path", "weight_file_sha256"}
                    )
                ]
                node.keys = [key for key, _ in pairs]
                node.values = [value for _, value in pairs]
                return node

        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.FunctionDef) and node.name == "_single_weight_file")
        ]
        tree = NormalizeGemma4().visit(tree)
    elif classification in {
        "allocator_provenance_only",
        "generation_allocator_provenance_only",
    } and path.endswith("phi4_mm.py"):
        body = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                node.names = [alias for alias in node.names if alias.name != "os"]
                if not node.names:
                    continue
            if isinstance(node, ast.FunctionDef) and node.name == "_cuda_allocator_provenance":
                continue
            if (
                classification == "generation_allocator_provenance_only"
                and isinstance(node, ast.FunctionDef)
                and node.name == "_tokenizer_eos_token_ids"
            ):
                continue
            if (
                classification == "generation_allocator_provenance_only"
                and isinstance(node, ast.ImportFrom)
                and node.module == "mprisk.models.base_wrapper"
            ):
                node.names = [
                    alias
                    for alias in node.names
                    if alias.name
                    not in {
                        "GenerationRequest",
                        "GenerationResult",
                        "generate_with_standard_kwargs",
                    }
                ]
            if (
                classification == "generation_allocator_provenance_only"
                and isinstance(node, ast.ClassDef)
                and node.name == "Phi4MmWrapper"
            ):
                node.body = [
                    item
                    for item in node.body
                    if not (
                        isinstance(item, ast.FunctionDef)
                        and item.name == "generate_conditioned"
                    )
                ]
            body.append(node)
        tree.body = body

        class RemoveAllocatorProvenance(ast.NodeTransformer):
            def visit_Dict(self, node: ast.Dict) -> ast.AST:
                self.generic_visit(node)
                pairs = [
                    (key, value)
                    for key, value in zip(node.keys, node.values, strict=True)
                    if not (
                        isinstance(key, ast.Constant)
                        and key.value == "cuda_allocator"
                    )
                ]
                node.keys = [key for key, _ in pairs]
                node.values = [value for _, value in pairs]
                return node

        tree = RemoveAllocatorProvenance().visit(tree)
    elif classification == "generation_only" and path.endswith("qwen_omni.py"):
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "mprisk.models.base_wrapper":
                node.names = [
                    alias
                    for alias in node.names
                    if alias.name != "generate_with_standard_kwargs"
                ]
            if isinstance(node, ast.ClassDef) and node.name == "QwenOmniWrapper":
                node.body = [
                    item
                    for item in node.body
                    if not (
                        isinstance(item, ast.FunctionDef)
                        and item.name == "generate_conditioned"
                    )
                ]
    else:
        raise SignatureMigrationError(
            f"No AST normalizer for semantic classification: {classification}:{path}"
        )
    ast.fix_missing_locations(tree)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _verify_file_reference(value: Any, *, label: str) -> Path:
    if not isinstance(value, dict):
        raise SignatureMigrationError(f"{label} file reference is required")
    path = Path(_required_text(value, "path")).expanduser().resolve()
    if not path.is_file():
        raise SignatureMigrationError(f"{label} does not exist: {path}")
    if value.get("sha256") != _sha256_file(path):
        raise SignatureMigrationError(f"{label} SHA-256 mismatch")
    return path


def _required_text(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise SignatureMigrationError(f"Required non-empty field is missing: {key}")
    return item


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SignatureMigrationError(f"Expected JSON object: {path}")
    return payload


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    ).stdout


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_bytes(path, serialized.encode())
