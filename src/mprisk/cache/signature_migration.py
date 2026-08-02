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
from mprisk.cache.integrity import CacheIntegrityError, audit_completed_cache


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
    )
    probe_results = _verify_probes(
        manifest,
        job=job,
        current_head=current_head,
    )
    _verify_inactive(job, manifest)

    ledger = _verify_complete_ledger(job)

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


def _verify_code_evidence(
    repo_root: Path,
    *,
    old_signature: dict[str, Any],
    current_signature: dict[str, Any],
    evidence: dict[str, Any],
    current_head: str,
) -> dict[str, Any]:
    if evidence.get("schema") != CODE_EVIDENCE_SCHEMA:
        raise SignatureMigrationError("Unsupported code diff evidence schema")
    base = _required_text(evidence, "base_git_sha")
    head = _required_text(evidence, "head_git_sha")
    if head != current_head:
        raise SignatureMigrationError("Code evidence head does not match current HEAD")
    old_files = _semantic_repository_files(old_signature)
    current_files = _semantic_repository_files(current_signature)
    if set(old_files) != set(current_files):
        raise SignatureMigrationError(
            "Old signature lacks field-exact v3 semantic-file provenance"
        )
    changed_paths = sorted(
        path for path in old_files if old_files[path] != current_files[path]
    )
    items = evidence.get("changed_files")
    if not isinstance(items, list) or not items:
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
                "protected_symbol_ast_sha256": ast_hashes,
            }
        )
    return {"base_git_sha": base, "head_git_sha": head, "changed_files": verified}


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
