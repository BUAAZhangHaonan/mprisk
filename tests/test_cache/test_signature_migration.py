from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import mprisk.cache.signature_migration as migration


def _job(root: Path, *, expected_tasks: int = 24) -> SimpleNamespace:
    return SimpleNamespace(
        job_id="target:model",
        output_root=root,
        asset_signature_evidence=root / "ASSET_SIGNATURE.json",
        domain=SimpleNamespace(expected_tasks=expected_tasks),
        model=SimpleNamespace(model_key="model"),
    )


def _ledger(root: Path, sample_id: str, *, changed: bool = False) -> None:
    root.mkdir(parents=True)
    connection = sqlite3.connect(root / "batch_state.sqlite3")
    connection.execute(
        "CREATE TABLE tasks(sample_id TEXT, status TEXT, entry_json TEXT)"
    )
    prompts = [f"p{i}" for i in range(8)]
    conditions = ["M1", "M2", "M12"]
    for prompt in prompts:
        for condition in conditions:
            checksum = f"sha-{prompt}-{condition}"
            if changed and prompt == "p0" and condition == "M1":
                checksum = "different"
            entry = {
                "checksum": checksum,
                "layer_count": 2,
                "hidden_dim": 3,
                "token_count": 4,
                "t0_token_index": 3,
                "model_key": "model",
                "protocol": "vt",
                "prompt_id": prompt,
                "condition": condition,
                "sample_id": sample_id,
            }
            connection.execute(
                "INSERT INTO tasks VALUES(?,?,?)",
                (sample_id, "completed", json.dumps(entry)),
            )
    connection.commit()
    connection.close()


def test_file_reference_rejects_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(migration.SignatureMigrationError, match="SHA-256 mismatch"):
        migration._verify_file_reference(
            {"path": str(path), "sha256": "0" * 64}, label="evidence"
        )


def test_complete_ledger_rejects_incomplete_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        migration,
        "_ledger_status",
        lambda *_: {
            "status": "incomplete",
            "counts": {"completed": 23, "running": 1},
        },
    )
    with pytest.raises(migration.SignatureMigrationError, match="not complete"):
        migration._verify_complete_ledger(_job(tmp_path))


def test_inactive_guard_rejects_active_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    monkeypatch.setattr(
        migration,
        "_process_cmdlines",
        lambda: {123: f"python extract.py --output-root {tmp_path}"},
    )
    monkeypatch.setattr(migration, "_gpu_process_ids", lambda: set())
    with pytest.raises(migration.SignatureMigrationError, match="Active cache writer"):
        migration._verify_inactive(
            job,
            {"writer_guard": {"process_markers": [str(tmp_path)]}},
        )


def test_inactive_guard_rejects_matching_gpu_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    monkeypatch.setattr(
        migration,
        "_process_cmdlines",
        lambda: {123: "python extract.py --model-key model"},
    )
    monkeypatch.setattr(migration, "_gpu_process_ids", lambda: {123})
    with pytest.raises(migration.SignatureMigrationError, match="matching GPU"):
        migration._verify_inactive(
            job,
            {"writer_guard": {"process_markers": [str(tmp_path)]}},
        )


def test_probe_comparison_requires_field_exact_payloads(tmp_path: Path) -> None:
    sample_id = "sample"
    canonical = tmp_path / "canonical"
    canary = tmp_path / "canary"
    _ledger(canonical, sample_id)
    _ledger(canary, sample_id, changed=True)
    with pytest.raises(migration.SignatureMigrationError, match="payload mismatch"):
        migration._compare_probe_ledgers(
            canonical,
            canary,
            sample_id=sample_id,
            expected_prompts=tuple(f"p{i}" for i in range(8)),
            expected_conditions=("M1", "M2", "M12"),
        )


def test_probe_comparison_accepts_complete_p8_by_three(tmp_path: Path) -> None:
    sample_id = "sample"
    canonical = tmp_path / "canonical"
    canary = tmp_path / "canary"
    _ledger(canonical, sample_id)
    _ledger(canary, sample_id)
    result = migration._compare_probe_ledgers(
        canonical,
        canary,
        sample_id=sample_id,
        expected_prompts=tuple(f"p{i}" for i in range(8)),
        expected_conditions=("M1", "M2", "M12"),
    )
    assert result["rows"] == 24
    assert len(result["contract_sha256"]) == 64


def test_code_evidence_rejects_unknown_semantic_diff(tmp_path: Path) -> None:
    old = {
        "extractor_semantic_files": {
            "repository": {"src/mprisk/models/unknown.py": "a" * 64}
        }
    }
    current = {
        "extractor_semantic_files": {
            "repository": {"src/mprisk/models/unknown.py": "b" * 64}
        }
    }
    evidence = {
        "schema": migration.CODE_EVIDENCE_SCHEMA,
        "base_git_sha": "a" * 40,
        "head_git_sha": "b" * 40,
        "changed_files": [
            {
                "path": "src/mprisk/models/unknown.py",
                "classification": "generation_only",
            }
        ],
    }
    with pytest.raises(migration.SignatureMigrationError, match="Unknown semantic diff"):
        migration._verify_code_evidence(
            tmp_path,
            old_signature=old,
            current_signature=current,
            evidence=evidence,
            current_head="b" * 40,
        )


def test_apply_rolls_back_signature_and_pointer_on_payload_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    job = _job(root)
    old = {"schema": "old"}
    current = {"schema": "current"}
    old_bytes = (json.dumps(old) + "\n").encode()
    job.asset_signature_evidence.write_bytes(old_bytes)
    receipt = root / "receipts" / "completion" / "old.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"old":true}\n', encoding="utf-8")
    pointer = {
        "schema": "mprisk_cache_completion_pointer_v1",
        "content_sha256": "a" * 64,
        "receipt_path": "receipts/completion/old.json",
    }
    pointer_bytes = (json.dumps(pointer) + "\n").encode()
    (root / "COMPLETION_RECEIPT.json").write_bytes(pointer_bytes)
    monkeypatch.setattr(
        migration,
        "audit_completed_cache",
        lambda *args, **kwargs: {
            "passed": True,
            "payload_tree_sha256": "after",
        },
    )
    with pytest.raises(migration.SignatureMigrationError, match="payload tree changed"):
        migration._apply_migration(
            job=job,
            manifest={},
            manifest_bytes=b"{}\n",
            old_signature_bytes=old_bytes,
            current_signature=current,
            expected_batch_signature={},
            before={"payload_tree_sha256": "before"},
            base_report={
                "old_signature_sha256": migration._fingerprint(old),
                "current_expected_signature_sha256": migration._fingerprint(current),
            },
        )
    assert job.asset_signature_evidence.read_bytes() == old_bytes
    assert (root / "COMPLETION_RECEIPT.json").read_bytes() == pointer_bytes
    record = next((root / "receipts" / "signature_migrations").glob("*/MIGRATION_RECORD.json"))
    assert json.loads(record.read_text(encoding="utf-8"))["status"] == "rolled_back"


def test_apply_preserves_old_evidence_and_writes_new_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    job = _job(root)
    old = {"schema": "old"}
    current = {"schema": "current"}
    old_bytes = (json.dumps(old) + "\n").encode()
    job.asset_signature_evidence.write_bytes(old_bytes)

    def successful_audit(*args: object, **kwargs: object) -> dict[str, object]:
        pointer = {
            "schema": "mprisk_cache_completion_pointer_v1",
            "content_sha256": "b" * 64,
            "receipt_path": "receipts/completion/new.json",
        }
        (root / "receipts" / "completion").mkdir(parents=True, exist_ok=True)
        (root / "receipts" / "completion" / "new.json").write_text(
            '{"new":true}\n', encoding="utf-8"
        )
        (root / "COMPLETION_RECEIPT.json").write_text(
            json.dumps(pointer) + "\n", encoding="utf-8"
        )
        return {
            "passed": True,
            "payload_tree_sha256": "same",
            "content_sha256": "b" * 64,
        }

    monkeypatch.setattr(migration, "audit_completed_cache", successful_audit)
    result = migration._apply_migration(
        job=job,
        manifest={},
        manifest_bytes=b"{}\n",
        old_signature_bytes=old_bytes,
        current_signature=current,
        expected_batch_signature={},
        before={"payload_tree_sha256": "same"},
        base_report={
            "old_signature_sha256": migration._fingerprint(old),
            "current_expected_signature_sha256": migration._fingerprint(current),
        },
    )
    assert result["status"] == "complete"
    assert json.loads(job.asset_signature_evidence.read_text(encoding="utf-8")) == current
    record_root = Path(result["record_root"])
    assert (record_root / "PREVIOUS_ASSET_SIGNATURE.json").read_bytes() == old_bytes
    assert json.loads((record_root / "MIGRATION_RECORD.json").read_text())["status"] == "complete"


def test_protected_symbol_ast_hash_ignores_unrelated_addition() -> None:
    before = b"class A:\n    def protected(self):\n        return 1\n"
    after = before + b"\ndef unrelated():\n    return 2\n"
    assert migration._symbol_ast_sha256(before, "A.protected") == migration._symbol_ast_sha256(
        after, "A.protected"
    )


def test_generation_classifier_removes_only_declared_generation_path() -> None:
    before = b"class PrefillRequest:\n    pass\nclass GenerationRequest:\n    pass\n"
    after = (
        b"REQUIRED_GENERATION_KWARGS = frozenset()\n"
        b"class PrefillRequest:\n    pass\n"
        b"class GenerationRequest:\n    changed = True\n"
        b"def validate_generation_kwargs(value):\n    return value\n"
        b"def generate_with_standard_kwargs():\n    return None\n"
    )
    assert migration._classification_ast_sha256(
        before,
        path="src/mprisk/models/base_wrapper.py",
        classification="generation_only",
    ) == migration._classification_ast_sha256(
        after,
        path="src/mprisk/models/base_wrapper.py",
        classification="generation_only",
    )
    changed_prefill = after.replace(
        b"class PrefillRequest:\n    pass",
        b"class PrefillRequest:\n    value = 1",
    )
    assert migration._classification_ast_sha256(
        before,
        path="src/mprisk/models/base_wrapper.py",
        classification="generation_only",
    ) != migration._classification_ast_sha256(
        changed_prefill,
        path="src/mprisk/models/base_wrapper.py",
        classification="generation_only",
    )


def test_allocator_classifier_removes_only_allocator_provenance() -> None:
    before = b"def extract():\n    return {'value': 1}\n"
    after = (
        b"import os\n"
        b"def extract():\n"
        b"    return {'value': 1, 'cuda_allocator': _cuda_allocator_provenance()}\n"
        b"def _cuda_allocator_provenance():\n    return {'environment': dict(os.environ)}\n"
    )
    kwargs = {
        "path": "src/mprisk/models/phi4_mm.py",
        "classification": "allocator_provenance_only",
    }
    assert migration._classification_ast_sha256(
        before, **kwargs
    ) == migration._classification_ast_sha256(after, **kwargs)
    changed_math = after.replace(b"'value': 1", b"'value': 2")
    assert migration._classification_ast_sha256(
        before, **kwargs
    ) != migration._classification_ast_sha256(changed_math, **kwargs)


@pytest.mark.parametrize(
    ("path", "classification", "before", "after", "changed"),
    [
        (
            "src/mprisk/cache/prefill_writer.py",
            "python_timezone_alias_only",
            b"from datetime import UTC, datetime\ndef f():\n    return datetime.now(UTC)\n",
            b"from datetime import datetime, timezone\ndef f():\n    return datetime.now(timezone.utc)\n",
            b"from datetime import datetime, timezone\ndef f():\n    return datetime.now()\n",
        ),
        (
            "src/mprisk/models/llava.py",
            "llava_onevision_class_move_only",
            b"class Keep:\n    x = 1\nclass LlavaOneVisionWrapper:\n    x = 2\n",
            b"class Keep:\n    x = 1\n",
            b"class Keep:\n    x = 3\n",
        ),
        (
            "src/mprisk/models/qwen_omni.py",
            "generation_only",
            b"from mprisk.models.base_wrapper import PrefillRequest\nclass QwenOmniWrapper:\n    def extract_prefill(self):\n        return 1\n    def generate_conditioned(self):\n        return 1\n",
            b"from mprisk.models.base_wrapper import PrefillRequest, generate_with_standard_kwargs\nclass QwenOmniWrapper:\n    def extract_prefill(self):\n        return 1\n    def generate_conditioned(self):\n        return 2\n",
            b"from mprisk.models.base_wrapper import PrefillRequest, generate_with_standard_kwargs\nclass QwenOmniWrapper:\n    def extract_prefill(self):\n        return 2\n    def generate_conditioned(self):\n        return 2\n",
        ),
    ],
)
def test_new_classifiers_are_exact_and_detect_nonclassified_changes(
    path: str,
    classification: str,
    before: bytes,
    after: bytes,
    changed: bytes,
) -> None:
    kwargs = {"path": path, "classification": classification}
    expected = migration._classification_ast_sha256(before, **kwargs)
    assert migration._classification_ast_sha256(after, **kwargs) == expected
    assert migration._classification_ast_sha256(changed, **kwargs) != expected


def test_llava_limit_domain_guard_rejects_changed_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    monkeypatch.setattr(
        migration,
        "_llava_onevision_effective_context_limit",
        lambda _signature: 8,
    )
    monkeypatch.setattr(
        migration,
        "_completed_ledger_rows",
        lambda _job: [{"task_id": "t", "attempts": 1, "entry": {"token_count": 8}}],
    )
    with pytest.raises(migration.SignatureMigrationError, match="reaches changed limit"):
        migration._verify_domain_guards(
            {
                "domain_guards": [
                    {
                        "kind": "llava_onevision_token_limit",
                        "expected_completed_rows": 1,
                        "max_position_embeddings": 8,
                        "maximum_completed_token_count": 8,
                    }
                ]
            },
            job=_job(tmp_path),
            old_signature={"schema": migration.LEGACY_V2_SCHEMA},
            current_signature={
                "wrapper_path": "src/mprisk/models/llava_onevision.py",
                "model_path": str(model_root),
            },
            ledger={"counts": {"completed": 1}},
        )


def test_phi_mixed_provenance_requires_exact_attempt_buckets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    job.job_id = "target:phi4_multimodal"
    rows = [
        {"task_id": "a", "attempts": 1, "entry": {"checksum": "x"}},
        {"task_id": "b", "attempts": 2, "entry": {"checksum": "y"}},
    ]
    allocator = {
        "backend": "native",
        "environment": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    monkeypatch.setattr(migration, "_completed_ledger_rows", lambda _job: rows)
    monkeypatch.setattr(
        migration,
        "_entry_sidecar",
        lambda _job, entry: (
            tmp_path / "sidecar.json",
            {"provenance": {} if entry["checksum"] == "x" else {"cuda_allocator": allocator}},
        ),
    )
    records = [
        {"task_id": "a", "attempts": 1, "bucket": "attempt_1_without_allocator", "checksum": "x"},
        {"task_id": "b", "attempts": 2, "bucket": "attempt_2_with_allocator", "checksum": "y"},
    ]
    declared = {
        "kind": "phi4_allocator_mixed_v1",
        "completed_rows": 2,
        "counts": {"attempt_1_without_allocator": 1, "attempt_2_with_allocator": 1},
        "expected_allocator": allocator,
        "records_sha256": migration._fingerprint(records),
    }
    assert migration._verify_ledger_provenance(
        {"ledger_provenance": declared},
        job=job,
        current_signature={"wrapper_path": "src/mprisk/models/phi4_mm.py"},
        ledger={"counts": {"completed": 2}},
    ) == declared


def test_llava_effective_context_limit_uses_bound_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "32768\n", "")

    monkeypatch.setattr(migration.subprocess, "run", run)
    signature = {
        "sys_executable": "/env/bin/python",
        "model_path": "/models/llava-onevision",
        "runtime_library_path": "/env/lib",
        "python_no_user_site": True,
    }
    assert migration._llava_onevision_effective_context_limit(signature) == 32768
    assert captured["command"][0] == "/env/bin/python"
    assert "LlavaOnevisionConfig.from_pretrained" in captured["command"][2]
    assert captured["command"][-1] == "/models/llava-onevision"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["LD_LIBRARY_PATH"].split(":", 1)[0] == "/env/lib"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_legacy_v2_provenance_requires_exact_baseline_and_asset_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    asset = model_root / "config.json"
    asset.write_text("{}\n", encoding="utf-8")
    old_source = b"def f():\n    return 1\n"
    old_hash = migration._sha256_bytes(old_source)
    old = {
        "schema": migration.LEGACY_V2_SCHEMA,
        "model_path": str(model_root),
        "wrapper_path": "src/mprisk/models/wrapper.py",
        "wrapper_git_sha": "1" * 40,
        "wrapper_file_sha256": old_hash,
        "stable": "same",
    }
    current = {
        **old,
        "schema": migration.CURRENT_V3_SCHEMA,
        "checkpoint_digest_receipt": str(tmp_path / "receipt.json"),
        "checkpoint_digest_schema": "checkpoint-v1",
        "checkpoint_sha256": "c" * 64,
        "extractor_semantic_files": {
            "repository": {"src/mprisk/models/wrapper.py": "n" * 64},
            "trust_remote_code": {},
        },
        "extractor_semantic_schema": "extractor-v1",
        "extractor_semantic_sha256": "e" * 64,
        "model_asset_fingerprint": "m" * 64,
    }
    signature_path = tmp_path / "ASSET_SIGNATURE.json"
    signature_path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    stat = signature_path.stat()
    asset_stat = asset.stat()
    baseline = "2" * 40
    baseline_files = {"src/mprisk/models/wrapper.py": old_hash}
    asset_records = [
        {
            "path": "config.json",
            "bytes": asset_stat.st_size,
            "sha256": "a" * 64,
            "role": "runtime_asset",
            "mtime_ns": asset_stat.st_mtime_ns,
            "ctime_ns": asset_stat.st_ctime_ns,
        }
    ]
    overlap = ["model_path", "stable", "wrapper_path"]
    legacy = {
        "asset_signature_stat": {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": migration._sha256_file(signature_path),
        },
        "missing_v3_fields": list(migration.V3_PROVENANCE_FIELDS),
        "allowed_changed_fields": ["schema", "wrapper_file_sha256", "wrapper_git_sha"],
        "changed_overlap_fields": ["schema"],
        "equal_overlap_fields": overlap,
        "equal_overlap_sha256": migration._fingerprint({key: old[key] for key in overlap}),
        "baseline_git_sha": baseline,
        "baseline_repository_files_sha256": baseline_files,
        "model_asset_age_guard": {
            "file_count": 1,
            "latest_mtime_ns": asset_stat.st_mtime_ns,
            "latest_ctime_ns": asset_stat.st_ctime_ns,
            "checkpoint_sha256": "c" * 64,
            "model_asset_fingerprint": "m" * 64,
            "files_sha256": migration._fingerprint(asset_records),
        },
    }
    monkeypatch.setattr(migration, "_git", lambda *_args: baseline + "\n")
    monkeypatch.setattr(migration, "_git_bytes", lambda *_args: old_source)
    monkeypatch.setattr(
        migration.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr(
        migration,
        "build_checkpoint_digest",
        lambda _root: {"checkpoint_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        migration,
        "build_model_asset_inventory",
        lambda _root, checkpoint_receipt: {
            "sha256": "m" * 64,
            "inventory": {
                "files": [
                    {
                        "path": "config.json",
                        "bytes": asset_stat.st_size,
                        "sha256": "a" * 64,
                        "role": "runtime_asset",
                    }
                ]
            },
        },
    )
    result = migration._verify_signature_provenance(
        tmp_path,
        old_signature_path=signature_path,
        old_signature=old,
        current_signature=current,
        manifest={"legacy_v2": legacy},
        current_head="3" * 40,
    )
    assert result["mode"] == "historical_baseline_and_asset_age_proof"
