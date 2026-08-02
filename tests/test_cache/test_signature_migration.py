from __future__ import annotations

import json
import sqlite3
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
