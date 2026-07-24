from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

import mprisk.cache.integrity as integrity
from mprisk.cache.integrity import (
    CacheIntegrityError,
    build_checkpoint_digest,
    build_extractor_semantic_digest,
    completion_receipt_status,
    validate_accepted_bundle,
    write_completion_receipt,
)


def _checkpoint(root: Path) -> None:
    root.mkdir()
    (root / "a.safetensors").write_bytes(b"a")
    (root / "b.safetensors").write_bytes(b"b")
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "a.safetensors",
                    "layer.1": "b.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_digest_is_deterministic_resumable_and_covers_shards(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "model"
    _checkpoint(model)
    receipt = tmp_path / "receipt.json"
    first = build_checkpoint_digest(
        model, receipt_path=receipt, write_receipt=True
    )
    calls = []
    original = integrity._sha256_file

    def counted(path: Path) -> str:
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(integrity, "_sha256_file", counted)
    second = build_checkpoint_digest(
        model, receipt_path=receipt, write_receipt=True
    )
    assert calls == []
    assert second["checkpoint_sha256"] == first["checkpoint_sha256"]

    (model / "b.safetensors").write_bytes(b"changed")
    third = build_checkpoint_digest(
        model, receipt_path=receipt, write_receipt=True
    )
    assert calls == ["b.safetensors"]
    assert third["checkpoint_sha256"] != first["checkpoint_sha256"]
    assert [item["role"] for item in third["files"]] == [
        "weight_index",
        "weight_shard",
        "weight_shard",
    ]


def test_checkpoint_digest_keeps_partial_receipt_after_fault(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "model"
    _checkpoint(model)
    receipt = tmp_path / "receipt.json"
    original = integrity._sha256_file
    calls = 0

    def fail_second(path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fault")
        return original(path)

    monkeypatch.setattr(integrity, "_sha256_file", fail_second)
    with pytest.raises(OSError, match="fault"):
        build_checkpoint_digest(model, receipt_path=receipt, write_receipt=True)
    partial = json.loads(receipt.read_text(encoding="utf-8"))
    assert partial["complete"] is False
    assert len(partial["files"]) == 1

    monkeypatch.setattr(integrity, "_sha256_file", original)
    completed = build_checkpoint_digest(
        model, receipt_path=receipt, write_receipt=True
    )
    assert completed["complete"] is True
    assert len(completed["files"]) == 3


def test_extractor_digest_covers_shared_wrapper_and_remote_code(
    tmp_path: Path
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    for relative in (
        *integrity.COMMON_EXTRACTOR_PATHS,
        integrity.WRAPPER_PATHS["qwen_vl"],
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repository, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    model = tmp_path / "model"
    model.mkdir()
    remote = model / "modeling_remote.py"
    remote.write_text("VALUE = 1\n", encoding="utf-8")

    first = build_extractor_semantic_digest(
        repository, family="qwen_vl", model_path=model
    )
    remote.write_text("VALUE = 2\n", encoding="utf-8")
    second = build_extractor_semantic_digest(
        repository, family="qwen_vl", model_path=model
    )
    assert first["sha256"] != second["sha256"]
    assert integrity.WRAPPER_PATHS["qwen_vl"] in first[
        "repository_files_sha256"
    ]
    assert "modeling_remote.py" in first["trust_remote_code_files_sha256"]


def _valid_cache(root: Path) -> tuple[dict[str, object], str]:
    root.mkdir()
    shard = root / "shards/model/vt/M12/sample.safetensors"
    sidecar = root / "shards/model/vt/M12/sample.json"
    shard.parent.mkdir(parents=True)
    save_file(
        {"hidden_states": np.zeros((2, 3), dtype=np.float32)},
        str(shard),
    )
    checksum = integrity._sha256_file(shard)
    request = {
        "sample_id": "sample",
        "model_key": "model",
        "protocol": "vt",
        "condition": "M12",
        "prompt_set_key": "p8",
        "prompt_id": "p0",
        "dataset_key": "source",
        "split": "all",
        "messages": [],
        "media_paths": {},
        "use_audio_in_video": False,
        "runtime_contracts": {},
    }
    identity = {
        key: request[key]
        for key in (
            "sample_id",
            "prompt_id",
            "condition",
            "protocol",
            "model_key",
            "runtime_contracts",
        )
    }
    task_id = integrity._fingerprint(identity)
    entry = {
        "cache_root": str(root),
        "shard_path": shard.relative_to(root).as_posix(),
        "checksum": checksum,
        "sample_id": "sample",
        "model_key": "model",
        "protocol": "vt",
        "condition": "M12",
        "prompt_set_key": "p8",
        "prompt_id": "p0",
        "layer_count": 2,
        "hidden_dim": 3,
        "token_count": 4,
        "t0_token_index": 3,
        "metadata": {
            "tensor_key": "hidden_states",
            "sidecar_path": sidecar.relative_to(root).as_posix(),
        },
    }
    sidecar.write_text(
        json.dumps(
            {
                "schema": integrity.SIDECAR_SCHEMA,
                "entry": entry,
                "request": request,
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    signature = {
        "manifest_sha256": "manifest",
        "prompt_set_sha256": "prompt",
        "prompt_ids": ["p0"],
        "protocol": "vt",
        "conditions": ["M12"],
        "model_key": "model",
        "family": "qwen_vl",
        "dtype": "bfloat16",
        "prefill_strategy": "full_prefill",
        "prefill_strategy_version": "v1",
    }
    connection = sqlite3.connect(root / "batch_state.sqlite3")
    connection.execute(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        """CREATE TABLE tasks(
        task_id TEXT PRIMARY KEY, sample_id TEXT, model_key TEXT, protocol TEXT,
        prompt_set_key TEXT, prompt_id TEXT, condition TEXT, status TEXT,
        checksum TEXT, entry_json TEXT)"""
    )
    connection.execute(
        "INSERT INTO metadata(key,value) VALUES('signature',?)",
        (integrity._canonical_json(signature),),
    )
    connection.execute(
        "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            task_id,
            "sample",
            "model",
            "vt",
            "p8",
            "p0",
            "M12",
            "completed",
            checksum,
            integrity._canonical_json(entry),
        ),
    )
    connection.commit()
    connection.close()
    return signature, task_id


def test_completion_receipt_validates_content_and_reuses_stats(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "cache"
    signature, _ = _valid_cache(root)
    result = write_completion_receipt(
        root, expected_signature=signature, expected_tasks=1
    )
    assert result["passed"] is True
    monkeypatch.setattr(
        integrity,
        "_sha256_file",
        lambda path: (_ for _ in ()).throw(AssertionError(path)),
    )
    status = completion_receipt_status(
        root, expected_signature=signature, expected_tasks=1
    )
    assert status["passed"] is True


@pytest.mark.parametrize("fault", ["shard", "sidecar", "task_id"])
def test_completion_receipt_fault_injection(
    tmp_path: Path, fault: str
) -> None:
    root = tmp_path / "cache"
    signature, task_id = _valid_cache(root)
    if fault == "shard":
        shard = next(root.rglob("*.safetensors"))
        shard.write_bytes(b"corrupt")
        match = "checksum"
    elif fault == "sidecar":
        sidecar = next(root.rglob("sample.json"))
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["request"]["prompt_id"] = "wrong"
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        match = "entry mismatch|request prompt_id"
    else:
        connection = sqlite3.connect(root / "batch_state.sqlite3")
        connection.execute(
            "UPDATE tasks SET task_id=? WHERE task_id=?",
            ("0" * 64, task_id),
        )
        connection.commit()
        connection.close()
        match = "identity"
    with pytest.raises(CacheIntegrityError, match=match):
        write_completion_receipt(
            root, expected_signature=signature, expected_tasks=1
        )


def test_accepted_bundle_requires_exact_identity_or_exact_waiver(
    tmp_path: Path
) -> None:
    signature = {
        "model_key": "model",
        "family": "qwen_vl",
        "protocol": "vt",
        "dtype": "bfloat16",
        "manifest_sha256": "manifest",
        "prompt_set_sha256": "prompt",
        "prompt_ids": ["p0"],
        "conditions": ["M1"],
        "model_path": "/model",
    }
    entries = [
        {
            "sample_id": "s0",
            "prompt_id": "p0",
            "condition": "M1",
            "model_key": "model",
            "protocol": "vt",
        }
    ]
    actual = {
        **signature,
        "prefill_strategy": "full_prefill",
        "prefill_strategy_version": "v1",
        "expected_tasks": 1,
        "task_set_sha256": integrity._fingerprint(
            [["s0", "p0", "M1", "model", "vt"]]
        ),
        "model_asset_fingerprint": "asset",
        "extractor_semantic_fingerprint": "extractor",
    }
    package = {
        "schema": "mprisk_prefill_cache_union_v2",
        "entries": entries,
        "provenance": {
            "expected_signature": signature,
            "expected_signature_sha256": integrity._fingerprint(signature),
            "prefill_strategy": "full_prefill",
            "prefill_strategy_version": "v1",
            "model_asset_fingerprint": "asset",
            "extractor_semantic_fingerprint": "extractor",
        },
    }
    index = tmp_path / "manifest.package.json"
    index.write_text(json.dumps(package), encoding="utf-8")
    validate_accepted_bundle(index, expected_identity=actual)

    expected = {**actual, "extractor_semantic_fingerprint": "new"}
    with pytest.raises(CacheIntegrityError, match="identity mismatch"):
        validate_accepted_bundle(index, expected_identity=expected)
    mismatches = ["extractor_semantic_fingerprint"]
    waiver_payload = {
        "schema": integrity.EQUIVALENCE_WAIVER_SCHEMA,
        "accepted_index_sha256": integrity._sha256_file(index),
        "accepted_identity_sha256": integrity._fingerprint(actual),
        "expected_identity_sha256": integrity._fingerprint(expected),
        "waived_fields": mismatches,
        "reason": "Independent source review proved semantic equivalence.",
        "approved_by": "reviewer",
    }
    waiver = {
        **waiver_payload,
        "payload_sha256": integrity._fingerprint(waiver_payload),
    }
    waiver_path = tmp_path / "waiver.json"
    waiver_path.write_text(json.dumps(waiver), encoding="utf-8")
    result = validate_accepted_bundle(
        index,
        expected_identity=expected,
        equivalence_waiver=waiver_path,
    )
    assert result["waived_fields"] == mismatches
