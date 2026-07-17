from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mprisk.cache.cache_union import (
    CacheSource,
    CacheUnionError,
    ExpectedCacheTask,
    blocked_tasks_from_rows,
    build_cache_union,
    write_extractor_evidence,
)
from mprisk.cache.prefill_writer import write_prefill_result
from mprisk.models.base_wrapper import PrefillRequest, PrefillResult


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _task_id(request: PrefillRequest) -> str:
    identity = {
        "sample_id": request.sample_id,
        "prompt_id": request.prompt_id,
        "condition": request.condition,
        "protocol": request.protocol,
        "model_key": request.model_key,
    }
    return hashlib.sha256(_canonical(identity).encode()).hexdigest()


def _signature(model_path: Path, *, manifest: str) -> dict[str, object]:
    return {
        "schema": "mprisk_prefill_batch_signature_v2",
        "asset_config_sha256": "a" * 64,
        "manifest_sha256": manifest,
        "prompt_set_sha256": "p" * 64,
        "prompt_ids": ["p1"],
        "prompt_variables": {},
        "protocol": "vt",
        "conditions": ["M1", "M2", "M12"],
        "model_key": "model",
        "family": "qwen_vl",
        "model_path": str(model_path.resolve()),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "min_pixels": None,
        "max_pixels": None,
        "joint_audio_mode": "embedded_video",
        "video_fps": 1.0,
        "video_num_segments": 8,
        "internvl_max_num": 1,
    }


def _make_model(tmp_path: Path) -> tuple[Path, str, str]:
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    weight_index = model / "model.safetensors.index.json"
    shard = model / "model-00001-of-00001.safetensors"
    config.write_text('{"model":"fake"}\n', encoding="utf-8")
    shard.write_bytes(b"fake weights")
    weight_index.write_text(
        json.dumps({"weight_map": {"layer.weight": shard.name}}) + "\n",
        encoding="utf-8",
    )
    return model, _sha256(config), _sha256(weight_index)


def _make_code_repo(tmp_path: Path, *, marker: str = "same") -> Path:
    repository = tmp_path / f"code-{marker}"
    paths = (
        "src/mprisk/assets/registry.py",
        "src/mprisk/cache/prefill_batch.py",
        "src/mprisk/cache/prefill_writer.py",
        "src/mprisk/models/base_wrapper.py",
        "src/mprisk/models/qwen_omni.py",
        "src/mprisk/models/qwen_vl.py",
        "src/mprisk/models/wrapper_registry.py",
        "src/mprisk/prompts/compiler.py",
        "src/mprisk/prompts/template_bank.py",
    )
    for relative in paths:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {marker}: {relative}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "semantic code"], check=True)
    return repository


def _request(sample_id: str, *, split: str, condition: str = "M1") -> PrefillRequest:
    return PrefillRequest(
        sample_id=sample_id,
        model_key="model",
        protocol="vt",
        condition=condition,
        prompt_set_key="prompts",
        prompt_id="p1",
        dataset_key="delivery_20260716",
        split=split,
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"/media/{sample_id}.mp4", "fps": 1.0},
                    {"type": "text", "text": "Describe the overall state."},
                ],
            },
        ),
        media_paths={"vision": f"/media/{sample_id}.mp4"},
        use_audio_in_video=False,
    )


def _source(
    tmp_path: Path,
    *,
    source_id: str,
    request: PrefillRequest,
    signature: dict[str, object],
    code_repo: Path,
    config_sha256: str,
    weight_sha256: str,
    status: str = "completed",
) -> CacheSource:
    return _source_many(
        tmp_path,
        source_id=source_id,
        requests=(request,),
        signature=signature,
        code_repo=code_repo,
        config_sha256=config_sha256,
        weight_sha256=weight_sha256,
        statuses={_task_id(request): status},
    )


def _source_many(
    tmp_path: Path,
    *,
    source_id: str,
    requests: tuple[PrefillRequest, ...],
    signature: dict[str, object],
    code_repo: Path,
    config_sha256: str,
    weight_sha256: str,
    statuses: dict[str, str] | None = None,
) -> CacheSource:
    source_root = tmp_path / source_id
    ledger = source_root / "batch_state.sqlite3"
    source_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ledger)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE tasks (
          task_id TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          entry_json TEXT
        );
        """
    )
    connection.execute("INSERT INTO metadata VALUES('signature',?)", (_canonical(signature),))
    for request in requests:
        artifact_root = source_root / "prompts" / request.prompt_id
        result = PrefillResult(
            request=request,
            trajectory=np.ones((2, 3), dtype=np.float32),
            token_count=5,
            t0_token_index=4,
            provenance={
                "schema": "fake_prefill_v1",
                "model_path": signature["model_path"],
                "model_class": "FakeModel",
                "processor_class": "FakeProcessor",
                "transformers_version": "1",
                "torch_version": "1",
                "source_dtype": "bfloat16",
                "stored_dtype": "float32",
                "attn_implementation": "sdpa",
                "num_hidden_layers": 2,
                "hidden_size": 3,
                "hidden_state_index_offset": 1,
                "model_config_sha256": config_sha256,
                "weight_index_sha256": weight_sha256,
                "video_fps": None if request.condition == "M2" else 1.0,
            },
        )
        artifact = write_prefill_result(result, output_root=artifact_root, update_manifest=False)
        status = (statuses or {}).get(_task_id(request), "completed")
        connection.execute(
            "INSERT INTO tasks VALUES(?,?,?)",
            (
                _task_id(request),
                status,
                _canonical(artifact.entry) if status == "completed" else None,
            ),
        )
    connection.commit()
    connection.close()
    evidence = source_root / "extractor_evidence.json"
    write_extractor_evidence(
        source_id=source_id,
        ledger_path=ledger,
        cache_root=source_root,
        code_root=code_repo,
        output_path=evidence,
    )
    return CacheSource(source_id, source_root.resolve(), ledger.resolve(), evidence.resolve())


def _set_runtime_provenance_field(
    source: CacheSource,
    request: PrefillRequest,
    field: str,
    value: object,
) -> None:
    connection = sqlite3.connect(source.ledger_path)
    entry = json.loads(
        connection.execute(
            "SELECT entry_json FROM tasks WHERE task_id=?",
            (_task_id(request),),
        ).fetchone()[0]
    )
    connection.close()
    sidecar = Path(entry["cache_root"]) / entry["metadata"]["sidecar_path"]
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["provenance"][field] = value
    sidecar.write_text(json.dumps(payload), encoding="utf-8")


def _expected(request: PrefillRequest) -> ExpectedCacheTask:
    return ExpectedCacheTask(
        task_id=_task_id(request),
        request=request,
        sample_type="Conflict",
        split=request.split,
        source_dataset=request.dataset_key,
    )


def _set_evidence_strategy(source: CacheSource, *, strategy: str, version: str) -> None:
    payload = json.loads(source.evidence_path.read_text(encoding="utf-8"))
    payload["prefill_strategy"] = strategy
    payload["prefill_strategy_version"] = version
    fingerprint_input = {
        "strategy": strategy,
        "strategy_version": version,
        "code_files_sha256": payload["code_files_sha256"],
        "extraction_signature": payload["extraction_signature"],
        "model_asset_fingerprint": payload["model_asset_fingerprint"],
    }
    payload["extractor_semantic_fingerprint"] = hashlib.sha256(
        _canonical(fingerprint_input).encode()
    ).hexdigest()
    source.evidence_path.write_text(json.dumps(payload), encoding="utf-8")


def test_union_resolves_disjoint_sources_and_attaches_only_new_split(tmp_path: Path) -> None:
    model, config_sha, weight_sha = _make_model(tmp_path)
    code_repo = _make_code_repo(tmp_path)
    expected_signature = _signature(model, manifest="new-full")
    expected_signature["prompt_ids"] = ("p1",)
    expected_signature["conditions"] = ("M1", "M2", "M12")
    first_source_request = _request("sample-a", split="delivery_test")
    second_source_request = _request("sample-b", split="delivery_train")
    sources = [
        _source(
            tmp_path,
            source_id="new-only",
            request=first_source_request,
            signature=_signature(model, manifest="new-only"),
            code_repo=code_repo,
            config_sha256=config_sha,
            weight_sha256=weight_sha,
        ),
        _source(
            tmp_path,
            source_id="overlap-reextracted",
            request=second_source_request,
            signature=_signature(model, manifest="overlap"),
            code_repo=code_repo,
            config_sha256=config_sha,
            weight_sha256=weight_sha,
        ),
    ]
    expected = [
        _expected(_request("sample-a", split="relation_train")),
        _expected(_request("sample-b", split="official_test")),
    ]

    result = build_cache_union(
        expected_tasks=expected,
        expected_signature=expected_signature,
        sources=sources,
        output_path=tmp_path / "union.json",
        expected_resolved_tasks=2,
        expected_raw_tasks=2,
        checksum_workers=2,
    )

    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "mprisk_prefill_cache_union_v2"
    assert payload["version"] == "v2"
    assert payload["prefill_strategy"] == "full_prefill"
    assert payload["prefill_strategy_version"] == "v1"
    assert {entry["split"] for entry in payload["entries"]} == {
        "relation_train",
        "official_test",
    }
    assert {entry["source_provenance"]["source_split"] for entry in payload["entries"]} == {
        "delivery_test",
        "delivery_train",
    }
    assert result.source_counts == {"new-only": 1, "overlap-reextracted": 1}
    assert all(Path(entry["shard_path"]).is_absolute() for entry in payload["entries"])


def test_union_fails_on_missing_duplicate_or_failed_expected_task(tmp_path: Path) -> None:
    model, config_sha, weight_sha = _make_model(tmp_path)
    code_repo = _make_code_repo(tmp_path)
    signature = _signature(model, manifest="expected")
    request = _request("sample", split="train")
    source = _source(
        tmp_path,
        source_id="one",
        request=request,
        signature=_signature(model, manifest="one"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    kwargs = {
        "expected_tasks": [_expected(request)],
        "expected_signature": signature,
        "output_path": tmp_path / "union.json",
        "expected_resolved_tasks": 1,
        "expected_raw_tasks": 1,
        "checksum_workers": 1,
    }
    with pytest.raises(CacheUnionError, match="At least one"):
        build_cache_union(sources=[], **kwargs)

    other_request = _request("other", split="train")
    other = _source(
        tmp_path,
        source_id="other",
        request=other_request,
        signature=_signature(model, manifest="other"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    with pytest.raises(CacheUnionError, match="missing"):
        build_cache_union(sources=[other], **kwargs)

    duplicate = _source(
        tmp_path,
        source_id="duplicate",
        request=request,
        signature=_signature(model, manifest="duplicate"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    with pytest.raises(CacheUnionError, match="duplicated"):
        build_cache_union(sources=[source, duplicate], **kwargs)

    failed = _source(
        tmp_path,
        source_id="failed",
        request=request,
        signature=_signature(model, manifest="failed"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
        status="failed",
    )
    with pytest.raises(CacheUnionError, match="not completed"):
        build_cache_union(sources=[failed], **kwargs)


def test_union_rejects_different_extractor_semantic_fingerprints(tmp_path: Path) -> None:
    model, config_sha, weight_sha = _make_model(tmp_path)
    first_code = _make_code_repo(tmp_path, marker="first")
    second_code = _make_code_repo(tmp_path, marker="second")
    signature = _signature(model, manifest="expected")
    first_request = _request("first", split="train")
    second_request = _request("second", split="train")
    sources = [
        _source(
            tmp_path,
            source_id="first",
            request=first_request,
            signature=_signature(model, manifest="first"),
            code_repo=first_code,
            config_sha256=config_sha,
            weight_sha256=weight_sha,
        ),
        _source(
            tmp_path,
            source_id="second",
            request=second_request,
            signature=_signature(model, manifest="second"),
            code_repo=second_code,
            config_sha256=config_sha,
            weight_sha256=weight_sha,
        ),
    ]

    with pytest.raises(CacheUnionError, match="different extractor semantic fingerprints"):
        build_cache_union(
            expected_tasks=[_expected(first_request), _expected(second_request)],
            expected_signature=signature,
            sources=sources,
            output_path=tmp_path / "union.json",
            expected_resolved_tasks=2,
            expected_raw_tasks=2,
            checksum_workers=1,
        )


def test_union_rejects_missing_or_disagreeing_prefill_strategy_identity(
    tmp_path: Path,
) -> None:
    model, config_sha, weight_sha = _make_model(tmp_path)
    code_repo = _make_code_repo(tmp_path)
    signature = _signature(model, manifest="expected")
    first_request = _request("first", split="train")
    second_request = _request("second", split="train")
    first = _source(
        tmp_path,
        source_id="first",
        request=first_request,
        signature=_signature(model, manifest="first"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    second = _source(
        tmp_path,
        source_id="second",
        request=second_request,
        signature=_signature(model, manifest="second"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    kwargs = {
        "expected_tasks": [_expected(first_request), _expected(second_request)],
        "expected_signature": signature,
        "sources": [first, second],
        "output_path": tmp_path / "union.json",
        "expected_resolved_tasks": 2,
        "expected_raw_tasks": 2,
        "checksum_workers": 1,
    }

    missing = json.loads(first.evidence_path.read_text(encoding="utf-8"))
    del missing["prefill_strategy"]
    first.evidence_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(CacheUnionError, match="Prefill strategy is missing"):
        build_cache_union(**kwargs)

    write_extractor_evidence(
        source_id=first.source_id,
        ledger_path=first.ledger_path,
        cache_root=first.cache_root,
        code_root=code_repo,
        output_path=first.evidence_path,
    )
    _set_evidence_strategy(second, strategy="prompt_kv", version="v2")
    with pytest.raises(CacheUnionError, match="different prefill strategy identities"):
        build_cache_union(**kwargs)


def test_union_rejects_semantic_request_or_checksum_mismatch(tmp_path: Path) -> None:
    model, config_sha, weight_sha = _make_model(tmp_path)
    code_repo = _make_code_repo(tmp_path)
    signature = _signature(model, manifest="expected")
    source_request = _request("sample", split="train")
    source = _source(
        tmp_path,
        source_id="source",
        request=source_request,
        signature=_signature(model, manifest="source"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    changed = _request("sample", split="train")
    object.__setattr__(
        changed,
        "messages",
        ({"role": "user", "content": [{"type": "text", "text": "changed"}]},),
    )
    with pytest.raises(CacheUnionError, match="Semantic request fingerprint mismatch"):
        build_cache_union(
            expected_tasks=[_expected(changed)],
            expected_signature=signature,
            sources=[source],
            output_path=tmp_path / "union.json",
            expected_resolved_tasks=1,
            expected_raw_tasks=1,
            checksum_workers=1,
        )

    connection = sqlite3.connect(source.ledger_path)
    source_entry = json.loads(connection.execute("SELECT entry_json FROM tasks").fetchone()[0])
    connection.close()
    shard = Path(source_entry["cache_root"]) / source_entry["shard_path"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(CacheUnionError, match="checksum mismatch"):
        build_cache_union(
            expected_tasks=[_expected(source_request)],
            expected_signature=signature,
            sources=[source],
            output_path=tmp_path / "union.json",
            expected_resolved_tasks=1,
            expected_raw_tasks=1,
            checksum_workers=1,
        )


def test_blocked_tasks_are_accounted_but_never_exposed(tmp_path: Path) -> None:
    model, config_sha, weight_sha = _make_model(tmp_path)
    code_repo = _make_code_repo(tmp_path)
    signature = _signature(model, manifest="expected")
    request = _request("valid", split="train")
    source = _source(
        tmp_path,
        source_id="source",
        request=request,
        signature=_signature(model, manifest="source"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    blocked = blocked_tasks_from_rows(
        [{"sample_id": "invalid", "reason": "missing_audio_stream"}],
        model_key="model",
        protocol="vt",
        prompt_ids=("p1",),
        conditions=("M1",),
    )

    result = build_cache_union(
        expected_tasks=[_expected(request)],
        expected_signature=signature,
        sources=[source],
        output_path=tmp_path / "union.json",
        blocked_tasks=blocked,
        expected_resolved_tasks=1,
        expected_blocked_tasks=1,
        expected_raw_tasks=2,
        checksum_workers=1,
    )

    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1
    assert payload["blocked_tasks"] == [
        {
            "condition": "M1",
            "exposed_as_cache_entry": False,
            "prompt_id": "p1",
            "reason": "missing_audio_stream",
            "sample_id": "invalid",
            "source_status": "not_scheduled",
            "task_id": blocked[0].task_id,
        }
    ]


def test_union_evidence_binds_every_referenced_model_weight_shard(tmp_path: Path) -> None:
    model, config_sha, weight_sha = _make_model(tmp_path)
    code_repo = _make_code_repo(tmp_path)
    signature = _signature(model, manifest="expected")
    request = _request("sample", split="train")
    source = _source(
        tmp_path,
        source_id="source",
        request=request,
        signature=_signature(model, manifest="source"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    evidence = json.loads(source.evidence_path.read_text(encoding="utf-8"))
    assets = {item["path"]: item for item in evidence["model_asset_inventory"]["files"]}
    shard_name = "model-00001-of-00001.safetensors"
    assert assets[shard_name]["role"] == "weight_shard"
    assert assets[shard_name]["bytes"] == len(b"fake weights")

    (model / shard_name).write_bytes(b"changed weights")
    with pytest.raises(CacheUnionError, match="Model asset fingerprint changed"):
        build_cache_union(
            expected_tasks=[_expected(request)],
            expected_signature=signature,
            sources=[source],
            output_path=tmp_path / "union.json",
            expected_resolved_tasks=1,
            expected_raw_tasks=1,
            checksum_workers=1,
        )


def test_runtime_fingerprints_are_condition_specific_and_match_across_sources(
    tmp_path: Path,
) -> None:
    model, config_sha, weight_sha = _make_model(tmp_path)
    code_repo = _make_code_repo(tmp_path)
    signature = _signature(model, manifest="expected")
    first_requests = (
        _request("first-m1", split="train", condition="M1"),
        _request("first-m2", split="train", condition="M2"),
        _request("first-m12", split="train", condition="M12"),
    )
    second_requests = (
        _request("second-m1", split="train", condition="M1"),
        _request("second-m2", split="train", condition="M2"),
        _request("second-m12", split="train", condition="M12"),
    )
    first = _source_many(
        tmp_path,
        source_id="first",
        requests=first_requests,
        signature=_signature(model, manifest="first"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    second = _source_many(
        tmp_path,
        source_id="second",
        requests=second_requests,
        signature=_signature(model, manifest="second"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    expected = [_expected(request) for request in (*first_requests, *second_requests)]
    common = {
        "expected_tasks": expected,
        "expected_signature": signature,
        "sources": [first, second],
        "expected_resolved_tasks": 6,
        "expected_raw_tasks": 6,
        "checksum_workers": 1,
    }

    result = build_cache_union(output_path=tmp_path / "valid-union.json", **common)
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    runtime_map = payload["provenance"]["runtime_provenance_fingerprints_by_condition"]
    assert set(runtime_map) == {"M1", "M2", "M12"}
    assert runtime_map["M1"] == runtime_map["M12"]
    assert runtime_map["M1"] != runtime_map["M2"]
    assert all(
        entry["source_provenance"]["runtime_condition"] == entry["condition"]
        for entry in payload["entries"]
    )
    m2_entries = [entry for entry in payload["entries"] if entry["condition"] == "M2"]
    assert all(
        entry["source_provenance"]["runtime_provenance"]["video_fps"] is None
        for entry in m2_entries
    )

    _set_runtime_provenance_field(second, second_requests[1], "video_fps", 2.0)
    with pytest.raises(CacheUnionError, match="different condition-specific runtime"):
        build_cache_union(output_path=tmp_path / "cross-source-mismatch.json", **common)


def test_union_rejects_multiple_runtime_fingerprints_within_one_condition(
    tmp_path: Path,
) -> None:
    model, config_sha, weight_sha = _make_model(tmp_path)
    code_repo = _make_code_repo(tmp_path)
    signature = _signature(model, manifest="expected")
    requests = (
        _request("first-m1", split="train", condition="M1"),
        _request("second-m1", split="train", condition="M1"),
    )
    source = _source_many(
        tmp_path,
        source_id="source",
        requests=requests,
        signature=_signature(model, manifest="source"),
        code_repo=code_repo,
        config_sha256=config_sha,
        weight_sha256=weight_sha,
    )
    _set_runtime_provenance_field(source, requests[1], "video_fps", 2.0)

    with pytest.raises(CacheUnionError, match="multiple runtime fingerprints for M1"):
        build_cache_union(
            expected_tasks=[_expected(request) for request in requests],
            expected_signature=signature,
            sources=[source],
            output_path=tmp_path / "union.json",
            expected_resolved_tasks=2,
            expected_raw_tasks=2,
            checksum_workers=1,
        )
