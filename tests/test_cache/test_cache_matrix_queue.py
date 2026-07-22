from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import mprisk.cache.cache_matrix_queue as queue
from mprisk.cache.cache_matrix_queue import (
    AuxiliaryPackage,
    CacheJob,
    DomainProtocol,
    ModelSpec,
    _ledger_status,
    _smoke_status,
    _task_estimate,
    _write_cache_asset_signature,
    build_asset_signature,
    build_job_environment,
    load_matrix_config,
    normalize_manifest,
)


def test_normalize_manifest_resolves_media_and_adds_batch_fields(tmp_path: Path) -> None:
    media_root = tmp_path / "bundle"
    media = media_root / "datasets" / "sample.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"video")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "protocol": "VT",
                "sample_type": "Conflict",
                "media_paths": {"vision": "datasets/sample.mp4"},
                "text_content": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    domain = DomainProtocol(
        domain="source",
        protocol="vt",
        source_manifest=source,
        prepared_manifest=tmp_path / "prepared.jsonl",
        media_root=media_root,
        source_dataset="generated_3810",
        split="all",
        expected_samples=1,
    )

    rows, digest = normalize_manifest(domain)

    assert rows[0]["media_paths"] == {"vision": str(media.resolve())}
    assert rows[0]["source_dataset"] == "generated_3810"
    assert rows[0]["split"] == "all"
    assert rows[0]["annotation_count"] == 0
    assert rows[0]["use_in_main"] is True
    assert len(digest) == 64


def test_ledger_status_reports_missing_only_and_refuses_failed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    connection = sqlite3.connect(root / "batch_state.sqlite3")
    connection.execute("CREATE TABLE tasks(status TEXT)")
    connection.executemany(
        "INSERT INTO tasks(status) VALUES(?)", [("completed",), ("pending",), ("pending",)]
    )
    connection.commit()
    connection.close()

    assert _ledger_status(root, 3) == {
        "status": "incomplete",
        "counts": {"completed": 1, "pending": 2},
        "completed": 1,
        "missing": 2,
    }

    connection = sqlite3.connect(root / "batch_state.sqlite3")
    connection.execute("UPDATE tasks SET status='failed' WHERE rowid=2")
    connection.commit()
    connection.close()

    status = _ledger_status(root, 3)
    assert status["status"] == "failed"
    assert status["missing"] == 2


def test_task_estimate_separates_accepted_and_remaining() -> None:
    estimate = _task_estimate(
        [
            {
                "status": "accepted_bundle",
                "domain": "source",
                "protocol": "vt",
                "expected_tasks": 48,
            },
            {
                "status": "blocked_smoke",
                "domain": "target",
                "protocol": "vt",
                "expected_tasks": 48,
                "ledger": {"missing": 40},
            },
            {
                "status": "blocked_asset_signature",
                "domain": "target",
                "protocol": "va",
                "expected_tasks": 60,
                "ledger": {"missing": 60},
            },
        ]
    )

    assert estimate == {
        "total_tasks": 156,
        "completed_or_accepted_tasks": 56,
        "remaining_tasks": 100,
        "remaining_by_domain": {"target": 100},
        "remaining_by_protocol": {"vt": 40, "va": 60},
    }


def test_job_environment_prefers_selected_python_environment_lib(
    tmp_path: Path, monkeypatch
) -> None:
    environment = tmp_path / "env"
    environment_lib = environment / "lib"
    environment_lib.mkdir(parents=True)
    python = environment / "bin" / "python"
    monkeypatch.setenv("LD_LIBRARY_PATH", "/system/lib")
    config = SimpleNamespace(repo_root=tmp_path, cpu_threads_per_job=8)
    job = SimpleNamespace(
        model=SimpleNamespace(python=python, gpu_lane=1),
    )

    env = build_job_environment(config, job)

    assert env["LD_LIBRARY_PATH"] == f"{environment_lib.resolve()}:/system/lib"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_cache_asset_signature_is_required_for_resume(tmp_path: Path) -> None:
    model = ModelSpec(
        model_key="model",
        family="qwen_vl",
        protocol="vt",
        dtype="bfloat16",
        python=tmp_path / "python",
        gpu_lane=0,
        trajectory_shape=(2, 3),
        requested_frames=8,
        frame_protocol="fixed_uniform_temporal_samples_v1",
        video_sampling_method="uniform_midpoint_decord_v1",
        auxiliary_packages=(AuxiliaryPackage("decord", "decord"),),
        extra_args=(),
        invalidated_domains={},
        accepted_bundle_domains={},
    )
    domain = DomainProtocol(
        domain="target",
        protocol="vt",
        source_manifest=tmp_path / "source.jsonl",
        prepared_manifest=tmp_path / "prepared.jsonl",
        media_root=tmp_path,
        source_dataset="target",
        split="all",
        expected_samples=1,
    )
    job = CacheJob(domain, model, tmp_path / "cache", tmp_path / "smoke.json")
    signature = {"schema": "mprisk_cache_asset_signature_v1", "digest": "current"}

    _write_cache_asset_signature(job, signature)
    assert json.loads(job.asset_signature_evidence.read_text(encoding="utf-8")) == signature

    connection = sqlite3.connect(job.output_root / "batch_state.sqlite3")
    connection.execute("CREATE TABLE tasks(status TEXT)")
    connection.execute("INSERT INTO tasks(status) VALUES('pending')")
    connection.commit()
    connection.close()
    _write_cache_asset_signature(job, signature)
    with pytest.raises(RuntimeError, match="stale asset signature"):
        _write_cache_asset_signature(job, {**signature, "digest": "changed"})


def test_smoke_gate_requires_exact_48_task_contract(
    tmp_path: Path, monkeypatch
) -> None:
    prompt_set = tmp_path / "p8.yaml"
    prompt_set.write_text("p8\n", encoding="utf-8")
    environment = tmp_path / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "lib").mkdir()
    python = environment / "bin" / "python"
    python.write_text("", encoding="utf-8")
    model = ModelSpec(
        model_key="model",
        family="family",
        protocol="vt",
        dtype="bfloat16",
        python=python,
        gpu_lane=0,
        trajectory_shape=(32, 2560),
        requested_frames=8,
        frame_protocol="fixed_uniform_temporal_samples_v1",
        video_sampling_method="uniform_midpoint_decord_v1",
        auxiliary_packages=(AuxiliaryPackage("decord", "decord"),),
        extra_args=(),
        invalidated_domains={},
        accepted_bundle_domains={},
    )
    domain = DomainProtocol(
        domain="source",
        protocol="vt",
        source_manifest=tmp_path / "source.jsonl",
        prepared_manifest=tmp_path / "prepared.jsonl",
        media_root=tmp_path,
        source_dataset="source",
        split="all",
        expected_samples=1,
    )
    smoke = tmp_path / "SMOKE_COMPLETE.json"
    job = CacheJob(domain, model, tmp_path / "out", smoke)
    config = SimpleNamespace(prompt_sets={"vt": prompt_set})
    signature = {"schema": "mprisk_cache_asset_signature_v1", "digest": "asset"}
    monkeypatch.setattr(
        queue,
        "_asset_signature_status",
        lambda config, model: {"passed": True, "signature": signature},
    )
    payload = {
        "schema": "mprisk_cache_smoke_evidence_v2",
        "status": "PASS",
        "model_key": "model",
        "family": "family",
        "protocol": "vt",
        "domain": "source",
        "expected_tasks": 48,
        "completed_tasks": 48,
        "failed_tasks": 0,
        "prompt_set_sha256": hashlib.sha256(b"p8\n").hexdigest(),
        "environment_python": str(python),
        "runtime_library_path": str((environment / "lib").resolve()),
        "dtype": "bfloat16",
        "trajectory_shape": [32, 2560],
        "requested_frames": 8,
        "frame_protocol": "fixed_uniform_temporal_samples_v1",
        "video_sampling_method": "uniform_midpoint_decord_v1",
        "asset_signature": signature,
        "prompt_ids": [f"p{i}" for i in range(8)],
    }
    smoke.write_text(json.dumps(payload), encoding="utf-8")

    assert _smoke_status(config, job)["passed"] is True
    payload["completed_tasks"] = 47
    smoke.write_text(json.dumps(payload), encoding="utf-8")
    result = _smoke_status(config, job)
    assert result["passed"] is False
    assert "completed_tasks" in result["mismatches"]


def test_complete_matrix_freezes_frames_and_accepts_only_internvl() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_matrix_config(
        repo_root / "configs/cache/complete_cache_matrix_20260722.yaml"
    )

    by_key = {model.model_key: model for model in config.models}
    assert by_key["llava_v1_5_7b"].requested_frames == 7
    assert by_key["llava_v1_5_7b"].dtype == "float16"
    assert by_key["llava_onevision_qwen2_7b"].dtype == "float16"
    assert all(
        model.dtype == "bfloat16"
        for model in config.models
        if model.family not in {"llava_v15", "llava_onevision"}
    )
    assert {
        model.requested_frames
        for model in config.models
        if model.model_key != "llava_v1_5_7b"
    } == {8}
    assert all(
        model.frame_protocol == "fixed_uniform_temporal_samples_v1"
        for model in config.models
    )
    assert by_key["gemma4_12b"].video_sampling_method == "uniform_linspace_pyav_v1"
    assert by_key["phi4_multimodal"].video_sampling_method == "uniform_midpoint_ffmpeg_v1"
    assert (
        by_key["qwen2_5_omni_7b"].video_sampling_method
        == "uniform_nframes_qwen_omni_utils_v1"
    )
    assert all("--video-num-segments" not in model.extra_args for model in config.models)
    accepted = {
        (model.model_key, domain)
        for model in config.models
        for domain in model.accepted_bundle_domains
    }
    assert accepted == {("internvl3_5_8b", "source")}
    for model_key in (
        "qwen3_vl_8b",
        "qwen3_5_4b",
        "qwen2_5_omni_7b",
        "gemma4_12b",
    ):
        assert set(by_key[model_key].invalidated_domains) == {"source", "target"}


def test_asset_signature_captures_runtime_model_processor_and_wrapper(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    wrapper = repo_root / "src/mprisk/models/qwen_vl.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("class Wrapper: pass\n", encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
    (model_path / "processor_config.json").write_text(
        '{"processor_class":"Test"}\n', encoding="utf-8"
    )
    asset = SimpleNamespace(local_model_path=model_path)
    monkeypatch.setattr(queue, "load_model_assets", lambda path: [asset])
    monkeypatch.setattr(queue, "index_assets", lambda assets: {"model": asset})
    monkeypatch.setattr(
        queue,
        "_inspect_runtime",
        lambda python, auxiliary, runtime_library_path: {
            "sys_executable": "/env/bin/python",
            "transformers": {"path": "/env/transformers/__init__.py", "version": "5.5.3"},
            "auxiliary_packages": {
                "decord": {
                    "distribution": "decord",
                    "path": "/env/decord/__init__.py",
                    "version": "0.6.0",
                }
            },
        },
    )

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "status", "--porcelain"]:
            return SimpleNamespace(stdout="")
        if command[:3] == ["git", "log", "-1"]:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        raise AssertionError(command)

    monkeypatch.setattr(queue.subprocess, "run", fake_run)
    environment = tmp_path / "configured"
    (environment / "bin").mkdir(parents=True)
    (environment / "lib").mkdir()
    model = ModelSpec(
        model_key="model",
        family="qwen_vl",
        protocol="vt",
        dtype="bfloat16",
        python=environment / "bin" / "python",
        gpu_lane=0,
        trajectory_shape=(2, 3),
        requested_frames=8,
        frame_protocol="fixed_uniform_temporal_samples_v1",
        video_sampling_method="uniform_midpoint_decord_v1",
        auxiliary_packages=(AuxiliaryPackage("decord", "decord"),),
        extra_args=(),
        invalidated_domains={},
        accepted_bundle_domains={},
    )
    config = SimpleNamespace(
        asset_config=tmp_path / "assets.yaml", repo_root=repo_root
    )

    signature = build_asset_signature(config, model)

    assert signature["sys_executable"] == "/env/bin/python"
    assert signature["transformers"]["version"] == "5.5.3"
    assert signature["auxiliary_packages"]["decord"]["version"] == "0.6.0"
    assert signature["requested_frames"] == 8
    assert signature["dtype"] == "bfloat16"
    assert signature["video_sampling_method"] == "uniform_midpoint_decord_v1"
    assert signature["runtime_library_path"] == str((environment / "lib").resolve())
    assert signature["wrapper_git_sha"] == "a" * 40
    assert len(signature["model_config_sha256"]) == 64
    assert len(signature["processor_contract_sha256"]) == 64
