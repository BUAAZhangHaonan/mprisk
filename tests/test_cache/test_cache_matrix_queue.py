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
    GPUCapacityBusy,
    ModelSpec,
    _apply_python_isolation,
    _ledger_status,
    _scoped_execution_paths,
    _smoke_status,
    _task_estimate,
    _terminate_running_processes,
    _wait_for_gpu_capacity,
    _write_cache_asset_signature,
    build_asset_signature,
    build_job_environment,
    execute_matrix,
    load_matrix_config,
    normalize_manifest,
)


class _ProcessFixture:
    def __init__(self, *, hangs: bool) -> None:
        self.hangs = hangs
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []
        self.pid = 100

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.hangs:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.hangs and self.returncode is None:
            raise queue.subprocess.TimeoutExpired("fixture", timeout)
        return self.returncode


class _HandleFixture:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_dual_lane_cleanup_terminates_waits_kills_and_closes() -> None:
    graceful = _ProcessFixture(hangs=False)
    hung = _ProcessFixture(hangs=True)
    graceful_handle = _HandleFixture()
    hung_handle = _HandleFixture()
    running = {
        0: (SimpleNamespace(), graceful, graceful_handle),
        1: (SimpleNamespace(), hung, hung_handle),
    }

    _terminate_running_processes(running, timeout_seconds=0.01)

    assert graceful.terminate_calls == 1
    assert graceful.kill_calls == 0
    assert graceful.wait_calls == [None]
    assert hung.terminate_calls == 1
    assert hung.kill_calls == 1
    assert hung.wait_calls == [0.01, None]
    assert graceful_handle.closed is True
    assert hung_handle.closed is True
    assert running == {}


def test_scoped_execution_paths_isolate_source_gpu_lanes(tmp_path: Path) -> None:
    config = SimpleNamespace(
        lock_path=tmp_path / "matrix.lock",
        runtime_record=tmp_path / "matrix.json",
    )

    lock, runtime = _scoped_execution_paths(config, stage="source", lane=1)

    assert lock == tmp_path / "matrix.source.gpu1.lock"
    assert runtime == tmp_path / "matrix.source.gpu1.json"


def test_source_lane_execution_never_enters_target(tmp_path: Path, monkeypatch) -> None:
    source_job = SimpleNamespace(
        job_id="source:model",
        domain=SimpleNamespace(domain="source"),
        model=SimpleNamespace(gpu_lane=0),
    )
    target_job = SimpleNamespace(
        job_id="target:model",
        domain=SimpleNamespace(domain="target"),
        model=SimpleNamespace(gpu_lane=0),
    )
    config = SimpleNamespace(
        jobs=(source_job, target_job),
        lock_path=tmp_path / "matrix.lock",
        runtime_record=tmp_path / "matrix.json",
    )
    monkeypatch.setattr(
        queue,
        "audit_matrix",
        lambda config: {
            "capacity": {"safe": True},
            "job_records": [
                {"job_id": "source:model", "status": "ready"},
                {"job_id": "target:model", "status": "ready"},
            ],
        },
    )
    executed: list[tuple[list[object], Path]] = []
    monkeypatch.setattr(
        queue,
        "_execute_stage",
        lambda config, jobs, **kwargs: executed.append(
            (jobs, kwargs["runtime_record"])
        ),
    )

    execute_matrix(config, stage="source", lane=0)

    assert executed == [
        ([source_job], tmp_path / "matrix.source.gpu0.json")
    ]
    assert not (tmp_path / "matrix.source.gpu0.lock").exists()


def test_wait_for_gpu_capacity_retries_only_busy_state(monkeypatch) -> None:
    attempts = iter([GPUCapacityBusy("occupied"), None])
    sleeps: list[float] = []

    def require(lane, fraction):
        result = next(attempts)
        if result is not None:
            raise result

    monkeypatch.setattr(queue, "_require_gpu_capacity", require)
    monkeypatch.setattr(queue.time, "sleep", sleeps.append)

    _wait_for_gpu_capacity(1, 0.88, poll_interval_seconds=5.0)

    assert sleeps == [5.0]


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
        model=SimpleNamespace(
            python=python,
            python_no_user_site=False,
            env_isolation=False,
            gpu_lane=1,
        ),
    )

    env = build_job_environment(config, job)

    assert env["LD_LIBRARY_PATH"] == f"{environment_lib.resolve()}:/system/lib"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "PYTHONNOUSERSITE" not in env


def test_python_isolation_is_explicit_and_fail_closed() -> None:
    isolated = {"PYTHONNOUSERSITE": "0"}
    _apply_python_isolation(
        isolated, python_no_user_site=True, env_isolation=True
    )
    assert isolated == {
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    shared = {"PYTHONNOUSERSITE": "1"}
    _apply_python_isolation(
        shared, python_no_user_site=False, env_isolation=False
    )
    assert "PYTHONNOUSERSITE" not in shared

    with pytest.raises(ValueError, match="must be equal"):
        _apply_python_isolation(
            {}, python_no_user_site=True, env_isolation=False
        )


def test_cache_asset_signature_is_required_for_resume(tmp_path: Path) -> None:
    model = ModelSpec(
        model_key="model",
        family="qwen_vl",
        protocol="vt",
        dtype="bfloat16",
        python=tmp_path / "python",
        python_no_user_site=False,
        env_isolation=False,
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
    signature = {"schema": "mprisk_cache_asset_signature_v2", "digest": "current"}

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
        python_no_user_site=False,
        env_isolation=False,
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
    config = SimpleNamespace(
        prompt_sets={"vt": prompt_set},
        repo_root=tmp_path,
        cpu_threads_per_job=1,
    )
    signature = {
        "schema": "mprisk_cache_asset_signature_v2",
        "digest": "asset",
        "model_path": str(tmp_path / "model"),
        "model_config_sha256": "sha",
    }
    monkeypatch.setattr(
        queue,
        "build_asset_signature",
        lambda config, model, **kwargs: signature,
    )
    monkeypatch.setattr(
        queue, "load_context_ceiling", lambda **kwargs: 4096
    )
    monkeypatch.setattr(
        queue, "audit_smoke_cache_context", lambda **kwargs: 128
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
        "python_no_user_site": False,
        "env_isolation": False,
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


def test_dynamic_smoke_gate_uses_subset_plan_sha_not_full_plan_sha(
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
        model_key="llava_v1_5_7b",
        family="llava_v15",
        protocol="vt",
        dtype="float16",
        python=python,
        python_no_user_site=False,
        env_isolation=False,
        gpu_lane=0,
        trajectory_shape=(32, 4096),
        requested_frames=None,
        frame_protocol="per_sample_shared_uniform_temporal_samples_v1",
        video_sampling_method="uniform_midpoint_decord_v1",
        auxiliary_packages=(AuxiliaryPackage("decord", "decord"),),
        extra_args=(),
        invalidated_domains={},
        accepted_bundle_domains={},
        max_candidate_frames=8,
        context_budget_mode="per_sample_shared_max_legal",
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
    smoke = tmp_path / "smoke" / "SMOKE_COMPLETE.json"
    smoke.parent.mkdir()
    subset_plan = smoke.parent / "frame_plan.json"
    subset_plan.write_text("subset\n", encoding="utf-8")
    full_plan = tmp_path / "full-plan.json"
    full_plan.write_text("full\n", encoding="utf-8")
    job = CacheJob(domain, model, tmp_path / "out", smoke, frame_plan=full_plan)
    config = SimpleNamespace(
        prompt_sets={"vt": prompt_set},
        repo_root=tmp_path,
        cpu_threads_per_job=1,
    )
    signature = {
        "schema": "mprisk_cache_asset_signature_v2",
        "scope": "smoke",
        "model_path": str(tmp_path / "model"),
        "model_config_sha256": "sha",
    }
    monkeypatch.setattr(
        queue,
        "build_asset_signature",
        lambda config, model, **kwargs: signature,
    )
    monkeypatch.setattr(
        queue, "load_context_ceiling", lambda **kwargs: 4096
    )
    monkeypatch.setattr(
        queue, "audit_smoke_cache_context", lambda **kwargs: 128
    )
    payload = {
        "schema": "mprisk_cache_smoke_evidence_v2",
        "status": "PASS",
        "model_key": model.model_key,
        "family": model.family,
        "protocol": "vt",
        "domain": "source",
        "expected_tasks": 48,
        "completed_tasks": 48,
        "failed_tasks": 0,
        "prompt_set_sha256": hashlib.sha256(b"p8\n").hexdigest(),
        "environment_python": str(python),
        "python_no_user_site": False,
        "env_isolation": False,
        "runtime_library_path": str((environment / "lib").resolve()),
        "dtype": "float16",
        "trajectory_shape": [32, 4096],
        "requested_frames": None,
        "max_candidate_frames": 8,
        "context_budget_mode": "per_sample_shared_max_legal",
        "frame_plan_sha256": hashlib.sha256(b"subset\n").hexdigest(),
        "frame_protocol": model.frame_protocol,
        "video_sampling_method": model.video_sampling_method,
        "asset_signature": signature,
        "prompt_ids": [f"p{i}" for i in range(8)],
        "context_budget_evidence": {
            "schema": "mprisk_llava_v15_context_budget_smoke_evidence_v1",
            "frame_plan_schema": "mprisk_llava_v15_frame_plan_v1",
            "all_token_counts_within_context": True,
            "no_truncation": True,
        },
    }
    smoke.write_text(json.dumps(payload), encoding="utf-8")

    assert _smoke_status(config, job)["passed"] is True
    payload["frame_plan_sha256"] = hashlib.sha256(b"full\n").hexdigest()
    smoke.write_text(json.dumps(payload), encoding="utf-8")
    assert _smoke_status(config, job)["passed"] is False


def test_complete_matrix_uses_dynamic_llava_context_and_accepts_only_internvl() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_matrix_config(
        repo_root / "configs/cache/complete_cache_matrix.yaml"
    )

    by_key = {model.model_key: model for model in config.models}
    llava = by_key["llava_v1_5_7b"]
    assert llava.requested_frames is None
    assert llava.max_candidate_frames == 8
    assert llava.context_budget_mode == "per_sample_shared_max_legal"
    assert llava.frame_count_argument == 8
    assert llava.frame_protocol == "per_sample_shared_uniform_temporal_samples_v1"
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
        if model.model_key != "llava_v1_5_7b"
    )
    assert set(config.frame_plans) == {"source", "target"}
    assert by_key["gemma4_12b"].video_sampling_method == "uniform_midpoint_decord_v1"
    assert by_key["phi4_multimodal"].video_sampling_method == "uniform_midpoint_ffmpeg_v1"
    assert (
        by_key["qwen2_5_omni_7b"].video_sampling_method
        == "uniform_midpoint_decord_v1"
    )
    assert {package.module for package in by_key["gemma4_12b"].auxiliary_packages} == {
        "av",
        "decord",
    }
    assert {package.module for package in by_key["phi4_multimodal"].auxiliary_packages} == {
        "soundfile"
    }
    assert {
        package.module for package in by_key["qwen2_5_omni_7b"].auxiliary_packages
    } == {"qwen_omni_utils", "decord"}
    assert all("--video-num-segments" not in model.extra_args for model in config.models)
    assert {
        model.model_key
        for model in config.models
        if model.python_no_user_site and model.env_isolation
    } == {"gemma4_12b", "phi4_multimodal"}
    assert all(
        model.python_no_user_site == model.env_isolation
        for model in config.models
    )
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
        lambda python, auxiliary, runtime_library_path, python_no_user_site,
        env_isolation, family: {
            "sys_executable": "/env/bin/python",
            "python_no_user_site": False,
            "site_enable_user_site": True,
            "transformers": {"path": "/env/transformers/__init__.py", "version": "5.5.3"},
            "transformers_classes": {},
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
    monkeypatch.setattr(
        queue,
        "build_checkpoint_digest",
        lambda *args, **kwargs: {
            "schema": "mprisk_checkpoint_digest_receipt_v1",
            "checkpoint_sha256": "checkpoint",
            "files": [],
        },
    )
    monkeypatch.setattr(
        queue,
        "build_extractor_semantic_digest",
        lambda *args, **kwargs: {
            "schema": "mprisk_extractor_semantic_digest_v1",
            "sha256": "extractor",
            "repository_files_sha256": {"shared.py": "shared"},
            "trust_remote_code_files_sha256": {"remote.py": "remote"},
        },
    )
    monkeypatch.setattr(
        queue,
        "build_model_asset_inventory",
        lambda *args, **kwargs: {"sha256": "model-asset"},
    )
    environment = tmp_path / "configured"
    (environment / "bin").mkdir(parents=True)
    (environment / "lib").mkdir()
    model = ModelSpec(
        model_key="model",
        family="qwen_vl",
        protocol="vt",
        dtype="bfloat16",
        python=environment / "bin" / "python",
        python_no_user_site=False,
        env_isolation=False,
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
        asset_config=tmp_path / "assets.yaml",
        repo_root=repo_root,
        output_root=tmp_path / "outputs",
    )

    signature = build_asset_signature(config, model)

    processor_file_sha = hashlib.sha256(
        (model_path / "processor_config.json").read_bytes()
    ).hexdigest()
    expected_signature = {
        "schema": "mprisk_cache_asset_signature_v3",
        "model_key": "model",
        "family": "qwen_vl",
        "dtype": "bfloat16",
        "python_no_user_site": False,
        "env_isolation": False,
        "frame_protocol": "fixed_uniform_temporal_samples_v1",
        "requested_frames": 8,
        "video_sampling_method": "uniform_midpoint_decord_v1",
        "runtime_library_path": str((environment / "lib").resolve()),
        "sys_executable": "/env/bin/python",
        "transformers": {
            "path": "/env/transformers/__init__.py",
            "version": "5.5.3",
        },
        "transformers_classes": {},
        "auxiliary_packages": {
            "decord": {
                "distribution": "decord",
                "path": "/env/decord/__init__.py",
                "version": "0.6.0",
            }
        },
        "model_path": str(model_path.resolve()),
        "model_config_sha256": hashlib.sha256(
            (model_path / "config.json").read_bytes()
        ).hexdigest(),
        "checkpoint_digest_schema": "mprisk_checkpoint_digest_receipt_v1",
        "checkpoint_sha256": "checkpoint",
        "checkpoint_digest_receipt": str(
            tmp_path / "outputs/receipts/checkpoints/model.json"
        ),
        "model_asset_fingerprint": "model-asset",
        "extractor_semantic_schema": "mprisk_extractor_semantic_digest_v1",
        "extractor_semantic_sha256": "extractor",
        "extractor_semantic_files": {
            "repository": {"shared.py": "shared"},
            "trust_remote_code": {"remote.py": "remote"},
        },
        "processor_contract_sha256": hashlib.sha256(
            queue._canonical_json(
                {"processor_config.json": processor_file_sha}
            ).encode()
        ).hexdigest(),
        "processor_files": {"processor_config.json": processor_file_sha},
        "wrapper_path": "src/mprisk/models/qwen_vl.py",
        "wrapper_git_sha": "a" * 40,
        "wrapper_file_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    }

    assert queue._canonical_json(signature) == queue._canonical_json(expected_signature)

    assert signature["sys_executable"] == "/env/bin/python"
    assert signature["schema"] == "mprisk_cache_asset_signature_v3"
    assert signature["python_no_user_site"] is False
    assert signature["env_isolation"] is False
    assert signature["transformers_classes"] == {}
    assert signature["transformers"]["version"] == "5.5.3"
    assert signature["auxiliary_packages"]["decord"]["version"] == "0.6.0"
    assert signature["requested_frames"] == 8
    assert "max_candidate_frames" not in signature
    assert "context_budget_mode" not in signature
    assert signature["dtype"] == "bfloat16"
    assert signature["video_sampling_method"] == "uniform_midpoint_decord_v1"
    assert signature["runtime_library_path"] == str((environment / "lib").resolve())
    assert signature["wrapper_git_sha"] == "a" * 40
    assert len(signature["model_config_sha256"]) == 64
    assert len(signature["processor_contract_sha256"]) == 64
