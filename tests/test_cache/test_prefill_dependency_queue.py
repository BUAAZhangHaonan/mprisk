from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import mprisk.cache.prefill_dependency_queue as dependency_queue
from mprisk.cache.prefill_dependency_queue import (
    CapacityFailure,
    GateFailure,
    QueueLockError,
    QueueScopeLock,
    UpstreamStatus,
    evaluate_capacity,
    evaluate_main_gate,
    evaluate_upstream_activity,
    load_queue_manifest,
    run_dependency_queue,
    wait_for_main_gate,
)

QUEUE_CONFIG = Path("configs/cache/prefill_followup_p8_queue.yaml")


def _write_ledger(path: Path, statuses: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE tasks (task_id TEXT PRIMARY KEY, status TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO tasks(task_id,status) VALUES (?,?)",
        [(f"task-{index}", status) for index, status in enumerate(statuses)],
    )
    connection.commit()
    connection.close()


def _write_runtime_record(path: Path, cache_statuses: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mprisk_run_records_v1",
                "class_code": {"A": "Conflict", "C": "Aligned"},
                "class_code_semantics": {
                    "A": "sample_type.Conflict",
                    "C": "sample_type.Aligned",
                },
                "commands": [],
                "gpus": [],
                "caches": [
                    {"cache_key": key, "status": status}
                    for key, status in cache_statuses.items()
                ],
                "experiments": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(tmp_path: Path, *, expected_tasks: int = 2) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    calibration = tmp_path / "calibration"
    shard = calibration / "prompts/p1/shards/model/vt/M1/sample.safetensors"
    sidecar = shard.with_suffix(".json")
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"s" * 100)
    sidecar.write_bytes(b"j" * 20)
    (calibration / "manifest.jsonl").write_text(
        json.dumps(
            {
                "cache_root": str(calibration / "prompts/p1"),
                "shard_path": "shards/model/vt/M1/sample.safetensors",
                "metadata": {"sidecar_path": "shards/model/vt/M1/sample.json"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "queue.yaml"
    manifest.write_text(
        f"""
schema: mprisk_prefill_dependency_queue_v1
version: v1
physical_gpu: 1
device: cuda:0
python: /test/python
extract_script: scripts/extract_prefill_batch.py
runtime_record: {tmp_path / 'followup-runtime.json'}
lock_path: {tmp_path / 'followup.lock'}
capacity_gate:
  filesystem_path: {tmp_path}
  max_projected_utilization: 0.9
  models:
    - model_key: model1
      calibration_root: {calibration}
      outputs:
        - output_root: {tmp_path / 'seed1-model1'}
          expected_tasks: {expected_tasks}
        - output_root: {tmp_path / 'seed2-model1'}
          expected_tasks: {expected_tasks}
main_gate:
  runtime_record: {tmp_path / 'main-runtime.json'}
  upstream:
    tmux_session: test-main-queue
    pid: {os.getpid()}
    heartbeat_max_age_seconds: 300
    heartbeat_paths:
      - {tmp_path / 'main' / 'batch_state.sqlite3'}
  jobs:
    - model_key: main_model
      ledger: {tmp_path / 'main' / 'batch_state.sqlite3'}
      expected_tasks: {expected_tasks}
      runtime_cache_key: main_model_main_p8
followup_jobs:
  - job_id: seed1_model1
    seed: 1
    model_key: model1
    protocol: vt
    manifest: manifest.jsonl
    prompt_set: prompts.yaml
    output_root: {tmp_path / 'seed1-model1'}
    log_path: {tmp_path / 'seed1-model1.log'}
    expected_tasks: {expected_tasks}
    extra_args: []
  - job_id: seed2_model1
    seed: 2
    model_key: model1
    protocol: vt
    manifest: manifest.jsonl
    prompt_set: prompts2.yaml
    output_root: {tmp_path / 'seed2-model1'}
    log_path: {tmp_path / 'seed2-model1.log'}
    expected_tasks: {expected_tasks}
    extra_args: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_versioned_followup_manifest_freezes_six_ordered_jobs() -> None:
    queue = load_queue_manifest(QUEUE_CONFIG)

    assert queue.physical_gpu == 1
    assert [job.seed for job in queue.followup_jobs] == [20260715] * 3 + [20260716] * 3
    assert [job.model_key for job in queue.followup_jobs] == [
        "qwen3_vl_8b",
        "internvl3_5_8b",
        "qwen2_5_omni_7b",
    ] * 2
    assert [job.expected_tasks for job in queue.followup_jobs] == [
        60288,
        60288,
        53808,
    ] * 2
    assert len({job.output_root for job in queue.followup_jobs}) == 6
    assert queue.lock_path.name == "prefill_followup_p8_queue_v1.lock"
    assert queue.main_gate.upstream.tmux_session == "mprisk-main-p8-queue"
    assert len(queue.main_gate.upstream.heartbeat_paths) == 10


def test_gate_requires_exact_sqlite_completion_and_runtime_records(tmp_path: Path) -> None:
    queue = load_queue_manifest(_write_manifest(tmp_path))
    gate = queue.main_gate.jobs[0]
    _write_ledger(gate.ledger, ["completed", "completed"])

    status = evaluate_main_gate(queue)
    assert status.ready is False
    assert status.reasons == ("runtime record is missing",)

    _write_runtime_record(queue.main_gate.runtime_record, {gate.runtime_cache_key: "complete"})
    status = evaluate_main_gate(queue)
    assert status.ready is True
    assert status.reasons == ()


def test_capacity_gate_projects_artifact_bytes_and_hard_blocks_above_limit(
    tmp_path: Path,
) -> None:
    queue = load_queue_manifest(_write_manifest(tmp_path))
    safe_filesystem = SimpleNamespace(
        f_frsize=1,
        f_blocks=1000,
        f_bfree=500,
        f_bavail=500,
        f_files=1000,
        f_ffree=500,
        f_favail=500,
    )
    status = evaluate_capacity(queue, statvfs_fn=lambda _: safe_filesystem)
    assert status.free_bytes == 500
    assert status.projected_bytes == 480
    assert status.projected_utilization == pytest.approx(0.98)
    assert status.safe is False
    with pytest.raises(CapacityFailure, match="98.00%"):
        status.require_safe()

    reserved_filesystem = SimpleNamespace(
        **{**vars(safe_filesystem), "f_bavail": 400}
    )
    reserved = evaluate_capacity(queue, statvfs_fn=lambda _: reserved_filesystem)
    assert reserved.projected_utilization == pytest.approx(980 / 900)

    roomy_filesystem = SimpleNamespace(
        **{
            **vars(safe_filesystem),
            "f_blocks": 10000,
            "f_bfree": 9500,
            "f_bavail": 9500,
        }
    )
    roomy = evaluate_capacity(queue, statvfs_fn=lambda _: roomy_filesystem)
    assert roomy.safe is True
    assert roomy.projected_inode_utilization < 0.9


def test_gate_hard_fails_on_failed_or_wrong_sized_ledger(tmp_path: Path) -> None:
    failed_queue = load_queue_manifest(_write_manifest(tmp_path / "failed"))
    failed_gate = failed_queue.main_gate.jobs[0]
    _write_ledger(failed_gate.ledger, ["completed", "failed"])
    with pytest.raises(GateFailure, match="failed=1"):
        evaluate_main_gate(failed_queue)

    wrong_queue = load_queue_manifest(_write_manifest(tmp_path / "wrong", expected_tasks=3))
    wrong_gate = wrong_queue.main_gate.jobs[0]
    _write_ledger(wrong_gate.ledger, ["completed", "completed"])
    with pytest.raises(GateFailure, match="expected 3 tasks, found 2"):
        evaluate_main_gate(wrong_queue)


def test_queue_scope_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    queue = load_queue_manifest(_write_manifest(tmp_path))

    with QueueScopeLock(queue):
        with pytest.raises(QueueLockError, match="already locked"):
            with QueueScopeLock(queue):
                pytest.fail("duplicate lock unexpectedly acquired")

    with QueueScopeLock(queue):
        assert queue.lock_path.is_file()


def test_lock_and_capacity_failures_write_atomic_runtime_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locked_queue = load_queue_manifest(_write_manifest(tmp_path / "locked"))
    with QueueScopeLock(locked_queue):
        with pytest.raises(QueueLockError, match="already locked"):
            run_dependency_queue(locked_queue)
    locked_runtime = json.loads(locked_queue.runtime_record.read_text(encoding="utf-8"))
    assert locked_runtime["dependency_queue"]["status"] == "failure"
    assert locked_runtime["dependency_queue"]["failure_code"] == "lock_unavailable"

    capacity_queue = load_queue_manifest(_write_manifest(tmp_path / "capacity"))
    full_filesystem = SimpleNamespace(
        f_frsize=1,
        f_blocks=1000,
        f_bfree=500,
        f_bavail=500,
        f_files=1000,
        f_ffree=500,
        f_favail=500,
    )
    blocked = evaluate_capacity(capacity_queue, statvfs_fn=lambda _: full_filesystem)
    monkeypatch.setattr(dependency_queue, "evaluate_capacity", lambda _: blocked)
    with pytest.raises(CapacityFailure):
        run_dependency_queue(capacity_queue)
    capacity_runtime = json.loads(capacity_queue.runtime_record.read_text(encoding="utf-8"))
    assert capacity_runtime["dependency_queue"]["status"] == "failure"
    assert capacity_runtime["dependency_queue"]["failure_code"] == "capacity_gate"


def test_dead_upstream_hard_fails_and_writes_atomic_runtime_record(tmp_path: Path) -> None:
    queue = load_queue_manifest(_write_manifest(tmp_path))
    dead = UpstreamStatus(
        running=False,
        reason="tmux session is missing",
        heartbeat_age_seconds=None,
        seconds_until_stale=0,
    )

    with pytest.raises(GateFailure, match="upstream_not_running"):
        run_dependency_queue(queue, upstream_checker=lambda _: dead)

    runtime = json.loads(queue.runtime_record.read_text(encoding="utf-8"))
    assert runtime["dependency_queue"]["status"] == "failure"
    assert runtime["dependency_queue"]["failure_code"] == "upstream_not_running"
    assert "tmux session is missing" in runtime["dependency_queue"]["error"]


def test_upstream_activity_requires_tmux_pid_and_fresh_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = load_queue_manifest(_write_manifest(tmp_path))
    heartbeat = queue.main_gate.upstream.heartbeat_paths[0]
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.touch()
    monkeypatch.setattr(
        dependency_queue.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"{os.getpid()}\n",
            stderr="",
        ),
    )

    alive = evaluate_upstream_activity(queue)
    assert alive.running is True
    assert alive.process_pid == os.getpid()

    stale_mtime = heartbeat.stat().st_mtime - 301
    os.utime(heartbeat, (stale_mtime, stale_mtime))
    monkeypatch.setattr(dependency_queue.time, "time", lambda: stale_mtime + 301)
    stale = evaluate_upstream_activity(queue)
    assert stale.running is False
    assert "heartbeat is stale" in stale.reason

    monkeypatch.setattr(
        dependency_queue.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="can't find session",
        ),
    )
    missing = evaluate_upstream_activity(queue)
    assert missing.running is False
    assert missing.reason == "can't find session"


def test_waiter_rechecks_exact_gate_after_filesystem_event(tmp_path: Path) -> None:
    queue = load_queue_manifest(_write_manifest(tmp_path))
    gate = queue.main_gate.jobs[0]

    class FakeWatcher:
        waits = 0

        def wait(self, timeout_seconds: float | None = None) -> None:
            self.waits += 1
            _write_ledger(gate.ledger, ["completed", "completed"])
            _write_runtime_record(
                queue.main_gate.runtime_record,
                {gate.runtime_cache_key: "complete"},
            )

        def close(self) -> None:
            return None

    watcher = FakeWatcher()
    alive = UpstreamStatus(
        running=True,
        reason="",
        heartbeat_age_seconds=0,
        seconds_until_stale=300,
    )
    wait_for_main_gate(
        queue,
        watcher_factory=lambda _, __: watcher,
        upstream_checker=lambda _: alive,
    )
    assert watcher.waits == 1


def test_dependency_queue_runs_sequentially_and_resumes_completed_jobs(tmp_path: Path) -> None:
    queue = load_queue_manifest(_write_manifest(tmp_path))
    gate = queue.main_gate.jobs[0]
    _write_ledger(gate.ledger, ["completed", "completed"])
    _write_runtime_record(queue.main_gate.runtime_record, {gate.runtime_cache_key: "complete"})
    calls = []

    def fake_executor(job, *, environment):
        calls.append((job.job_id, dict(environment)))
        _write_ledger(job.output_root / "batch_state.sqlite3", ["completed", "completed"])
        (job.output_root / "batch_summary.json").write_text(
            json.dumps(
                {"completed": 2, "failed": 0, "pending": 0, "running": 0, "total": 2}
            )
            + "\n",
            encoding="utf-8",
        )

    run_dependency_queue(queue, job_executor=fake_executor)
    assert [job_id for job_id, _ in calls] == ["seed1_model1", "seed2_model1"]
    assert all(env["CUDA_VISIBLE_DEVICES"] == "1" for _, env in calls)

    run_dependency_queue(queue, job_executor=fake_executor)
    assert len(calls) == 2
    runtime = json.loads(queue.runtime_record.read_text(encoding="utf-8"))
    assert runtime["class_code"] == {"A": "Conflict", "C": "Aligned"}
    assert [job["status"] for job in runtime["dependency_queue"]["jobs"]] == [
        "complete",
        "complete",
    ]
