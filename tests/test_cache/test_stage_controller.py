from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mprisk.cache.stage_controller as controller
from mprisk.cache.stage_controller import (
    build_stage_lane_command,
    launch_target_lanes,
    plan_target_lane_launches,
    prospective_target_job_ids,
    scoped_launch_blockers,
    summarize_stage,
    validate_active_lanes,
    wait_for_lane_startup,
)


def _record(stage: str, index: int, status: str, *, lane: int = 0) -> dict:
    record = {
        "job_id": f"{stage}:model{index}",
        "domain": stage,
        "gpu_lane": lane,
        "status": status,
        "expected_tasks": 24,
        "asset_signature": {"passed": True},
    }
    if status == "complete":
        record["ledger"] = {"status": "complete", "missing": 0}
    elif status == "ready":
        record["ledger"] = {"status": "incomplete", "missing": 10}
    return record


def _audit(source: list[dict], target: list[dict]) -> dict:
    return {
        "schema": "mprisk_complete_cache_matrix_audit_v1",
        "job_records": source + target,
    }


def _lane_status(
    lane: int,
    *,
    session_exists: bool = False,
    lock_exists: bool = False,
    active: bool = False,
) -> dict:
    return {
        "lane": lane,
        "session_exists": session_exists,
        "lock_exists": lock_exists,
        "active": active,
    }


def test_source_requires_exactly_fifteen_complete_and_one_accepted() -> None:
    source = [_record("source", index, "complete") for index in range(15)]
    source.append(_record("source", 15, "accepted_bundle"))
    target = [_record("target", index, "ready") for index in range(16)]

    summary = summarize_stage(
        _audit(source, target),
        stage="source",
        expected_jobs=16,
        expected_accepted=1,
    )

    assert summary["strict_complete"] is True
    assert summary["missing_tasks"] == 0
    assert summary["status_counts"] == {
        "accepted_bundle": 1,
        "complete": 15,
    }


def test_signature_mismatch_is_a_terminal_blocker() -> None:
    source = [_record("source", index, "complete") for index in range(16)]
    source[3]["status"] = "blocked_cache_asset_signature"
    source[3]["cache_asset_signature"] = {
        "passed": False,
        "reason": "mismatch",
    }

    summary = summarize_stage(
        _audit(source, []),
        stage="source",
        expected_jobs=16,
        expected_accepted=0,
    )

    assert summary["strict_complete"] is False
    assert summary["signature_mismatches"] == ["source:model3"]
    assert summary["blocked"] == ["source:model3=blocked_cache_asset_signature"]


def test_scoped_launch_ignores_unrelated_completed_signature_mismatch() -> None:
    source = [_record("source", index, "complete") for index in range(16)]
    source[3]["status"] = "blocked_cache_asset_signature"
    source[3]["cache_asset_signature"] = {"passed": False}
    target = [
        _record("target", index, "ready", lane=index % 2) for index in range(16)
    ]
    audit = _audit(source, target)
    audit["ready_to_launch"] = False

    prospective = prospective_target_job_ids(target, (0,))

    assert prospective == ("target:model0",)
    assert scoped_launch_blockers(audit, prospective) == []


def test_scoped_launch_blocks_current_lane_job_mismatch() -> None:
    target = [
        _record("target", index, "ready", lane=index % 2) for index in range(16)
    ]
    target[0]["status"] = "blocked_cache_asset_signature"
    target[0]["cache_asset_signature"] = {"passed": False}
    audit = _audit([], target)

    prospective = prospective_target_job_ids(target, (0,))

    assert scoped_launch_blockers(audit, prospective) == [
        "target:model0=blocked_cache_asset_signature"
    ]


def test_incomplete_stage_requires_live_supervisor(monkeypatch) -> None:
    summary = {
        "stage": "source",
        "records": [_record("source", 0, "ready", lane=1)],
    }
    monkeypatch.setattr(
        controller,
        "lane_supervisor_status",
        lambda *args, **kwargs: {
            "lane": 1,
            "session_exists": False,
            "lock_exists": True,
            "lock_pid_alive": False,
            "active": False,
        },
    )

    with pytest.raises(RuntimeError, match="supervisor is inactive"):
        validate_active_lanes(SimpleNamespace(), summary, {0: "source0", 1: "source1"})


def test_target_launch_is_exactly_two_waiting_target_lanes(tmp_path: Path, monkeypatch) -> None:
    config = SimpleNamespace(
        repo_root=tmp_path,
        source_path=tmp_path / "matrix.yaml",
        lock_path=tmp_path / "matrix.lock",
        runtime_record=tmp_path / "matrix.json",
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "list-panes" in command:
            return SimpleNamespace(returncode=0, stdout="1234\n")
        return SimpleNamespace(returncode=1 if "has-session" in command else 0)

    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    monkeypatch.setattr(
        controller,
        "wait_for_lane_startup",
        lambda config, *, stage, lane, session, manager_log, timeout_seconds: {
            "stage": stage,
            "lane": lane,
            "session": session,
            "active": True,
        },
    )

    launched = launch_target_lanes(
        config,
        source_sessions={0: "source0", 1: "source1"},
        sessions={0: "target0", 1: "target1"},
        manager_logs={
            0: tmp_path / "target0.log",
            1: tmp_path / "target1.log",
        },
        python=Path("/env/bin/python"),
    )

    assert [item["lane"] for item in launched] == [0, 1]
    new_sessions = [call for call in calls if "new-session" in call]
    assert len(new_sessions) == 2
    for lane, command in enumerate(new_sessions):
        shell_command = command[-1]
        assert f"PYTHONPATH={tmp_path / 'src'}" in shell_command
        assert "--stage target" in shell_command
        assert f"--lane {lane}" in shell_command
        assert "--wait-for-gpu" in shell_command
        assert "misread" not in shell_command.lower()
        assert "api" not in shell_command.lower()


def test_stage_lane_command_sets_absolute_pythonpath_for_both_stages(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        repo_root=tmp_path.resolve(),
        source_path=(tmp_path / "matrix.yaml").resolve(),
    )

    for stage in ("source", "target"):
        command = build_stage_lane_command(
            config,
            python=Path("/env/bin/python"),
            stage=stage,
            lane=0,
        )
        assert command[:2] == [
            "env",
            f"PYTHONPATH={tmp_path.resolve() / 'src'}",
        ]
        assert command[2] == "/env/bin/python"
        assert command[command.index("--stage") + 1] == stage


def test_lane_supervisor_requires_session_and_live_scoped_lock(tmp_path: Path, monkeypatch) -> None:
    config = SimpleNamespace(
        lock_path=tmp_path / "matrix.lock",
        runtime_record=tmp_path / "matrix.json",
    )
    lock_path, _ = controller._scoped_execution_paths(config, stage="target", lane=0)
    monkeypatch.setattr(controller, "tmux_session_exists", lambda session: True)

    assert (
        controller.lane_supervisor_status(config, stage="target", lane=0, session="target0")[
            "active"
        ]
        is False
    )
    lock_path.write_text(f"{controller.os.getpid()}\n", encoding="utf-8")
    assert (
        controller.lane_supervisor_status(config, stage="target", lane=0, session="target0")[
            "active"
        ]
        is True
    )
    lock_path.write_text("999999999\n", encoding="utf-8")
    assert (
        controller.lane_supervisor_status(config, stage="target", lane=0, session="target0")[
            "active"
        ]
        is False
    )


def test_lane_startup_waits_for_live_lock(monkeypatch, tmp_path: Path) -> None:
    statuses = iter(
        [
            {"session_exists": True, "active": False},
            {"session_exists": True, "active": True},
        ]
    )
    clock = {"value": 0.0}
    monkeypatch.setattr(
        controller,
        "lane_supervisor_status",
        lambda *args, **kwargs: next(statuses),
    )

    status = wait_for_lane_startup(
        SimpleNamespace(),
        stage="target",
        lane=0,
        session="target0",
        manager_log=tmp_path / "target0.log",
        timeout_seconds=10,
        sleep_fn=lambda seconds: clock.__setitem__("value", clock["value"] + seconds),
        monotonic_fn=lambda: clock["value"],
    )

    assert status["active"] is True
    assert clock["value"] > 0


def test_lane_startup_reports_manager_error_when_pane_exits(monkeypatch, tmp_path: Path) -> None:
    manager_log = tmp_path / "target0.log"
    manager_log.write_text("ModuleNotFoundError: No module named 'mprisk'\n", encoding="utf-8")
    monkeypatch.setattr(
        controller,
        "lane_supervisor_status",
        lambda *args, **kwargs: {"session_exists": False, "active": False},
    )

    with pytest.raises(RuntimeError, match="ModuleNotFoundError"):
        wait_for_lane_startup(
            SimpleNamespace(),
            stage="target",
            lane=0,
            session="target0",
            manager_log=manager_log,
            timeout_seconds=10,
        )


def test_lane_startup_timeout_stops_unclaimed_session(monkeypatch, tmp_path: Path) -> None:
    clock = {"value": 0.0}
    calls: list[list[str]] = []
    monkeypatch.setattr(
        controller,
        "lane_supervisor_status",
        lambda *args, **kwargs: {"session_exists": True, "active": False},
    )
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )

    with pytest.raises(TimeoutError, match="live lock"):
        wait_for_lane_startup(
            SimpleNamespace(),
            stage="target",
            lane=0,
            session="target0",
            manager_log=tmp_path / "target0.log",
            timeout_seconds=1,
            poll_seconds=0.6,
            sleep_fn=lambda seconds: clock.__setitem__("value", clock["value"] + seconds),
            monotonic_fn=lambda: clock["value"],
        )

    assert calls == [["tmux", "kill-session", "-t", "target0"]]


def test_target_launcher_rechecks_source_lane_ownership(tmp_path: Path, monkeypatch) -> None:
    config = SimpleNamespace(
        repo_root=tmp_path,
        source_path=tmp_path / "matrix.yaml",
        lock_path=tmp_path / "matrix.lock",
        runtime_record=tmp_path / "matrix.json",
    )
    monkeypatch.setattr(
        controller,
        "lane_supervisor_status",
        lambda *args, **kwargs: _lane_status(0, session_exists=True, lock_exists=True, active=True),
    )

    with pytest.raises(RuntimeError, match="Source lane 0 still owns GPU 0"):
        launch_target_lanes(
            config,
            source_sessions={0: "source0", 1: "source1"},
            sessions={0: "target0", 1: "target1"},
            manager_logs={0: tmp_path / "target0.log"},
            python=Path("/env/bin/python"),
            lanes=(0,),
        )


def test_parallel_plan_launches_only_idle_gpu_lane() -> None:
    plan = plan_target_lane_launches(
        source_statuses=[
            _lane_status(0, session_exists=True, lock_exists=True, active=True),
            _lane_status(1),
        ],
        target_statuses=[_lane_status(0), _lane_status(1)],
        pending_target_lanes={0, 1},
        allow_launch=True,
    )

    assert plan == {
        "source_owned": [0],
        "target_active": [],
        "target_waiting": [0],
        "target_launchable": [1],
    }


def test_parallel_plan_adopts_existing_target_supervisor_without_duplicate() -> None:
    plan = plan_target_lane_launches(
        source_statuses=[
            _lane_status(0, session_exists=True, lock_exists=True, active=True),
            _lane_status(1),
        ],
        target_statuses=[
            _lane_status(0),
            _lane_status(1, session_exists=True, lock_exists=True, active=True),
        ],
        pending_target_lanes={0, 1},
        allow_launch=True,
    )

    assert plan["target_active"] == [1]
    assert plan["target_waiting"] == [0]
    assert plan["target_launchable"] == []


def test_parallel_plan_rejects_cross_domain_lane_collision() -> None:
    with pytest.raises(RuntimeError, match="same GPU lanes: 0"):
        plan_target_lane_launches(
            source_statuses=[
                _lane_status(0, session_exists=True, lock_exists=True, active=True),
                _lane_status(1),
            ],
            target_statuses=[
                _lane_status(0, session_exists=True, lock_exists=True, active=True),
                _lane_status(1),
            ],
            pending_target_lanes={0},
            allow_launch=True,
        )


def test_parallel_plan_rejects_stale_target_ownership_marker() -> None:
    with pytest.raises(RuntimeError, match="markers exist"):
        plan_target_lane_launches(
            source_statuses=[_lane_status(0), _lane_status(1)],
            target_statuses=[
                _lane_status(0, lock_exists=True, active=False),
                _lane_status(1),
            ],
            pending_target_lanes={0},
            allow_launch=True,
        )


def test_controller_runs_parallel_dag_and_completes_only_after_both_domains(
    tmp_path: Path, monkeypatch
) -> None:
    phase = 0
    active_target_lanes: set[int] = set()
    launched_lanes: list[int] = []
    source_running = [_record("source", 0, "ready", lane=0)] + [
        _record("source", index, "complete", lane=index % 2) for index in range(1, 16)
    ]
    source_complete = [_record("source", index, "complete", lane=index % 2) for index in range(16)]
    target_ready = [_record("target", index, "ready", lane=index % 2) for index in range(16)]
    target_complete = [_record("target", index, "complete", lane=index % 2) for index in range(16)]

    def summary(records: list[dict], *, stage: str) -> dict:
        return summarize_stage(
            _audit(
                records if stage == "source" else source_complete,
                records if stage == "target" else target_ready,
            ),
            stage=stage,
            expected_jobs=16,
            expected_accepted=0,
        )

    def fake_progress(config, stage: str) -> dict:
        if stage == "source":
            records = source_running if phase == 0 else source_complete
        else:
            records = target_complete if phase >= 2 else target_ready
        return summary(records, stage=stage)

    def fake_stage_statuses(config, *, stage: str, sessions) -> list[dict]:
        if stage == "source":
            return [
                _lane_status(
                    0,
                    session_exists=phase == 0,
                    lock_exists=phase == 0,
                    active=phase == 0,
                ),
                _lane_status(1),
            ]
        return [
            _lane_status(
                lane,
                session_exists=lane in active_target_lanes,
                lock_exists=lane in active_target_lanes,
                active=lane in active_target_lanes,
            )
            for lane in (0, 1)
        ]

    def fake_launch(config, *, lanes, **kwargs):
        active_target_lanes.update(lanes)
        launched_lanes.extend(lanes)
        return [{"lane": lane, "session": f"target{lane}"} for lane in lanes]

    def fake_sleep(seconds: float) -> None:
        nonlocal phase
        phase += 1
        if phase >= 2:
            active_target_lanes.clear()

    def fake_audit(config) -> dict:
        audit = _audit(
            source_complete,
            target_complete if phase >= 2 else target_ready,
        )
        audit["ready_to_launch"] = True
        return audit

    monkeypatch.setattr(controller, "read_stage_progress", fake_progress)
    monkeypatch.setattr(
        controller,
        "validate_active_lanes",
        lambda config, stage_summary, sessions: fake_stage_statuses(
            config, stage="source", sessions=sessions
        ),
    )
    monkeypatch.setattr(controller, "stage_lane_statuses", fake_stage_statuses)
    monkeypatch.setattr(
        controller,
        "stage_is_finalized",
        lambda config, *, stage, sessions: (
            phase >= (1 if stage == "source" else 2),
            fake_stage_statuses(config, stage=stage, sessions=sessions),
        ),
    )
    monkeypatch.setattr(controller, "launch_target_lanes", fake_launch)
    monkeypatch.setattr(controller, "_git_head", lambda path: "head")
    config = SimpleNamespace(
        source_path=tmp_path / "matrix.yaml",
        repo_root=tmp_path,
        models=(),
        allow_parallel_domain_extraction=True,
    )
    stage_controller = controller.StageController(
        config,
        paths=controller.build_controller_paths(tmp_path / "status"),
        poll_interval_seconds=1,
        source_sessions={0: "source0", 1: "source1"},
        target_sessions={0: "target0", 1: "target1"},
        audit_fn=fake_audit,
        launch_audit_fn=lambda config, job_ids: fake_audit(config),
        sleep_fn=fake_sleep,
    )

    assert stage_controller.run() == 0
    assert launched_lanes == [1, 0]
    assert phase == 2
    status = json.loads((tmp_path / "status" / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["source"]["strict_complete"] is True
    assert status["target"]["strict_complete"] is True
    assert (tmp_path / "status" / "SOURCE_COMPLETE_AUDIT.json").is_file()
    assert (tmp_path / "status" / "EXTRACTION_LAUNCH_AUDIT.json").is_file()
    assert (tmp_path / "status" / "FINAL_CACHE_AUDIT.json").is_file()


def test_controller_stops_after_single_lane_startup_failure(tmp_path: Path, monkeypatch) -> None:
    source = [_record("source", index, "complete", lane=index % 2) for index in range(16)]
    target = [_record("target", index, "ready", lane=0) for index in range(16)]
    audit = _audit(source, target)
    audit["ready_to_launch"] = True
    source_summary = summarize_stage(
        audit,
        stage="source",
        expected_jobs=16,
        expected_accepted=0,
    )
    target_summary = summarize_stage(
        audit,
        stage="target",
        expected_jobs=16,
        expected_accepted=0,
    )
    launch_calls = 0
    audit_calls = 0

    def fail_launch(*args, **kwargs):
        nonlocal launch_calls
        launch_calls += 1
        raise RuntimeError("target lane 0 startup failed before acquiring its live lock")

    def fake_audit(config):
        nonlocal audit_calls
        audit_calls += 1
        return audit

    monkeypatch.setattr(
        controller,
        "read_stage_progress",
        lambda config, stage: (source_summary if stage == "source" else target_summary),
    )
    monkeypatch.setattr(
        controller,
        "stage_is_finalized",
        lambda *args, **kwargs: (True, []),
    )
    monkeypatch.setattr(
        controller,
        "stage_lane_statuses",
        lambda config, *, stage, sessions: [
            _lane_status(0),
            _lane_status(1),
        ],
    )
    monkeypatch.setattr(controller, "launch_target_lanes", fail_launch)
    monkeypatch.setattr(controller, "_git_head", lambda path: "head")
    config = SimpleNamespace(
        source_path=tmp_path / "matrix.yaml",
        repo_root=tmp_path,
        models=(),
        allow_parallel_domain_extraction=True,
    )
    stage_controller = controller.StageController(
        config,
        paths=controller.build_controller_paths(tmp_path / "status"),
        poll_interval_seconds=1,
        source_sessions={0: "source0", 1: "source1"},
        target_sessions={0: "target0", 1: "target1"},
        audit_fn=fake_audit,
        launch_audit_fn=lambda config, job_ids: fake_audit(config),
        sleep_fn=lambda _: pytest.fail("controller must not retry startup"),
    )

    assert stage_controller.run() == 1
    assert launch_calls == 1
    assert audit_calls == 1
    status = json.loads((tmp_path / "status" / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "startup failed" in status["error"]
    assert status["target_launches"] == []
    assert not (tmp_path / "status" / "controller.lock").exists()


def test_controller_fails_closed_without_launching_target(tmp_path: Path, monkeypatch) -> None:
    source = [_record("source", index, "complete") for index in range(15)]
    source.append(_record("source", 15, "failed"))
    target = [_record("target", index, "ready") for index in range(16)]
    audit = _audit(source, target)
    audit["ready_to_launch"] = False
    config = SimpleNamespace(
        source_path=tmp_path / "matrix.yaml",
        repo_root=tmp_path,
        models=(),
        allow_parallel_domain_extraction=True,
    )
    launched = False

    def forbidden_launch(*args, **kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr(controller, "launch_target_lanes", forbidden_launch)
    monkeypatch.setattr(controller, "_git_head", lambda path: "head")
    source_summary = summarize_stage(
        audit,
        stage="source",
        expected_jobs=16,
        expected_accepted=0,
    )
    target_summary = summarize_stage(
        audit,
        stage="target",
        expected_jobs=16,
        expected_accepted=0,
    )
    monkeypatch.setattr(
        controller,
        "read_stage_progress",
        lambda config, stage: source_summary if stage == "source" else target_summary,
    )
    stage_controller = controller.StageController(
        config,
        paths=controller.build_controller_paths(tmp_path / "status"),
        poll_interval_seconds=1,
        source_sessions={0: "source0", 1: "source1"},
        target_sessions={0: "target0", 1: "target1"},
        audit_fn=lambda _: pytest.fail("full audit must not run after ledger failure"),
        sleep_fn=lambda _: None,
    )

    assert stage_controller.run() == 1
    assert launched is False
    status = json.loads((tmp_path / "status" / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "Source audit failed" in status["error"]
    assert not (tmp_path / "status" / "SOURCE_COMPLETE_AUDIT.json").exists()


def test_controller_final_audit_remains_global_and_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    source_records = [
        _record("source", index, "complete", lane=index % 2) for index in range(16)
    ]
    target_records = [
        _record("target", index, "complete", lane=index % 2) for index in range(16)
    ]
    progress_audit = _audit(source_records, target_records)
    source_summary = summarize_stage(
        progress_audit,
        stage="source",
        expected_jobs=16,
        expected_accepted=0,
    )
    target_summary = summarize_stage(
        progress_audit,
        stage="target",
        expected_jobs=16,
        expected_accepted=0,
    )
    strict_audit = _audit(
        [dict(record) for record in source_records],
        [dict(record) for record in target_records],
    )
    strict_audit["job_records"][3]["status"] = "blocked_cache_asset_signature"
    strict_audit["job_records"][3]["cache_asset_signature"] = {"passed": False}
    strict_audit["ready_to_launch"] = False
    audit_calls = 0

    def fake_audit(config):
        nonlocal audit_calls
        audit_calls += 1
        return strict_audit

    monkeypatch.setattr(
        controller,
        "read_stage_progress",
        lambda config, stage: source_summary if stage == "source" else target_summary,
    )
    monkeypatch.setattr(
        controller,
        "stage_is_finalized",
        lambda *args, **kwargs: (True, []),
    )
    monkeypatch.setattr(
        controller,
        "stage_lane_statuses",
        lambda config, *, stage, sessions: [_lane_status(0), _lane_status(1)],
    )
    monkeypatch.setattr(controller, "_git_head", lambda path: "head")
    config = SimpleNamespace(
        source_path=tmp_path / "matrix.yaml",
        repo_root=tmp_path,
        models=(),
        allow_parallel_domain_extraction=True,
    )
    stage_controller = controller.StageController(
        config,
        paths=controller.build_controller_paths(tmp_path / "status"),
        poll_interval_seconds=1,
        source_sessions={0: "source0", 1: "source1"},
        target_sessions={0: "target0", 1: "target1"},
        audit_fn=fake_audit,
        launch_audit_fn=lambda config, job_ids: pytest.fail(
            "no target launch audit is expected"
        ),
        sleep_fn=lambda _: pytest.fail("controller must fail in this cycle"),
    )

    assert stage_controller.run() == 1
    assert audit_calls == 1
    status = json.loads((tmp_path / "status" / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "Final ledger candidate failed the full strict audit" in status["error"]
    assert not (tmp_path / "status" / "FINAL_CACHE_AUDIT.json").exists()
