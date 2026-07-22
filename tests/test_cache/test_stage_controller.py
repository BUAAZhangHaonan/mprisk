from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mprisk.cache.stage_controller as controller
from mprisk.cache.stage_controller import (
    launch_target_lanes,
    summarize_stage,
    validate_active_lanes,
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
    assert summary["blocked"] == [
        "source:model3=blocked_cache_asset_signature"
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
        validate_active_lanes(
            SimpleNamespace(), summary, {0: "source0", 1: "source1"}
        )


def test_target_launch_is_exactly_two_waiting_target_lanes(
    tmp_path: Path, monkeypatch
) -> None:
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

    launched = launch_target_lanes(
        config,
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
        assert "--stage target" in shell_command
        assert f"--lane {lane}" in shell_command
        assert "--wait-for-gpu" in shell_command
        assert "misread" not in shell_command.lower()
        assert "api" not in shell_command.lower()


def test_controller_fails_closed_without_launching_target(
    tmp_path: Path, monkeypatch
) -> None:
    source = [_record("source", index, "complete") for index in range(15)]
    source.append(_record("source", 15, "failed"))
    target = [_record("target", index, "ready") for index in range(16)]
    audit = _audit(source, target)
    audit["ready_to_launch"] = False
    config = SimpleNamespace(
        source_path=tmp_path / "matrix.yaml",
        repo_root=tmp_path,
        models=(),
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
    status = json.loads(
        (tmp_path / "status" / "status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert "Source audit failed" in status["error"]
    assert not (tmp_path / "status" / "SOURCE_COMPLETE_AUDIT.json").exists()
