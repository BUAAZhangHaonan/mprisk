from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from mprisk.diagnostic_affect.watcher import watch_description_generation


class _FakeProcess:
    def __init__(self, command, **kwargs) -> None:
        self.command = command
        self.kwargs = kwargs
        self.pid = 4321
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "output"
    config = tmp_path / "diagnostic.yaml"
    config.write_text(f"output_root: {output}\n", encoding="utf-8")
    return config, output


def _write_ledger(output: Path, statuses: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output / "batch_state.sqlite3") as connection:
        connection.execute("DROP TABLE IF EXISTS tasks")
        connection.execute("CREATE TABLE tasks(status TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO tasks(status) VALUES(?)",
            [(status,) for status in statuses],
        )


def test_watcher_resumes_full_plan_and_requires_strict_completion(tmp_path: Path) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["completed", "pending"])
    process_box: list[_FakeProcess] = []
    clock = _Clock()

    def factory(command, **kwargs):
        process = _FakeProcess(command, **kwargs)
        process_box.append(process)
        return process

    def advance(_seconds: float) -> None:
        _write_ledger(output, ["completed", "completed"])
        process_box[0].returncode = 0
        clock.advance(1.0)

    returncode = watch_description_generation(
        config_path=config,
        python_executable=Path("/env/bin/python"),
        stall_timeout_seconds=60,
        poll_interval_seconds=1,
        terminate_grace_seconds=1,
        popen_factory=factory,
        monotonic_fn=clock.monotonic,
        sleep_fn=advance,
    )

    assert returncode == 0
    assert "--sample-id" not in process_box[0].command
    assert process_box[0].command[-2:] == ["--config", str(config.resolve())]
    status = json.loads((output / "watcher_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert status["ledger"]["completed"] == 2


def test_watcher_isolates_child_and_attests_exact_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["pending"])
    process_box: list[_FakeProcess] = []
    clock = _Clock()
    contract = {
        "environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "/repository/src",
        },
        "module_files": {"mprisk": "/repository/src/mprisk/__init__.py"},
        "python_executable": "/env/bin/python",
        "python_prefix": "/env",
        "python_version": "3.11.11",
        "user_site_enabled": False,
        "torch_cuda_version": "12.1",
        "package_versions": {"transformers": "4.43.0"},
    }

    def factory(command, **kwargs):
        process = _FakeProcess(command, **kwargs)
        process_box.append(process)
        return process

    def runtime_probe(*args, **kwargs):
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert kwargs["env"]["PYTHONPATH"] == "/repository/src"
        assert json.loads(args[0][-2]) == ["PYTHONNOUSERSITE", "PYTHONPATH"]
        assert json.loads(args[0][-1]) == ["mprisk"]
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(contract),
            stderr="",
        )

    def advance(_seconds: float) -> None:
        _write_ledger(output, ["completed"])
        process_box[0].returncode = 0
        clock.advance(1.0)

    monkeypatch.delenv("PYTHONPATH", raising=False)
    returncode = watch_description_generation(
        config_path=config,
        python_executable=Path("/env/bin/python"),
        python_environment={
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "/repository/src",
        },
        runtime_contract=contract,
        retry_failed=True,
        stall_timeout_seconds=60,
        poll_interval_seconds=1,
        terminate_grace_seconds=1,
        popen_factory=factory,
        run_factory=runtime_probe,
        monotonic_fn=clock.monotonic,
        sleep_fn=advance,
    )

    assert returncode == 0
    assert process_box[0].kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    assert process_box[0].kwargs["env"]["PYTHONPATH"] == "/repository/src"
    assert process_box[0].command[-1] == "--retry-failed"
    receipt = json.loads(
        (output / "runtime_contract_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "PASS"
    assert receipt["observed"] == contract


def test_watcher_rejects_runtime_contract_mismatch_before_launch(
    tmp_path: Path,
) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["pending"])
    contract = {
        "python_executable": "/env/bin/python",
        "python_prefix": "/env",
        "python_version": "3.11.11",
        "user_site_enabled": False,
        "torch_cuda_version": "12.1",
        "package_versions": {"transformers": "4.43.0"},
    }

    with pytest.raises(RuntimeError, match="runtime contract mismatch"):
        watch_description_generation(
            config_path=config,
            python_executable=Path("/env/bin/python"),
            python_environment={"PYTHONNOUSERSITE": "1"},
            runtime_contract=contract,
            stall_timeout_seconds=60,
            poll_interval_seconds=1,
            terminate_grace_seconds=1,
            popen_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("child must not launch")
            ),
            run_factory=lambda *args, **kwargs: subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        **contract,
                        "package_versions": {"transformers": "5.5.3"},
                    }
                ),
                stderr="",
            ),
        )


def test_watcher_rejects_non_boolean_user_site_contract_before_probe(
    tmp_path: Path,
) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["pending"])
    contract = {
        "python_executable": "/env/bin/python",
        "python_prefix": "/env",
        "python_version": "3.11.11",
        "user_site_enabled": "false",
        "torch_cuda_version": "12.1",
        "package_versions": {"transformers": "4.43.0"},
    }

    with pytest.raises(ValueError, match="user_site_enabled must be boolean"):
        watch_description_generation(
            config_path=config,
            python_executable=Path("/env/bin/python"),
            python_environment={"PYTHONNOUSERSITE": "1"},
            runtime_contract=contract,
            stall_timeout_seconds=60,
            poll_interval_seconds=1,
            terminate_grace_seconds=1,
            popen_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("child must not launch")
            ),
            run_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("runtime probe must not launch")
            ),
        )


def test_watcher_waits_for_atomic_retry_reset_before_enforcing_failed_gate(
    tmp_path: Path,
) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["failed", "running", "pending"])
    process_box: list[_FakeProcess] = []
    clock = _Clock()
    sleeps = 0

    def factory(command, **kwargs):
        process = _FakeProcess(command, **kwargs)
        process_box.append(process)
        return process

    def advance(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            _write_ledger(output, ["pending", "running", "pending"])
        else:
            _write_ledger(output, ["completed", "completed", "completed"])
            process_box[0].returncode = 0
        clock.advance(1.0)

    returncode = watch_description_generation(
        config_path=config,
        python_executable=Path("/env/bin/python"),
        retry_failed=True,
        stall_timeout_seconds=60,
        poll_interval_seconds=1,
        terminate_grace_seconds=1,
        popen_factory=factory,
        monotonic_fn=clock.monotonic,
        sleep_fn=advance,
    )

    assert returncode == 0
    assert process_box[0].terminated is False
    assert process_box[0].command[-1] == "--retry-failed"
    status = json.loads((output / "watcher_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"


def test_watcher_rejects_failed_task_after_retry_reset_transition(
    tmp_path: Path,
) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["failed", "pending"])
    process = _FakeProcess([])
    clock = _Clock()

    def advance(_seconds: float) -> None:
        _write_ledger(output, ["failed", "pending", "pending"])
        clock.advance(1.0)

    returncode = watch_description_generation(
        config_path=config,
        python_executable=Path("/env/bin/python"),
        retry_failed=True,
        stall_timeout_seconds=60,
        poll_interval_seconds=1,
        terminate_grace_seconds=1,
        popen_factory=lambda *_args, **_kwargs: process,
        monotonic_fn=clock.monotonic,
        sleep_fn=advance,
    )

    assert returncode == 1
    assert process.terminated is True
    status = json.loads((output / "watcher_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["detail"] == "invalid failed-task retry reset transition"


def test_watcher_fails_when_retry_child_exits_before_atomic_reset(
    tmp_path: Path,
) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["failed", "pending"])
    process = _FakeProcess([])
    process.returncode = 9

    returncode = watch_description_generation(
        config_path=config,
        python_executable=Path("/env/bin/python"),
        retry_failed=True,
        stall_timeout_seconds=60,
        poll_interval_seconds=1,
        terminate_grace_seconds=1,
        popen_factory=lambda *_args, **_kwargs: process,
    )

    assert returncode == 1
    status = json.loads((output / "watcher_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["child_returncode"] == 9


def test_watcher_times_out_when_atomic_retry_reset_never_occurs(
    tmp_path: Path,
) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["failed", "pending"])
    process = _FakeProcess([])
    clock = _Clock()

    returncode = watch_description_generation(
        config_path=config,
        python_executable=Path("/env/bin/python"),
        retry_failed=True,
        stall_timeout_seconds=1,
        poll_interval_seconds=1,
        terminate_grace_seconds=1,
        popen_factory=lambda *_args, **_kwargs: process,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.advance,
    )

    assert returncode == 1
    assert process.terminated is True
    status = json.loads((output / "watcher_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "timed_out"


def test_watcher_propagates_abnormal_child_exit(tmp_path: Path) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["pending"])
    process = _FakeProcess([])
    process.returncode = 7

    returncode = watch_description_generation(
        config_path=config,
        python_executable=Path("/env/bin/python"),
        stall_timeout_seconds=60,
        poll_interval_seconds=1,
        terminate_grace_seconds=1,
        popen_factory=lambda *args, **kwargs: process,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda _: None,
    )

    assert returncode == 1
    status = json.loads((output / "watcher_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["child_returncode"] == 7


def test_watcher_times_out_live_child_without_ledger_heartbeat(tmp_path: Path) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["pending"])
    process = _FakeProcess([])
    clock = _Clock()

    returncode = watch_description_generation(
        config_path=config,
        python_executable=Path("/env/bin/python"),
        stall_timeout_seconds=5,
        poll_interval_seconds=3,
        terminate_grace_seconds=1,
        popen_factory=lambda *args, **kwargs: process,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.advance,
    )

    assert returncode == 1
    assert process.terminated is True
    status = json.loads((output / "watcher_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "timed_out"
    assert "no ledger heartbeat" in status["detail"]


def test_watcher_does_not_launch_when_ledger_is_already_complete(tmp_path: Path) -> None:
    config, output = _write_config(tmp_path)
    _write_ledger(output, ["completed", "completed"])

    returncode = watch_description_generation(
        config_path=config,
        python_executable=Path("/env/bin/python"),
        stall_timeout_seconds=5,
        poll_interval_seconds=1,
        terminate_grace_seconds=1,
        popen_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("child must not launch")
        ),
    )

    assert returncode == 0
