from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

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
