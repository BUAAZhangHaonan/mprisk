from __future__ import annotations

import sys

import pytest

from mprisk.cli import scaffold_main


def test_scaffold_dry_run_only_validates_wiring(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["disabled-command", "--dry-run"])

    assert scaffold_main("disabled-command") == 0
    assert capsys.readouterr().out.strip() == (
        "disabled-command: unavailable scaffold; dry-run wiring valid"
    )


def test_scaffold_execution_fails_closed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["disabled-command"])

    with pytest.raises(SystemExit) as exc_info:
        scaffold_main("disabled-command")

    assert exc_info.value.code == 2
    assert "is not implemented" in capsys.readouterr().err
