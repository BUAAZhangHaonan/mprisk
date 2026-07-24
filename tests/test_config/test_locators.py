from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mprisk.config.locators import (
    LocatorError,
    LocatorResolver,
    is_locator,
    resolve_locator,
)


def _overlay(tmp_path: Path, roots: dict[str, dict[str, str]]) -> Path:
    path = tmp_path / "local_paths.yaml"
    path.write_text(
        yaml.safe_dump(
            {"schema_version": "mprisk_local_paths_v1", "roots": roots},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_resolves_file_and_records_provenance(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "config.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    overlay = _overlay(tmp_path, {"repo": {"project": str(repo)}})

    result = resolve_locator(
        "repo://project/config.yaml",
        local_paths=overlay,
        expected_type="file",
    )

    assert result.resolved_path == config
    assert result.provenance() == {
        "locator": "repo://project/config.yaml",
        "overlay_sha256": result.overlay_sha256,
        "resolved_path": str(config),
    }
    assert len(result.overlay_sha256) == 64


def test_resolves_directory_root(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    overlay = _overlay(tmp_path, {"model": {"qwen": str(model)}})
    result = LocatorResolver(overlay).resolve("model://qwen", expected_type="dir")
    assert result.resolved_path == model


@pytest.mark.parametrize(
    "locator",
    [
        "/home/team/model",
        "relative/model",
        "unknown://model",
        "repo://project/../secret",
        "repo://project/%2e%2e/secret",
        r"repo://project\\secret",
        "repo://project/file?x=1",
        "repo://project/file#x",
        "repo:///missing-key",
        "repo://bad$key/file",
    ],
)
def test_rejects_invalid_locator_forms(tmp_path: Path, locator: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    overlay = _overlay(tmp_path, {"repo": {"project": str(repo)}})
    with pytest.raises(LocatorError):
        LocatorResolver(overlay).resolve(locator)


def test_rejects_archive_identity_resolution(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path, {"repo": {"project": str(tmp_path)}})
    with pytest.raises(LocatorError, match="immutable identity"):
        LocatorResolver(overlay).resolve("archive://delivery/sample.jsonl")
    assert is_locator("archive://delivery/sample.jsonl")


def test_rejects_unknown_key(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path, {"repo": {"project": str(tmp_path)}})
    with pytest.raises(LocatorError, match="Unknown locator key"):
        LocatorResolver(overlay).resolve("repo://other")


def test_rejects_missing_and_wrong_type_targets(tmp_path: Path) -> None:
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    missing = tmp_path / "missing"
    overlay = _overlay(
        tmp_path,
        {"external": {"file": str(file_path), "missing": str(missing)}},
    )
    resolver = LocatorResolver(overlay)
    with pytest.raises(LocatorError, match="does not exist"):
        resolver.resolve("external://missing")
    with pytest.raises(LocatorError, match="not a directory"):
        resolver.resolve("external://file", expected_type="dir")


@pytest.mark.parametrize(
    "payload,match",
    [
        ({}, "schema_version"),
        (
            {"schema_version": "mprisk_local_paths_v1", "roots": []},
            "roots mapping",
        ),
        (
            {
                "schema_version": "mprisk_local_paths_v1",
                "roots": {"unknown": {}},
            },
            "Unknown overlay scheme",
        ),
        (
            {
                "schema_version": "mprisk_local_paths_v1",
                "roots": {"repo": {"project": "relative"}},
            },
            "must be absolute",
        ),
    ],
)
def test_rejects_invalid_overlay(
    tmp_path: Path,
    payload: dict[str, object],
    match: str,
) -> None:
    path = tmp_path / "local_paths.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(LocatorError, match=match):
        LocatorResolver(path)


def test_overlay_must_be_explicit_existing_file(tmp_path: Path) -> None:
    with pytest.raises(LocatorError, match="not a file"):
        LocatorResolver(tmp_path / "absent.yaml")
