from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mprisk.data.generated_archive_freeze import (
    _resolve_anchor,
    freeze_generated_round1,
    select_natural_context,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _media(archive_root: Path, relative: str, content: bytes) -> str:
    path = archive_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return relative


def _row(
    *,
    archive_root: Path,
    archive_label: str,
    data_type: str,
    protocol: str,
    recorded_anchor: bool,
    setting: str | None,
    trigger: str | None,
    dialogue: str,
) -> dict[str, object]:
    primary = _media(
        archive_root,
        "video_audio/S0001.mp4",
        f"{archive_label}:primary".encode(),
    )
    silent = None
    if protocol == "VT":
        silent = _media(
            archive_root,
            "silent_video_text/S0001.silent.mp4",
            f"{archive_label}:silent".encode(),
        )
    row: dict[str, object] = {
        "sample_id": "S0001",
        "original_variant_id": f"{archive_label}:original",
        "data_type": data_type,
        "conflict_type": "silent_video_text" if protocol == "VT" else "video_audio",
        "bucket": "accept",
        "gt_emotion": "joy",
        "dialogue_text": dialogue,
        "setting": setting,
        "trigger": trigger,
        "files": {"primary": primary, "silent": silent},
    }
    if recorded_anchor:
        row.update({"archetype_id": 1 if data_type == "A" else 101, "archetype_name": "joy"})
    return row


@pytest.fixture
def frozen_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    row_specs = {
        "accept_a_svt": dict(
            data_type="A",
            protocol="VT",
            recorded_anchor=True,
            setting="recorded setting",
            trigger="natural trigger that must lose to setting",
            dialogue="A SVT dialogue",
        ),
        "accept_a_va": dict(
            data_type="A",
            protocol="VA",
            recorded_anchor=True,
            setting=None,
            trigger="natural trigger",
            dialogue="A VA dialogue",
        ),
        "accept_c_svt": dict(
            data_type="C",
            protocol="VT",
            recorded_anchor=False,
            setting="official SVT setting",
            trigger=None,
            dialogue="C SVT dialogue",
        ),
        "accept_c_va": dict(
            data_type="C",
            protocol="VA",
            recorded_anchor=False,
            setting="official VA setting",
            trigger=None,
            dialogue="C VA dialogue",
        ),
    }
    for archive, row_spec in row_specs.items():
        source_archive = source_root / archive
        source_archive.mkdir()
        row = _row(
            archive_root=source_archive,
            archive_label=archive,
            **row_spec,
        )
        _write_jsonl(source_archive / "index.jsonl", [row])

    archetypes = tmp_path / "glm_client.py"
    archetypes.write_text(
        "ARCHETYPES_GLM = {\n"
        "  1: {'name': 'joy', 'type': 'A', 'gt': 'joy', 'surface': 'smile'},\n"
        "  101: {'name': 'joy', 'type': 'C', 'gt': 'joy', 'surface': None},\n"
        "}\n",
        encoding="utf-8",
    )
    c_templates = tmp_path / "c_type_batch.py"
    c_templates.write_text(
        "EMOTION_VARIANTS = [\n"
        "  {'emotion': 'joy', 'v': 1, 'setting': 'official SVT setting', "
        "'dialogue': 'C SVT dialogue'},\n"
        "  {'emotion': 'joy', 'v': 2, 'setting': 'official VA setting', "
        "'dialogue': 'C VA dialogue'},\n"
        "]\n",
        encoding="utf-8",
    )
    tool = tmp_path / "unused-tool"
    tool.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    tool.chmod(0o755)
    config = {
        "schema_name": "mprisk_generated_archive_freeze_config_v1",
        "freeze_id": "generated_round1_v1",
        "source_root": str(source_root),
        "output_root": "data/frozen/generated_round1_v1",
        "external_media_root": str(tmp_path / "external-media"),
        "official_archetypes_path": str(archetypes),
        "official_c_templates_path": str(c_templates),
        "ffmpeg_path": str(tool),
        "ffprobe_path": str(tool),
        "archives": {
            "accept_a_svt": {
                "data_type": "A",
                "sample_type": "Conflict",
                "protocol": "VT",
                "media_field": "silent",
                "expected_count": 1,
                "expected_gt_eligible": 1,
            },
            "accept_a_va": {
                "data_type": "A",
                "sample_type": "Conflict",
                "protocol": "VA",
                "media_field": "primary",
                "expected_count": 1,
                "expected_gt_eligible": 1,
            },
            "accept_c_svt": {
                "data_type": "C",
                "sample_type": "Aligned",
                "protocol": "VT",
                "media_field": "silent",
                "expected_count": 1,
                "expected_gt_eligible": 1,
            },
            "accept_c_va": {
                "data_type": "C",
                "sample_type": "Aligned",
                "protocol": "VA",
                "media_field": "primary",
                "expected_count": 1,
                "expected_gt_eligible": 1,
            },
        },
        "silent_copy_overrides": [],
    }
    config_path = tmp_path / "freeze.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return repo_root, config_path


def test_freeze_is_complete_deterministic_and_provenanced(
    frozen_fixture: tuple[Path, Path],
) -> None:
    repo_root, config_path = frozen_fixture
    first = freeze_generated_round1(repo_root=repo_root, config_path=config_path)
    first_bytes = {
        path: path.read_bytes()
        for path in (first.archive_manifest_path, first.gt_eligible_path, first.provenance_path)
    }
    second = freeze_generated_round1(repo_root=repo_root, config_path=config_path)

    assert first.total_count == second.total_count == 4
    assert first.gt_eligible_count == second.gt_eligible_count == 4
    assert {path: path.read_bytes() for path in first_bytes} == first_bytes
    archive_rows = _read_jsonl(first.archive_manifest_path)
    eligible_rows = _read_jsonl(first.gt_eligible_path)
    assert len({(row["source_archive"], row["original_variant_id"]) for row in archive_rows}) == 4
    assert len({row["sample_id"] for row in archive_rows}) == 4
    assert [row["context_source"] for row in eligible_rows] == [
        "setting",
        "trigger",
        "setting",
        "setting",
    ]
    assert [row["anchor"]["source_kind"] for row in eligible_rows] == [
        "recorded",
        "recorded",
        "official_template",
        "official_template",
    ]
    provenance = json.loads(first.provenance_path.read_text(encoding="utf-8"))
    assert provenance["counts"]["gt_eligible_anchor_source"] == {
        "official_template": 2,
        "recorded": 2,
    }
    assert provenance["counts"]["gt_eligible_context_source"] == {
        "setting": 3,
        "trigger": 1,
    }
    for artifact in provenance["artifacts"].values():
        path = repo_root / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_existing_freeze_rejects_changed_source(
    frozen_fixture: tuple[Path, Path],
) -> None:
    repo_root, config_path = frozen_fixture
    result = freeze_generated_round1(repo_root=repo_root, config_path=config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    index_path = Path(config["source_root"]) / "accept_a_svt/index.jsonl"
    row = _read_jsonl(index_path)[0]
    row["dialogue_text"] = "changed but still valid"
    _write_jsonl(index_path, [row])

    with pytest.raises(ValueError, match="Immutable freeze outputs"):
        freeze_generated_round1(repo_root=repo_root, config_path=config_path)
    assert result.archive_manifest_path.exists()


def test_source_media_path_cannot_escape_archive(
    frozen_fixture: tuple[Path, Path],
) -> None:
    repo_root, config_path = frozen_fixture
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    index_path = Path(config["source_root"]) / "accept_a_va/index.jsonl"
    row = _read_jsonl(index_path)[0]
    row["files"]["primary"] = "../escaped.mp4"
    _write_jsonl(index_path, [row])

    with pytest.raises(ValueError, match="contained relative path"):
        freeze_generated_round1(repo_root=repo_root, config_path=config_path)


@pytest.mark.parametrize("trigger", [None, "", "   ", "T1", " T2 ", "T3", "T4"])
def test_context_rejects_missing_or_template_trigger(trigger: str | None) -> None:
    assert select_natural_context({"setting": None, "trigger": trigger}) is None


def test_context_prefers_setting_and_keeps_regex_boundary() -> None:
    assert select_natural_context({"setting": " place ", "trigger": "reason"}) == (
        "place",
        "setting",
    )
    assert select_natural_context({"setting": None, "trigger": "T5"}) == ("T5", "trigger")
    assert select_natural_context({"setting": None, "trigger": "T1: reason"}) == (
        "T1: reason",
        "trigger",
    )


def test_recorded_anchor_pair_is_strict() -> None:
    with pytest.raises(ValueError, match="complete pair"):
        _resolve_anchor(
            source_archive="accept_a_svt",
            source_line=1,
            row={"data_type": "A", "archetype_id": 1, "gt_emotion": "joy"},
            official_archetypes={},
            official_c_templates=[],
        )
