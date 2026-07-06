from __future__ import annotations

import pytest

from mprisk.data.dataset_registry import final_manifest_path
from mprisk.data.manifests import (
    FinalManifestRow,
    filter_manifest_rows,
    read_final_manifest,
    read_jsonl,
    write_jsonl,
)


def _manifest_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": "sample-1",
        "source_dataset": "ch_sims_v2",
        "source_id": "source-1",
        "protocol": "vt",
        "sample_type": "Conflict",
        "split_group_id": "source-1",
        "media_paths": {"vision": "video.mp4", "text": "caption.txt"},
        "views": {
            "M1": {"modality": "vision", "label": "positive", "is_clear": True},
            "M2": {"modality": "text", "label": "negative", "is_clear": True},
            "M12": {"modality": "vision+text", "label": "negative", "is_clear": True},
        },
        "use_in_main": True,
    }
    row.update(overrides)
    return row


def test_processed_manifest_placeholders_are_readable() -> None:
    for key in ("unified", "conflict", "aligned"):
        rows = read_jsonl(final_manifest_path(key))
        assert rows[0]["schema"] == "mprisk_sample_manifest_v1"
        assert read_final_manifest(final_manifest_path(key)) == []


def test_read_final_manifest_skips_placeholder_rows_and_normalizes_protocol(tmp_path) -> None:
    path = tmp_path / "unified_sample_manifest.jsonl"
    write_jsonl(
        path,
        [
            {"schema": "mprisk_sample_manifest_v1"},
            {"schema": "mprisk_sample_manifest_v1", "sample_type": "Conflict"},
            _manifest_row(protocol="vt"),
        ],
    )

    rows = read_final_manifest(path)

    assert len(rows) == 1
    assert isinstance(rows[0], FinalManifestRow)
    assert rows[0].sample_id == "sample-1"
    assert rows[0].protocol == "VT"
    assert rows[0].views.M1["modality"] == "vision"
    assert rows[0].media_paths == {"vision": "video.mp4", "text": "caption.txt"}
    assert rows[0].use_in_main is True


def test_read_final_manifest_rejects_rows_missing_required_final_fields(tmp_path) -> None:
    path = tmp_path / "conflict_manifest.jsonl"
    invalid = _manifest_row()
    invalid.pop("use_in_main")
    write_jsonl(path, [invalid])

    with pytest.raises(ValueError, match="use_in_main"):
        read_final_manifest(path)


def test_filter_manifest_rows_accepts_sample_type_protocol_source_and_main_flag(tmp_path) -> None:
    path = tmp_path / "aligned_manifest.jsonl"
    write_jsonl(
        path,
        [
            _manifest_row(sample_id="main", protocol="VA", sample_type="Aligned", use_in_main=True),
            _manifest_row(sample_id="supp", protocol="VT", source_dataset="dfew", use_in_main=False),
        ],
    )
    rows = read_final_manifest(path)

    filtered = filter_manifest_rows(
        rows,
        sample_type="Aligned",
        protocol="va",
        source_dataset="ch_sims_v2",
        use_in_main=True,
    )

    assert [row.sample_id for row in filtered] == ["main"]
