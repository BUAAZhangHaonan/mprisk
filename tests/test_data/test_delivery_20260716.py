from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import mprisk.data.delivery_20260716 as delivery
from mprisk.data.manifests import read_jsonl
from mprisk.data.splits import assign_split


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_for_split(prefix: str, split: str, start: int = 0) -> str:
    for index in range(start, start + 10000):
        candidate = f"{prefix}-{index}"
        if assign_split(f"delivery_20260716:{candidate}") == split:
            return candidate
    raise AssertionError("could not construct deterministic split fixture")


def _write_delivery(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, str], Path]:
    root.mkdir()
    media = root.parent / "media"
    media.mkdir()
    aligned_val_ids = [
        _group_for_split("aligned-val", "val", index * 100) for index in range(4)
    ]
    file_roles = (
        (
            "vt_a_manifest.jsonl",
            "VT",
            "Conflict",
            [_group_for_split("vt-a", "train"), _group_for_split("vt-a-val", "val")],
        ),
        (
            "vt_c_manifest.jsonl",
            "VT",
            "Aligned",
            [_group_for_split("vt-c", "train"), *aligned_val_ids[:2]],
        ),
        (
            "va_a_manifest.jsonl",
            "VA",
            "Conflict",
            [_group_for_split("va-a", "test"), _group_for_split("va-a-val", "val")],
        ),
        (
            "va_c_manifest.jsonl",
            "VA",
            "Aligned",
            [_group_for_split("va-c", "train"), *aligned_val_ids[2:]],
        ),
    )
    specs = []
    invalid_asset: tuple[str, str, Path] | None = None
    for filename, protocol, sample_type, source_ids in file_roles:
        rows = []
        for index, source_id in enumerate(source_ids):
            media_path = media / f"{filename}-{index}.mp4"
            media_path.write_bytes(b"media")
            paths = {"vision": str(media_path)}
            if protocol == "VA":
                paths["audio"] = str(media_path)
            rows.append(
                {
                    "sample_id": f"sample:{filename}:{index}",
                    "source_id": source_id,
                    "protocol": protocol,
                    "sample_type": sample_type,
                    "media_paths": paths,
                    "text_content": "A complete utterance.",
                    "gt_emotion": "sadness",
                    "surface_emotion": "composure" if sample_type == "Conflict" else None,
                    "gt_describe": "A complete grounded affect description.",
                    "rationale": "",
                    "generation_info": {"seed": index},
                    "source_is_generated": True,
                }
            )
            if filename == "va_a_manifest.jsonl" and index == 0:
                invalid_asset = (rows[-1]["sample_id"], source_id, media_path)
        path = root / filename
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        specs.append(delivery.SourceSpec(filename, protocol, sample_type, len(rows), _sha256(path)))
    monkeypatch.setattr(delivery, "SOURCE_SPECS", tuple(specs))
    assert invalid_asset is not None
    invalid_sample_id, invalid_source_id, invalid_media_path = invalid_asset
    monkeypatch.setattr(
        delivery,
        "EXPECTED_INVALID_VA_ASSETS",
        (
            delivery.InvalidVaAssetSpec(
                invalid_sample_id,
                invalid_source_id,
                _sha256(invalid_media_path),
            ),
        ),
    )

    def fake_ffprobe(command: list[str], **_kwargs: object) -> SimpleNamespace:
        media_path = Path(command[-1])
        streams = [{"index": 0, "codec_type": "video", "codec_name": "h264"}]
        if media_path != invalid_media_path:
            streams.append({"index": 1, "codec_type": "audio", "codec_name": "aac"})
        return SimpleNamespace(stdout=json.dumps({"streams": streams}))

    monkeypatch.setattr(delivery.subprocess, "run", fake_ffprobe)
    return {spec.filename: spec.sha256 for spec in specs}, media


def test_ingestion_is_read_only_and_builds_deterministic_manifests_and_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    hashes_before, _media = _write_delivery(source, monkeypatch)
    output = tmp_path / "derived"

    result = delivery.ingest_delivery_20260716(source_root=source, output_root=output)

    assert result.total_rows == 10
    assert {name: _sha256(source / name) for name in hashes_before} == hashes_before
    unified = read_jsonl(output / "manifests/unified_sample_manifest.jsonl")
    assert {row["sample_type"] for row in unified} == {"Conflict", "Aligned"}
    assert {row["protocol"] for row in unified} == {"VT", "VA"}
    assert all(row["split"] == assign_split(row["split_group_id"]) for row in unified)
    assert all(row["use_in_main"] is True and row["annotation_count"] == 1 for row in unified)
    assert len(read_jsonl(output / "manifests/va_aux.jsonl")) == 5
    assert len(read_jsonl(output / "manifests/va_state_valid.jsonl")) == 4
    invalid_assets = read_jsonl(output / "manifests/invalid_assets.jsonl")
    assert len(invalid_assets) == 1
    assert invalid_assets[0]["reason"] == "missing_audio_stream"
    assert len(read_jsonl(output / "splits/representation_split_assignment_v1.jsonl")) == 9
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_read_only"] is True
    assert provenance["counts"]["total"] == 10
    assert provenance["counts"]["va_state_valid"] == 4
    assert provenance["counts"]["invalid_assets"] == 1
    cache_plan = yaml.safe_load((output / "state_cache_plan_v1.yaml").read_text())
    assert cache_plan["manifests"]["VA"] == "manifests/va_state_valid.jsonl"
    assert cache_plan["expected_tasks"]["VA_per_model"] == 4 * 8 * 3


def test_ingestion_fails_closed_on_file_role_label_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_delivery(source, monkeypatch)
    path = source / "vt_a_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["sample_type"] = "Aligned"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    specs = tuple(
        delivery.SourceSpec(
            spec.filename,
            spec.protocol,
            spec.sample_type,
            spec.rows,
            _sha256(source / spec.filename),
        )
        for spec in delivery.SOURCE_SPECS
    )
    monkeypatch.setattr(delivery, "SOURCE_SPECS", specs)

    with pytest.raises(ValueError, match="file role requires VT/Conflict"):
        delivery.ingest_delivery_20260716(source_root=source, output_root=tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_ingestion_rejects_source_sha_mismatch_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_delivery(source, monkeypatch)
    path = source / "va_c_manifest.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        delivery.ingest_delivery_20260716(source_root=source, output_root=tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_ingestion_fails_closed_when_invalid_va_asset_set_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_delivery(source, monkeypatch)
    monkeypatch.setattr(delivery, "EXPECTED_INVALID_VA_ASSETS", ())

    with pytest.raises(ValueError, match="VA invalid-asset set changed"):
        delivery.ingest_delivery_20260716(source_root=source, output_root=tmp_path / "derived")
    assert not (tmp_path / "derived").exists()
