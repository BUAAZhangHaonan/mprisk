from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import mprisk.recovery.pipeline as recovery_pipeline
from mprisk.recovery.pipeline import _export, _prepare_inputs


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _reuse_config(tmp_path: Path, *, max_new_tokens: int = 64) -> dict[str, object]:
    sample_ids = ["s1", "s2"]
    legacy = tmp_path / "legacy.jsonl"
    formal = tmp_path / "formal.jsonl"
    descriptions = tmp_path / "descriptions.jsonl"
    legacy_sha = _write_jsonl(
        legacy,
        [
            {"sample_id": sample_id, "gt_describe": f"GT {sample_id}"}
            for sample_id in sample_ids
        ],
    )
    formal_sha = _write_jsonl(
        formal, [{"sample_id": sample_id} for sample_id in sample_ids]
    )
    description_sha = _write_jsonl(
        descriptions,
        [
            {
                "schema": "mprisk_v2_diagnostic_description_v1",
                "sample_id": sample_id,
                "subject_model_key": "llava_v1_5_7b",
                "protocol": "VT",
                "max_new_tokens": max_new_tokens,
                "generated_description": f"Description {sample_id}",
            }
            for sample_id in sample_ids
        ],
    )
    return {
        "model_key": "llava_v1_5_7b",
        "protocol": "vt",
        "output_root": str(tmp_path / "output"),
        "legacy_assigned_manifest": str(legacy),
        "formal_manifest": str(formal),
        "counts": {"diagnostic": 0, "formal": 2, "unmatched": 0, "prompts": 8},
        "sha256": {
            "legacy_assigned_manifest": legacy_sha,
            "formal_manifest": formal_sha,
        },
        "reused_description": {
            "path": str(descriptions),
            "sha256": description_sha,
            "rows": 2,
            "schema": "mprisk_v2_diagnostic_description_v1",
            "subject_model_key": "llava_v1_5_7b",
            "protocol": "VT",
            "max_new_tokens": 64,
        },
    }


def test_prepare_inputs_attests_reused_descriptions_without_regeneration(
    tmp_path: Path,
) -> None:
    config = _reuse_config(tmp_path)
    result = _prepare_inputs(config)

    output = Path(config["output_root"])
    receipt = json.loads(
        (output / "inputs" / "reused_description_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert result == {"diagnostic_rows": 0, "formal_rows": 2, "unmatched_count": 0}
    assert receipt["status"] == "PASS"
    assert receipt["generation_contract"]["max_new_tokens"] == 64
    assert not (output / "inputs" / "diagnostic_manifest.jsonl").exists()
    assert not (output / "inputs" / "gt_descriptions.jsonl").exists()


def test_prepare_inputs_rejects_reused_description_policy_mismatch(
    tmp_path: Path,
) -> None:
    config = _reuse_config(tmp_path, max_new_tokens=63)

    with pytest.raises(ValueError, match="max_new_tokens"):
        _prepare_inputs(config)


def _mock_frozen_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, relation_rows: int, bundle_rows: int
) -> dict[str, object]:
    frozen_root = tmp_path / "frozen_export"
    frozen_root.mkdir(parents=True)
    manifest = frozen_root / "spherical_embedding_manifest.jsonl"
    receipt = frozen_root / "spherical_embedding_manifest.receipt.json"
    manifest.write_text("{}\n" * bundle_rows, encoding="utf-8")
    receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        recovery_pipeline,
        "export_frozen_representations",
        lambda **_: SimpleNamespace(
            count=relation_rows,
            bundle_manifest_path=manifest,
        ),
    )
    monkeypatch.setattr(
        recovery_pipeline,
        "read_validated_jsonl",
        lambda *_, **__: [{} for _ in range(bundle_rows)],
    )
    return {
        "output_root": str(tmp_path),
        "counts": {"formal": 2, "prompts": 8},
    }


def test_export_distinguishes_relation_rows_from_formal_bundle_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _mock_frozen_export(
        tmp_path, monkeypatch, relation_rows=16, bundle_rows=2
    )

    result = _export(config)

    assert result["count"] == 2
    assert result["relation_rows"] == 16


@pytest.mark.parametrize(
    ("relation_rows", "bundle_rows", "error"),
    [
        (15, 2, "Frozen relation export count mismatch"),
        (16, 1, "Frozen bundle count is not formal cache-closed count"),
    ],
)
def test_export_rejects_relation_or_bundle_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relation_rows: int,
    bundle_rows: int,
    error: str,
) -> None:
    config = _mock_frozen_export(
        tmp_path,
        monkeypatch,
        relation_rows=relation_rows,
        bundle_rows=bundle_rows,
    )

    with pytest.raises(RuntimeError, match=error):
        _export(config)


def test_phi3_recovery_description_uses_supported_eager_attention() -> None:
    repository = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (
            repository
            / "configs/recovery/phi3_5_vision_descriptions_20260727.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["attn_implementation"] == "eager"
