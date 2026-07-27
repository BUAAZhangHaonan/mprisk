from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mprisk.recovery.pipeline import _prepare_inputs


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
