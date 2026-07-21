from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("build_taffc_complete_bundle.py")
SPEC = importlib.util.spec_from_file_location("build_taffc_complete_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


def test_normalize_rel_rejects_absolute_and_parent_paths() -> None:
    assert bundle.normalize_rel("datasets/a.jsonl") == "datasets/a.jsonl"
    with pytest.raises(bundle.BundleError):
        bundle.normalize_rel("/absolute/path")
    with pytest.raises(bundle.BundleError):
        bundle.normalize_rel("a/../b")


def test_validate_task_matrix_accepts_exact_8_by_3_matrix() -> None:
    sample_ids = {"sample:a", "sample:b"}
    rows = [
        {"sample_id": sample_id, "prompt_id": f"p{prompt}", "condition": condition}
        for sample_id in sorted(sample_ids)
        for prompt in range(8)
        for condition in ("M1", "M2", "M12")
    ]
    bundle.validate_task_matrix(rows, sample_ids, "fixture")


def test_validate_task_matrix_fails_on_duplicate_task() -> None:
    rows = [
        {"sample_id": "sample:a", "prompt_id": f"p{prompt}", "condition": condition}
        for prompt in range(8)
        for condition in ("M1", "M2", "M12")
    ]
    rows[-1] = dict(rows[0])
    with pytest.raises(bundle.BundleError, match="duplicate task"):
        bundle.validate_task_matrix(rows, {"sample:a"}, "fixture")


def test_builder_hardlinks_and_records_inode(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    source = repo / "source.bin"
    source.write_bytes(b"payload")
    output = repo / "outputs/delivery"
    builder = bundle.BundleBuilder(repo, output, workers=1, dry_run=False, probe_streams=False)
    builder.prepare()
    builder.link(source, "payload/source.bin", nonzero=True)
    target = builder.staging / "payload/source.bin"
    assert source.stat().st_ino == target.stat().st_ino
    assert builder.records["payload/source.bin"].mode == "hardlink"


def test_formal_model_scope_is_exact() -> None:
    assert len(bundle.FORMAL_LABEL_MODELS) == 15
    assert bundle.FORMAL_LABEL_MODELS["gemma4_12b"] == "VA"
    assert bundle.FORMAL_LABEL_MODELS["qwen2_5_omni_7b"] == "VA"
    assert "gemma4_12b_it" not in bundle.FORMAL_LABEL_MODELS
    assert "phi4_multimodal" not in bundle.FORMAL_LABEL_MODELS
    assert set(bundle.STATE_SPECS) == {
        "qwen3_vl_8b",
        "internvl3_5_8b",
        "qwen2_5_omni_7b",
    }
