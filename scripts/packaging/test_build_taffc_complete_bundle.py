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


def test_cache_path_must_stay_inside_dataset_model_subtree() -> None:
    root = "caches/generated_set/qwen3_vl_8b"
    path = f"{root}/payload/source_001/shard.safetensors"
    assert bundle.require_within_package_subtree(path, root, "fixture") == path
    with pytest.raises(bundle.BundleError, match="path escapes"):
        bundle.require_within_package_subtree(
            "caches/natural_set/ch_sims_v2/qwen3_5_4b/payload/shard.safetensors",
            root,
            "fixture",
        )


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
    assert bundle.GENERATED_CACHE_ROOT == "caches/generated_set"
    assert bundle.NATURAL_CACHE_ROOT == "caches/natural_set/ch_sims_v2"
    assert set(bundle.UNION_CACHE_SPECS) | set(bundle.FULL_CACHE_SPECS) == {
        "qwen3_vl_8b",
        "internvl3_5_8b",
        "qwen2_5_omni_7b",
        "qwen3_5_4b",
        "gemma4_12b",
    }


def test_atomic_exchange_directories(tmp_path: Path) -> None:
    first = tmp_path / "candidate"
    second = tmp_path / "final"
    first.mkdir()
    second.mkdir()
    (first / "identity.txt").write_text("candidate", encoding="utf-8")
    (second / "identity.txt").write_text("old", encoding="utf-8")
    bundle.atomic_exchange_directories(first, second)
    assert (first / "identity.txt").read_text(encoding="utf-8") == "old"
    assert (second / "identity.txt").read_text(encoding="utf-8") == "candidate"


def test_independent_verification_record_is_fail_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "SHA256SUMS").write_text("0" * 64 + "  payload.bin\n", encoding="utf-8")
    status = tmp_path / "verify.status"
    status.write_text("0\n", encoding="utf-8")
    log = tmp_path / "verify.log"
    log.write_text(
        '{"status":"PASS","sha_files":1,"generated":3810,'
        '"ch_sims_protocol_rows":4225,"generated_cache_models":5,'
        '"natural_cache_models":{"ch_sims_v2":["qwen3_5_4b"]}}\n',
        encoding="utf-8",
    )
    record = bundle.require_independent_verification_record(candidate, status, log)
    assert record["sha_files"] == 1
    status.write_text("1\n", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="status is not 0"):
        bundle.require_independent_verification_record(candidate, status, log)
