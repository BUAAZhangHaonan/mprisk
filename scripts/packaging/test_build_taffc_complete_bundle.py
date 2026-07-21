from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("build_taffc_complete_bundle.py")
SPEC = importlib.util.spec_from_file_location("build_taffc_complete_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


def _write_legacy_control_bundle(output: Path, identity_path: Path) -> bytes:
    output.mkdir()
    payload = output / "payload.bin"
    payload.write_bytes(b"immutable payload")
    inventory = {
        "schema": "taffc_complete_bundle_inventory_v2",
        "bundle_name": identity_path.name,
        "bundle_path": str(identity_path.absolute()),
        "scope": {},
    }
    controls = bundle.build_control_payloads(identity_path, inventory, [])
    for rel, content in controls.items():
        (output / rel).write_bytes(content)
    hashed = {
        "payload.bin": bundle.sha256_file(payload),
        **{rel: bundle.sha256_file(output / rel) for rel in bundle.ROOT_METADATA_FILES},
    }
    sha_bytes = b"".join(f"{hashed[rel]}  {rel}\n".encode() for rel in sorted(hashed))
    (output / "SHA256SUMS").write_bytes(sha_bytes)
    with (output / "file_provenance.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["relative_path", "source_path", "size_bytes", "allocated_bytes", "sha256", "link_mode"])
        for rel in sorted(hashed):
            target = output / rel
            writer.writerow([rel, "fixture", target.stat().st_size, 0, hashed[rel], "generated"])
    return b"".join(
        line for line in sha_bytes.splitlines(keepends=True) if line.split(b"  ", 1)[1].decode().strip() not in bundle.CONTROL_FILES
    )


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


def test_finalize_excludes_all_root_controls_from_payload_manifests(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    output = repo / "outputs/final"
    builder = bundle.BundleBuilder(repo, output, workers=1, dry_run=False, probe_streams=False)
    builder.prepare()
    builder.write_bytes("payload.bin", b"payload", "fixture")
    builder.write_control_payload()
    builder.finalize()

    sha_paths = {
        line.split("  ", 1)[1]
        for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert sha_paths == {"payload.bin"}
    with (output / "file_provenance.tsv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["relative_path"] for row in rows] == ["payload.bin"]
    assert {path.name for path in output.iterdir()} == bundle.CONTROL_FILES | {"payload.bin"}
    bundle.verify_control_metadata(output)


def test_control_refresh_migrates_legacy_entries_without_payload_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "canonical_bundle"
    expected_payload_sha = _write_legacy_control_bundle(output, tmp_path / "candidate_bundle")
    payload = output / "payload.bin"
    before = (payload.read_bytes(), payload.stat().st_ino, payload.stat().st_mtime_ns)
    original_sha256_file = bundle.sha256_file
    hashed_paths: list[Path] = []

    def control_only_sha(path: Path) -> str:
        path = Path(path)
        assert path.parent == output
        assert path.name in bundle.CONTROL_FILES
        hashed_paths.append(path)
        return original_sha256_file(path)

    monkeypatch.setattr(bundle, "sha256_file", control_only_sha)
    result = bundle.refresh_control_metadata(output)

    assert result["payload_rehashed"] is False
    assert result["payload_sha_files"] == 1
    assert result["payload_sha256sums_digest"] == hashlib.sha256(expected_payload_sha).hexdigest()
    assert set(result["legacy_control_entries_removed"]) == bundle.LEGACY_HASHED_CONTROL_FILES
    assert (payload.read_bytes(), payload.stat().st_ino, payload.stat().st_mtime_ns) == before
    assert (output / "SHA256SUMS").read_bytes() == expected_payload_sha
    assert {path.name for path in hashed_paths} == bundle.LEGACY_HASHED_CONTROL_FILES | {"SHA256SUMS"}
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert inventory["bundle_name"] == output.name
    assert inventory["bundle_path"] == str(output)
    assert report["bundle"] == bundle.bundle_identity(output)
    assert report["checksum_policy"]["excluded_control_files"] == sorted(bundle.CONTROL_FILES)
    with (output / "file_provenance.tsv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["relative_path"] for row in rows] == ["payload.bin"]

    hashed_paths.clear()
    second = bundle.refresh_control_metadata(output)
    assert second["legacy_control_entries_removed"] == []
    assert (output / "SHA256SUMS").read_bytes() == expected_payload_sha
    assert hashed_paths == [output / "SHA256SUMS"]


def test_promotion_refreshes_candidate_identity_to_final_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate_bundle"
    final = tmp_path / "canonical_bundle"
    _write_legacy_control_bundle(candidate, candidate)
    bundle.refresh_control_metadata(candidate)
    final.mkdir()
    (final / "old.txt").write_text("old bundle", encoding="utf-8")
    monkeypatch.setattr(bundle, "require_independent_verification_record", lambda *args: {"status": "PASS"})
    monkeypatch.setattr(
        bundle,
        "_verify_package_indexes",
        lambda _path: {"generated_cache_models": 5},
    )
    verified_identities: list[dict[str, str]] = []

    def verify_identity(path: Path, _workers: int) -> None:
        bundle.verify_control_metadata(path)
        verified_identities.append(bundle.bundle_identity(path))

    monkeypatch.setattr(bundle, "verify_bundle", verify_identity)
    bundle.promote_verified_bundle(
        candidate,
        final,
        workers=1,
        verified_status=tmp_path / "verify.status",
        verified_log=tmp_path / "verify.log",
    )

    assert verified_identities == [bundle.bundle_identity(final)]
    assert not candidate.exists()
    assert (final / "payload.bin").read_bytes() == b"immutable payload"
    inventory = json.loads((final / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["bundle_name"] == final.name
    assert inventory["bundle_path"] == str(final)


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
