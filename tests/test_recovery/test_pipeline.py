from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import mprisk.recovery.pipeline as recovery_pipeline
from mprisk.judge.ensemble_misread_v4 import EnsembleMisreadConfig
from mprisk.recovery.pipeline import (
    _export,
    _formal_judgment_intersection,
    _prepare_inputs,
    _run_description,
)


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


def _formal_diagnostic_config(tmp_path: Path) -> dict[str, object]:
    legacy = tmp_path / "legacy.jsonl"
    formal = tmp_path / "formal.jsonl"
    media = tmp_path / "media.mp4"
    media.write_bytes(b"media")
    legacy_rows = [
        {
            "sample_id": sample_id,
            "gt_describe": f"GT {sample_id}",
            "media_paths": {"vision": str(media), "audio": str(media)},
        }
        for sample_id in ("formal-1", "excluded-1", "formal-2")
    ]
    formal_rows = [{"sample_id": sample_id} for sample_id in ("formal-1", "formal-2")]
    return {
        "model_key": "phi4_multimodal",
        "protocol": "va",
        "output_root": str(tmp_path / "output"),
        "legacy_assigned_manifest": str(legacy),
        "formal_manifest": str(formal),
        "diagnostic_scope": "formal_intersection",
        "diagnostic_dataset": "formal-dataset",
        "diagnostic_split": "formal_intersection",
        "counts": {
            "legacy": 3,
            "diagnostic": 2,
            "formal": 2,
            "unmatched": 1,
            "prompts": 8,
        },
        "sha256": {
            "legacy_assigned_manifest": _write_jsonl(legacy, legacy_rows),
            "formal_manifest": _write_jsonl(formal, formal_rows),
        },
    }


def test_prepare_inputs_formal_scope_excludes_unmatched_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_diagnostic_config(tmp_path)
    monkeypatch.setattr(
        recovery_pipeline,
        "_probe_media_stream_types",
        lambda _: {"video", "audio"},
    )

    result = _prepare_inputs(config)

    output = Path(config["output_root"])
    diagnostic = recovery_pipeline._read_jsonl(
        output / "inputs" / "diagnostic_manifest.jsonl"
    )
    gt_rows = recovery_pipeline._read_jsonl(
        output / "inputs" / "gt_descriptions.jsonl"
    )
    report = json.loads(
        (output / "inputs" / "formal_intersection_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert result == {"diagnostic_rows": 2, "formal_rows": 2, "unmatched_count": 1}
    assert [row["sample_id"] for row in diagnostic] == ["formal-1", "formal-2"]
    assert [row["sample_id"] for row in gt_rows] == ["formal-1", "formal-2"]
    assert all(row["split"] == "formal_intersection" for row in diagnostic)
    assert report["diagnostic_scope"] == "formal_intersection"
    assert report["unmatched_ids"] == ["excluded-1"]


def test_prepare_inputs_rejects_formal_va_media_without_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_diagnostic_config(tmp_path)
    monkeypatch.setattr(
        recovery_pipeline,
        "_probe_media_stream_types",
        lambda _: {"video"},
    )

    with pytest.raises(ValueError, match="audio media has no audio stream"):
        _prepare_inputs(config)


def test_formal_scoped_judgments_attest_excluded_unmatched_ids(
    tmp_path: Path,
) -> None:
    config = _formal_diagnostic_config(tmp_path)
    root = Path(config["output_root"])
    _write_jsonl(
        root / "judgments_v4" / "judgments.jsonl",
        [{"sample_id": "formal-1"}, {"sample_id": "formal-2"}],
    )
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "inputs" / "formal_intersection_report.json").write_text(
        json.dumps({"unmatched_ids": ["excluded-1"]}), encoding="utf-8"
    )

    result = _formal_judgment_intersection(config)

    report = json.loads(
        (root / "judgments_v4" / "formal_intersection_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert result == {"formal_rows": 2, "unmatched_count": 1}
    assert report["input_rows"] == 2
    assert report["judgment_extra_ids"] == []
    assert report["unmatched_ids"] == ["excluded-1"]


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

    pipeline = yaml.safe_load(
        (
            repository
            / "configs/recovery/phi3_5_vision_in_domain_pipeline_20260727.yaml"
        ).read_text(encoding="utf-8")
    )
    assert pipeline["description_python_environment"] == {
        "PYTHONNOUSERSITE": "1"
    }
    assert pipeline["description_retry_failed"] is True
    assert pipeline["description_runtime_contract"] == {
        "python_executable": (
            "/home/team/zhanghaonan/.venvs/mprisk-phi3-vision-4.43-cu121/bin/python"
        ),
        "python_prefix": (
            "/home/team/zhanghaonan/.venvs/mprisk-phi3-vision-4.43-cu121"
        ),
        "python_version": "3.11.11",
        "user_site_enabled": False,
        "torch_cuda_version": "12.1",
        "package_versions": {
            "Pillow": "10.3.0",
            "PyYAML": "6.0.2",
            "accelerate": "0.30.0",
            "decord": "0.6.0",
            "numpy": "1.26.4",
            "torch": "2.3.0+cu121",
            "torchvision": "0.18.0+cu121",
            "transformers": "4.43.0",
        },
    }


def test_phi4_recovery_description_uses_pinned_isolated_runtime() -> None:
    repository = Path(__file__).resolve().parents[2]
    pipeline = yaml.safe_load(
        (
            repository
            / "configs/recovery/phi4_multimodal_in_domain_pipeline_20260727.yaml"
        ).read_text(encoding="utf-8")
    )

    assert pipeline["python_executable"] == (
        "/home/team/zhanghaonan/.venvs/mprisk-phi4-py310/bin/python"
    )
    assert pipeline["description_python_environment"] == {
        "PYTHONNOUSERSITE": "1"
    }
    assert pipeline["description_retry_failed"] is False
    assert pipeline["description_runtime_contract"] == {
        "python_executable": (
            "/home/team/zhanghaonan/.venvs/mprisk-phi4-py310/bin/python"
        ),
        "python_prefix": "/home/team/zhanghaonan/.venvs/mprisk-phi4-py310",
        "python_version": "3.10.18",
        "user_site_enabled": False,
        "torch_cuda_version": "12.4",
        "package_versions": {
            "Pillow": "11.1.0",
            "PyYAML": "6.0.3",
            "accelerate": "1.3.0",
            "backoff": "2.2.1",
            "peft": "0.13.2",
            "safetensors": "0.8.0",
            "scipy": "1.15.2",
            "soundfile": "0.13.1",
            "torch": "2.6.0+cu124",
            "torchvision": "0.21.0+cu124",
            "transformers": "4.48.2",
        },
    }


def test_phi4_recovery_loads_empty_started_call_binding_without_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    config_path = (
        repository
        / "configs/recovery/phi4_multimodal_in_domain_pipeline_20260727.yaml"
    )
    empty = repository / "configs/recovery/empty_forbidden_started_calls.jsonl"
    monkeypatch.setattr(recovery_pipeline, "_validate_static_inputs", lambda _: None)
    monkeypatch.setattr(recovery_pipeline, "load_training_config", lambda _: object())

    config = recovery_pipeline.load_pipeline_config(config_path)
    config["output_root"] = str(tmp_path / "phi4_multimodal")
    preflight = recovery_pipeline.dry_run_stage(config, "judgment")

    assert empty.read_bytes() == b""
    assert hashlib.sha256(empty.read_bytes()).hexdigest() == (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )
    assert config["forbidden_started_calls"] == [
        {
            "path": str(empty),
            "sha256": hashlib.sha256(empty.read_bytes()).hexdigest(),
        }
    ]
    assert preflight["status"] == "blocked_by_dependency"
    assert preflight["would_issue_api_requests"] is False


def test_description_stage_passes_explicit_retry_failed_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    description_config = tmp_path / "description.yaml"
    description_config.write_text(
        yaml.safe_dump(
            {
                "run_id": "run",
                "condition": "M12",
                "dataset": "dataset",
                "split": "recovery_all",
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def watcher(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(recovery_pipeline, "watch_description_generation", watcher)
    monkeypatch.setattr(
        recovery_pipeline,
        "verify_diagnostic_affect_descriptions",
        lambda **_: {"counts": {"VT": 2}},
    )
    config = {
        "counts": {"diagnostic": 2},
        "description_config": str(description_config),
        "description_retry_failed": True,
        "python_executable": "/env/bin/python",
        "model_key": "phi3_5_vision",
        "protocol": "vt",
        "output_root": str(tmp_path / "output"),
    }

    result = _run_description(config)

    assert result["status"] == "completed"
    assert captured["retry_failed"] is True
    with pytest.raises(ValueError, match="description_retry_failed must be boolean"):
        _run_description({**config, "description_retry_failed": "true"})


def _judgment_stage_config(
    tmp_path: Path, *, prompt_sha256: str = "a" * 64
) -> dict[str, object]:
    output_root = tmp_path / "output"
    descriptions = output_root / "descriptions"
    _write_jsonl(
        descriptions / "manifest.jsonl",
        [{"sample_id": "sample-1", "DIAGNOSTIC_AFFECT_DESCRIPTION": "Worried."}],
    )
    descriptions.joinpath("provenance.json").write_text(
        json.dumps(
            {
                "signature": {
                    "run_id": "subject-model-diagnostic-run",
                    "prompt_sha256": prompt_sha256,
                    "generation_policy_sha256": "b" * 64,
                    "request_protocol_signature_sha256": "c" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    forbidden_started_calls = output_root / "retired" / "started_calls.jsonl"
    forbidden_started_calls_sha256 = _write_jsonl(
        forbidden_started_calls,
        [{"call_id": "retired-paid-call"}],
    )
    return {
        "model_key": "subject_model",
        "protocol": "vt",
        "output_root": str(output_root),
        "counts": {"diagnostic": 1},
        "forbidden_started_calls": [
            {
                "path": str(forbidden_started_calls),
                "sha256": forbidden_started_calls_sha256,
            }
        ],
    }


def test_judgment_config_publishes_validated_json_compatible_yaml(tmp_path: Path) -> None:
    config = _judgment_stage_config(tmp_path)

    parsed = recovery_pipeline._build_judgment_config(config, publish=True)

    published_path = Path(config["output_root"]) / "judgments_v4" / "config.yaml"
    first_bytes = published_path.read_bytes()
    published = yaml.safe_load(published_path.read_text(encoding="utf-8"))
    assert published == parsed.model_dump(mode="json")
    path_fields = {
        field_name
        for field_name, field in EnsembleMisreadConfig.model_fields.items()
        if field.annotation is Path
    }
    assert path_fields == {
        "gt_coverage_receipt_path",
        "gt_description_manifest_path",
        "diagnostic_affect_description_manifest_path",
        "output_root",
    }
    for field in path_fields:
        assert isinstance(published[field], str)
    assert EnsembleMisreadConfig.model_validate(published) == parsed
    assert published["schema_name"] == "mprisk_ensemble_misread_judgment_config_v4"
    assert published["api_url"] == "https://api.deepseek.com/beta/chat/completions"
    assert published["thinking"] == "disabled"
    assert published["max_tokens"] == 256
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        EnsembleMisreadConfig.model_validate({**published, "unknown": "field"})

    assert recovery_pipeline._build_judgment_config(config, publish=True) == parsed
    assert published_path.read_bytes() == first_bytes


def test_judgment_config_dry_run_does_not_publish(tmp_path: Path) -> None:
    config = _judgment_stage_config(tmp_path)
    judgment_root = Path(config["output_root"]) / "judgments_v4"

    parsed = recovery_pipeline._build_judgment_config(config, publish=False)

    assert isinstance(parsed, EnsembleMisreadConfig)
    assert not judgment_root.exists()


def test_judgment_config_rejects_invalid_binding_before_publish(tmp_path: Path) -> None:
    config = _judgment_stage_config(tmp_path, prompt_sha256="invalid")

    with pytest.raises(ValueError, match="SHA-256"):
        recovery_pipeline._build_judgment_config(config, publish=True)

    assert not (Path(config["output_root"]) / "judgments_v4" / "config.yaml").exists()


def test_phi4_formal1934_queue_has_isolated_end_to_end_contract() -> None:
    repository = Path(__file__).resolve().parents[2]
    queue = yaml.safe_load(
        (
            repository
            / "configs/recovery/phi3_phi4_llava_in_domain_formal1934_20260804.yaml"
        ).read_text(encoding="utf-8")
    )
    pipeline = yaml.safe_load(
        (
            repository
            / "configs/recovery/phi4_multimodal_in_domain_pipeline_formal1934_20260804.yaml"
        ).read_text(encoding="utf-8")
    )
    descriptions = yaml.safe_load(
        (
            repository
            / "configs/recovery/phi4_multimodal_descriptions_formal1934_20260804.yaml"
        ).read_text(encoding="utf-8")
    )

    assert pipeline["diagnostic_scope"] == "formal_intersection"
    assert pipeline["counts"] == {
        "legacy": 1939,
        "diagnostic": 1934,
        "formal": 1934,
        "unmatched": 5,
        "prompts": 8,
    }
    assert descriptions["run_id"] == "phi4_multimodal_in_domain_formal1934_20260804"
    assert descriptions["split"] == "formal_intersection"
    assert descriptions["manifest_path"].endswith(
        "/in_domain_recovery_formal1934_20260804/phi4_multimodal/inputs/diagnostic_manifest.jsonl"
    )
    phi4_steps = [step for step in queue["steps"] if step["id"].startswith("phi4_")]
    assert {step["id"] for step in phi4_steps} >= {
        "phi4_descriptions_1934",
        "phi4_judgments_1934",
        "phi4_formal_1934_and_5_unmatched",
    }
    assert all("1939" not in step["id"] for step in phi4_steps)
    assert all(
        any(
            value.endswith(
                "phi4_multimodal_in_domain_pipeline_formal1934_20260804.yaml"
            )
            for value in step["command"]
        )
        for step in phi4_steps
    )
    completion_text = json.dumps(
        [step["completion"] for step in phi4_steps], sort_keys=True
    )
    assert "in_domain_recovery_formal1934_20260804" in completion_text
    assert "expected_rows\": 1939" not in completion_text
