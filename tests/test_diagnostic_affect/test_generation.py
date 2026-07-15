from __future__ import annotations

import json
from pathlib import Path

import pytest

from mprisk.diagnostic_affect.generation import (
    CANONICAL_DIAGNOSTIC_AFFECT_PROMPT,
    DiagnosticAffectDescriptionLedger,
    GenerationRequest,
    GenerationResult,
    _read_config,
    build_diagnostic_affect_description_plan,
    export_diagnostic_affect_descriptions,
    generate_diagnostic_affect_descriptions,
    validate_diagnostic_affect_description,
    verify_diagnostic_affect_descriptions,
)


def _manifest_row(
    sample_id: str,
    *,
    protocol: str,
    sample_type: str,
    media: Path,
) -> dict[str, object]:
    media_paths = {"vision": str(media)}
    if protocol == "VA":
        media_paths["audio"] = str(media)
    return {
        "sample_id": sample_id,
        "source_dataset": "demo",
        "source_id": sample_id,
        "protocol": protocol,
        "sample_type": sample_type,
        "split": "test",
        "split_group_id": sample_id,
        "media_paths": media_paths,
        "text_content": "A short dialogue.",
        "setting_text": "Must not be passed to the subject model.",
        "trigger_text": "Must not be passed to the subject model.",
        "surface_emotion": "Must not be passed to the subject model.",
        "use_in_main": True,
    }


def _write_manifest(tmp_path: Path) -> Path:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    rows = [
        _manifest_row("vt-conflict", protocol="VT", sample_type="Conflict", media=media),
        _manifest_row("vt-aligned", protocol="VT", sample_type="Aligned", media=media),
        _manifest_row("va-conflict", protocol="VA", sample_type="Conflict", media=media),
        _manifest_row("va-aligned", protocol="VA", sample_type="Aligned", media=media),
        {
            **_manifest_row(
                "other-dataset", protocol="VT", sample_type="Conflict", media=media
            ),
            "source_dataset": "other",
        },
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _model_path(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return model


def _plan(tmp_path: Path, *, protocol: str = "VT"):
    return build_diagnostic_affect_description_plan(
        manifest_path=_write_manifest(tmp_path),
        subject_model_key="subject_model",
        model_family="subject_family",
        model_path=_model_path(tmp_path),
        protocol=protocol,
        condition="M12",
        dataset="demo",
        split="test",
        max_new_tokens=32,
    )


def test_plan_uses_explicit_identity_and_does_not_leak_annotations(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert plan.counts == {"VT": 2}
    assert plan.signature["subject_model_key"] == "subject_model"
    assert plan.signature["model_family"] == "subject_family"
    assert plan.signature["dataset"] == "demo"
    assert plan.signature["split"] == "test"
    assert {task.request.sample_id for task in plan.tasks} == {"vt-conflict", "vt-aligned"}
    for task in plan.tasks:
        assert task.request.model_key == "subject_model"
        assert task.request.protocol == "vt"
        assert task.request.condition == "M12"
        assert task.request.use_audio_in_video is False
        assert set(task.request.media_paths) == {"vision"}
        text = task.request.messages[0]["content"][-1]["text"]
        assert text.endswith(CANONICAL_DIAGNOSTIC_AFFECT_PROMPT)
        for forbidden in ("setting", "trigger", "surface_emotion", "sample_type"):
            assert forbidden not in json.dumps(task.request.messages).lower()


def test_va_plan_uses_joint_vision_audio_condition(tmp_path: Path) -> None:
    plan = _plan(tmp_path, protocol="VA")

    assert plan.counts == {"VA": 2}
    for task in plan.tasks:
        assert task.request.protocol == "va"
        assert task.request.condition == "M12"
        assert task.request.use_audio_in_video is True
        assert set(task.request.media_paths) == {"vision", "audio"}
        assert task.request.messages[0]["content"][-1]["text"] == (
            CANONICAL_DIAGNOSTIC_AFFECT_PROMPT
        )


def test_result_requires_exactly_one_sentence() -> None:
    request = GenerationRequest(
        sample_id="sample",
        model_key="subject_model",
        protocol="vt",
        condition="M12",
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CANONICAL_DIAGNOSTIC_AFFECT_PROMPT}
                ],
            },
        ),
        media_paths={"vision": "/tmp/sample.mp4"},
        use_audio_in_video=False,
        generation_kwargs={"do_sample": False, "num_beams": 1, "max_new_tokens": 32},
    )
    good = GenerationResult(
        request=request,
        text="The person appears emotionally unsettled.",
        token_ids=(101, 102),
        eos_token_ids=(102,),
        finish_reason="eos",
        input_token_count=9,
    )
    validate_diagnostic_affect_description(good)
    with pytest.raises(ValueError, match="exactly one sentence"):
        validate_diagnostic_affect_description(
            GenerationResult(
                request=request,
                text="First sentence. Second sentence.",
                token_ids=(1,),
                eos_token_ids=(),
                finish_reason="max_new_tokens",
                input_token_count=9,
            )
        )


def test_ledger_and_export_use_diagnostic_affect_description_field(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    ledger = DiagnosticAffectDescriptionLedger(tmp_path / "batch_state.sqlite3")
    ledger.prepare(plan.signature)
    with pytest.raises(ValueError, match="signature"):
        ledger.prepare({**plan.signature, "split": "train"})
    ledger.add_tasks(plan.tasks)
    task, attempt = next(ledger.pending_tasks(plan.tasks))
    result = GenerationResult(
        request=task.request,
        text="The person appears emotionally unsettled.",
        token_ids=(1, 2),
        eos_token_ids=(2,),
        finish_reason="eos",
        input_token_count=5,
    )
    ledger.complete(task.task_id, attempt, result, {"model_path": "model"})
    destination = tmp_path / "descriptions.jsonl"
    export_diagnostic_affect_descriptions(ledger.completed_records(), destination)
    row = json.loads(destination.read_text(encoding="utf-8"))
    assert row["subject_model_key"] == "subject_model"
    assert row["condition"] == "M12"
    assert row["DIAGNOSTIC_AFFECT_DESCRIPTION"] == result.text
    assert "text" not in row
    ledger.close()


def test_runner_and_verifier_are_model_family_independent(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    model_path = _model_path(tmp_path)
    plan = build_diagnostic_affect_description_plan(
        manifest_path=manifest,
        subject_model_key="subject_model",
        model_family="subject_family",
        model_path=model_path,
        protocol="VT",
        condition="M12",
        dataset="demo",
        split="test",
        max_new_tokens=32,
    )
    wrapper_kwargs: dict[str, object] = {}

    class FakeWrapper:
        def __init__(self, **kwargs: object) -> None:
            wrapper_kwargs.update(kwargs)

        def load(self) -> None:
            return None

        def generate_conditioned(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(
                request=request,
                text="The person appears emotionally unsettled.",
                token_ids=(10, 11),
                eos_token_ids=(11,),
                finish_reason="eos",
                input_token_count=4,
            )

        def close(self) -> None:
            return None

    output = tmp_path / "output"
    summary = generate_diagnostic_affect_descriptions(
        plan,
        output_root=output,
        subject_model_key="subject_model",
        model_family="subject_family",
        model_path=model_path,
        device="cpu",
        dtype="bfloat16",
        attn_implementation="sdpa",
        wrapper_factory=FakeWrapper,
    )

    assert wrapper_kwargs["model_key"] == "subject_model"
    assert summary == {"total": 2, "pending": 0, "running": 0, "completed": 2, "failed": 0}
    verification = verify_diagnostic_affect_descriptions(
        manifest_path=manifest,
        output_root=output,
        subject_model_key="subject_model",
        protocol="VT",
        condition="M12",
        dataset="demo",
        split="test",
    )
    assert verification["count"] == 2


def test_config_is_strict_and_rejects_legacy_schema(tmp_path: Path) -> None:
    config = {
        "schema_name": "mprisk_diagnostic_affect_description_config_v1",
        "run_name": "test",
        "asset_config": "assets.yaml",
        "manifest_path": "manifest.jsonl",
        "output_root": "output",
        "subject_model_key": "subject_model",
        "model_path": "model",
        "protocol": "VT",
        "condition": "M12",
        "dataset": "demo",
        "split": "test",
        "device": "cpu",
        "dtype": "bfloat16",
        "max_new_tokens": 32,
        "video_fps": 1.0,
        "attn_implementation": "sdpa",
    }
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(f"{key}: {value}" for key, value in config.items()), encoding="utf-8")
    assert _read_config(path) == config
    path.write_text(
        "schema_name: mprisk_diagnostic_description_legacy_config_v0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported"):
        _read_config(path)
