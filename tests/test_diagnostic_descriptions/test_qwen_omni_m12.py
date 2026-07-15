from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mprisk.diagnostic_descriptions.qwen_omni_m12 import (
    CANONICAL_PROMPT,
    DescriptionLedger,
    GenerationRequest,
    GenerationResult,
    build_description_plan,
    export_description_jsonl,
    run_generation,
    validate_description_result,
    verify_description_artifacts,
)


def _eligible_row(sample_id: str, protocol: str, media: Path) -> dict[str, object]:
    return {
        "schema_name": "mprisk_generated_gt_eligible_v1",
        "freeze_id": "generated_round1_v1",
        "sample_id": sample_id,
        "source_archive": "archive",
        "original_variant_id": sample_id,
        "data_type": "A",
        "sample_type": "Conflict",
        "protocol": protocol,
        "dialogue_text": "A short dialogue.",
        "setting_text": "A setting that must not be passed to the model.",
        "trigger_text": "A trigger that must not be passed to the model.",
        "context_text": "A context that must not be passed to the model.",
        "context_source": "setting",
        "anchor": {"emotion": "forbidden"},
        "model_input_path": str(media),
        "model_input_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "source_row_sha256": "source-hash",
    }


def _write_eligible(tmp_path: Path) -> Path:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    rows = [
        *[_eligible_row(f"vt-{index:03d}", "VT", media) for index in range(141)],
        *[_eligible_row(f"va-{index:03d}", "VA", media) for index in range(21)],
    ]
    path = tmp_path / "eligible.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _model_path(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return model


def test_plan_builds_vt_and_va_m12_requests_without_forbidden_metadata(tmp_path: Path) -> None:
    eligible = _write_eligible(tmp_path)

    plan = build_description_plan(
        eligible_path=eligible,
        model_path=_model_path(tmp_path),
        max_new_tokens=32,
        selected_sample_ids=("vt-000", "va-000"),
    )

    assert plan.counts == {"VT": 1, "VA": 1}
    vt, va = plan.tasks
    assert vt.request.protocol == "vt"
    assert vt.request.condition == "M12"
    assert vt.request.use_audio_in_video is False
    assert set(vt.request.media_paths) == {"vision"}
    assert [item["type"] for item in vt.request.messages[0]["content"]] == ["video", "text"]
    assert "A short dialogue." in vt.request.messages[0]["content"][-1]["text"]
    assert vt.request.messages[0]["content"][-1]["text"].endswith(CANONICAL_PROMPT)
    assert va.request.protocol == "va"
    assert va.request.condition == "M12"
    assert va.request.use_audio_in_video is True
    assert [item["type"] for item in va.request.messages[0]["content"]] == ["video", "text"]
    assert va.request.messages[0]["content"][-1]["text"] == CANONICAL_PROMPT
    serialized = json.dumps([task.request.messages for task in plan.tasks])
    for forbidden in (
        "setting",
        "trigger",
        "context",
        "anchor",
        "forbidden",
        "surface_emotion",
        "data_type",
    ):
        assert forbidden not in serialized.lower()


def test_full_plan_maps_the_exact_vt_va_eligible_population(tmp_path: Path) -> None:
    plan = build_description_plan(
        eligible_path=_write_eligible(tmp_path),
        model_path=_model_path(tmp_path),
        max_new_tokens=32,
    )

    assert len(plan.tasks) == 162
    assert plan.counts == {"VT": 141, "VA": 21}


def test_generation_result_requires_new_tokens_and_valid_one_sentence() -> None:
    request = GenerationRequest(
        sample_id="sample",
        model_key="qwen2_5_omni_7b",
        protocol="vt",
        condition="M12",
        messages=({"role": "user", "content": [{"type": "text", "text": CANONICAL_PROMPT}]},),
        media_paths={"vision": "/tmp/sample.mp4", "audio": "/tmp/sample.mp4"},
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
    validate_description_result(good)
    with pytest.raises(ValueError, match="exactly one sentence"):
        validate_description_result(
            GenerationResult(
                request=request,
                text="First sentence. Second sentence.",
                token_ids=(1,),
                eos_token_ids=(),
                finish_reason="max_new_tokens",
                input_token_count=9,
            )
        )


def test_ledger_rejects_signature_mismatch_and_atomic_jsonl_export(tmp_path: Path) -> None:
    eligible = _write_eligible(tmp_path)
    plan = build_description_plan(
        eligible_path=eligible,
        model_path=_model_path(tmp_path),
        max_new_tokens=32,
        selected_sample_ids=("vt-000", "va-000"),
    )
    ledger = DescriptionLedger(tmp_path / "batch_state.sqlite3")
    ledger.prepare(plan.signature)
    with pytest.raises(ValueError, match="signature"):
        ledger.prepare({**plan.signature, "max_new_tokens": 33})
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
    ledger.complete(task.task_id, attempt, result, {"model_sha256": "model"})
    ledger.validate_completed(plan.tasks)
    ledger.connection.execute(
        "UPDATE tasks SET input_sha256='tampered' WHERE task_id=?", (task.task_id,)
    )
    ledger.connection.commit()
    with pytest.raises(ValueError, match="(?i)completed task input hash"):
        ledger.validate_completed(plan.tasks)
    ledger.connection.execute(
        "UPDATE tasks SET input_sha256=? WHERE task_id=?", (task.input_sha256, task.task_id)
    )
    ledger.connection.commit()
    destination = tmp_path / "descriptions.jsonl"
    export_description_jsonl(ledger.completed_records(), destination)
    assert not (tmp_path / ".descriptions.jsonl.tmp").exists()
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["sample_id"] == "vt-000"
    assert rows[0]["text"] == result.text
    ledger.close()


def test_runner_uses_formal_ledger_and_verifies_smoke_manifest(tmp_path: Path) -> None:
    eligible = _write_eligible(tmp_path)
    plan = build_description_plan(
        eligible_path=eligible,
        model_path=_model_path(tmp_path),
        max_new_tokens=32,
        selected_sample_ids=("vt-000", "va-000"),
    )

    class FakeWrapper:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def load(self) -> None:
            return None

        def generate_conditioned(self, request):
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
    summary = run_generation(
        plan,
        output_root=output,
        model_path=_model_path(tmp_path),
        device="cpu",
        attn_implementation="sdpa",
        wrapper_factory=FakeWrapper,
    )

    assert summary == {"total": 2, "pending": 0, "running": 0, "completed": 2, "failed": 0}
    verification = verify_description_artifacts(
        eligible_path=eligible,
        output_root=output,
        strict_full=False,
    )
    assert verification["count"] == 2
