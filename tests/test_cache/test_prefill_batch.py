from __future__ import annotations

import json

import numpy as np
import pytest

from mprisk.cache.prefill_batch import BatchLedger, build_batch_plan, build_parser
from mprisk.cache.prefill_writer import write_full_cache_manifest, write_prefill_result
from mprisk.models.base_wrapper import PrefillRequest, PrefillResult


def _write_inputs(tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    rows = [
        {
            "sample_id": "included",
            "protocol": "va",
            "source_dataset": "dataset",
            "split": "train",
            "sample_type": "Aligned",
            "use_in_main": True,
            "annotation_count": 0,
            "text_content": "first",
            "media_paths": {"vision": str(media), "audio": str(media)},
        },
        {
            "sample_id": "excluded",
            "protocol": "va",
            "source_dataset": "dataset",
            "split": "test",
            "sample_type": "Conflict",
            "use_in_main": False,
            "annotation_count": 0,
            "text_content": "second",
            "media_paths": {"vision": str(media), "audio": str(media)},
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prompt_set = tmp_path / "prompts.yaml"
    prompt_set.write_text(
        """schema: mprisk_equiv_prompt_set_v1
key: va
protocol: va
version: v1
active: true
templates:
  - prompt_id: p1
    role: user
    enabled: true
    template_text: "Sample {sample_text}; {question}"
  - prompt_id: disabled
    role: user
    enabled: false
    template_text: "unused {sample_text}"
  - prompt_id: p2
    role: user
    enabled: true
    template_text: "Question {question}; {sample_text}"
""",
        encoding="utf-8",
    )
    return manifest, prompt_set


def _args(tmp_path, *, question: str | None):
    manifest, prompt_set = _write_inputs(tmp_path)
    argv = [
        "--manifest",
        str(manifest),
        "--prompt-set",
        str(prompt_set),
        "--model-path",
        str(tmp_path / "model"),
        "--output-root",
        str(tmp_path / "output"),
    ]
    if question is not None:
        argv += ["--prompt-variable", f"question={question}"]
    return build_parser().parse_args(argv)


def test_plan_uses_every_row_enabled_prompt_and_three_conditions(tmp_path) -> None:
    plan = build_batch_plan(_args(tmp_path, question="judge emotion"))

    assert len(plan.tasks) == 2 * 2 * 3
    assert plan.prompt_ids == ("p1", "p2")
    assert {task.sample_id for task in plan.tasks} == {"included", "excluded"}
    assert {task.condition for task in plan.tasks} == {"M1", "M2", "M12"}
    assert not plan.unresolved_prompt_variables
    assert all(task.prompt_text is not None for task in plan.tasks)


def test_plan_reports_missing_prompt_variable_without_inventing_value(tmp_path) -> None:
    plan = build_batch_plan(_args(tmp_path, question=None))

    assert plan.unresolved_prompt_variables == ("question",)
    assert len(plan.tasks) == 12
    assert all(task.prompt_text is None for task in plan.tasks)


def test_ledger_resets_interrupted_and_retries_failed_only_explicitly(tmp_path) -> None:
    plan = build_batch_plan(_args(tmp_path, question="judge emotion"))
    ledger = BatchLedger(tmp_path / "output" / "batch_state.sqlite3")
    ledger.prepare(plan, retry_failed=False)
    first, second = plan.tasks[:2]
    ledger.connection.execute("UPDATE tasks SET status='running' WHERE task_id=?", (first.task_id,))
    ledger.connection.execute("UPDATE tasks SET status='failed' WHERE task_id=?", (second.task_id,))
    ledger.connection.commit()

    ledger.prepare(plan, retry_failed=False)
    states = dict(ledger.connection.execute("SELECT task_id,status FROM tasks").fetchall())
    assert states[first.task_id] == "pending"
    assert states[second.task_id] == "failed"

    ledger.prepare(plan, retry_failed=True)
    state = ledger.connection.execute(
        "SELECT status FROM tasks WHERE task_id=?", (second.task_id,)
    ).fetchone()[0]
    assert state == "pending"
    ledger.close()


def test_writer_can_defer_and_atomically_materialize_manifest(tmp_path) -> None:
    request = PrefillRequest(
        sample_id="sample",
        model_key="model",
        protocol="va",
        condition="M12",
        dataset_key="dataset",
        split="test",
        messages=({"role": "user", "content": [{"type": "text", "text": "task"}]},),
        media_paths={"vision": "/tmp/media.mp4", "audio": "/tmp/media.mp4"},
        use_audio_in_video=True,
    )
    result = PrefillResult(
        request=request,
        trajectory=np.ones((2, 3), dtype=np.float32),
        token_count=2,
        t0_token_index=1,
        provenance={},
    )

    artifact = write_prefill_result(result, output_root=tmp_path, update_manifest=False)
    assert not artifact.manifest_path.exists()
    write_full_cache_manifest([artifact.entry], artifact.manifest_path)
    payload = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert payload["entries"] == [artifact.entry]

    with pytest.raises(ValueError, match="duplicate"):
        write_full_cache_manifest([artifact.entry, artifact.entry], artifact.manifest_path)
