from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mprisk.cache.prefill_batch import (
    BatchLedger,
    build_batch_plan,
    build_parser,
    main,
    rematerialize_completed_batch,
)
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
        prompt_set_key="va",
        prompt_id="p1",
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


def test_plan_signature_records_family_and_prompt_identity(tmp_path) -> None:
    args = _args(tmp_path, question="judge emotion")
    plan = build_batch_plan(args)

    assert plan.signature["family"] == "qwen_omni"
    assert plan.signature["prefill_strategy"] == "full_prefill"
    assert plan.signature["prefill_strategy_version"] == "v1"
    assert all(task.prompt_set_key == "va" for task in plan.tasks)


def test_prompt_kv_strategy_requires_qwen_vl_vt(tmp_path) -> None:
    args = _args(tmp_path, question="judge emotion")
    args.prefill_strategy = "qwen_vl_prompt_kv"
    with pytest.raises(ValueError, match="requires family 'qwen_vl'"):
        build_batch_plan(args)

    args.family = "qwen_vl"
    args.model_key = "qwen3_vl_8b"
    with pytest.raises(ValueError, match="requires protocol 'vt'"):
        build_batch_plan(args)


def test_prompt_kv_batch_groups_prompts_and_resume_checks_identity(tmp_path, capsys) -> None:
    manifest, prompt_set = _write_inputs(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["protocol"] = "vt"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    prompt_set.write_text(
        prompt_set.read_text(encoding="utf-8").replace("protocol: va", "protocol: vt"),
        encoding="utf-8",
    )
    output_root = tmp_path / "kv-output"
    argv = [
        "--manifest",
        str(manifest),
        "--prompt-set",
        str(prompt_set),
        "--prompt-variable",
        "question=judge emotion",
        "--protocol",
        "vt",
        "--family",
        "qwen_vl",
        "--model-key",
        "qwen3_vl_8b",
        "--model-path",
        str(tmp_path / "model"),
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--prefill-strategy",
        "qwen_vl_prompt_kv",
    ]

    class FakeWrapper:
        family = "qwen_vl"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            return None

        def close(self):
            return None

    class FakeExtractor:
        calls = 0

        def extract_condition_batch(self, **kwargs):
            type(self).calls += 1
            sample_row = kwargs["sample_row"]
            results = []
            for prompt_text, prompt_id in zip(
                kwargs["prompt_texts"],
                kwargs["prompt_ids"],
                strict=True,
            ):
                request = kwargs["build_request_fn"](
                    sample_id=str(sample_row["sample_id"]),
                    model_key="qwen3_vl_8b",
                    protocol="vt",
                    condition=kwargs["condition"],
                    dataset_key=str(sample_row["source_dataset"]),
                    split=str(sample_row["split"]),
                    media_paths=sample_row["media_paths"],
                    transcript=str(sample_row["text_content"]),
                    task_prompt=prompt_text,
                    prompt_set_key=kwargs["prompt_set_key"],
                    prompt_id=prompt_id,
                    **kwargs["common_kwargs"],
                )
                results.append(
                    PrefillResult(
                        request=request,
                        trajectory=np.ones((2, 3), dtype=np.float32),
                        token_count=4,
                        t0_token_index=3,
                        provenance={
                            "elapsed_seconds": 0.1,
                            "prefill_strategy": "qwen_vl_prompt_kv",
                            "prefill_strategy_version": "v1",
                            "prefix_identity": "a" * 64,
                        },
                    )
                )
            return results

    def extractor_factory(strategy, wrapper):
        assert strategy == "qwen_vl_prompt_kv"
        assert wrapper.family == "qwen_vl"
        return FakeExtractor()

    assert (
        main(
            argv,
            wrapper_factory=FakeWrapper,
            prompt_kv_extractor_factory=extractor_factory,
        )
        == 0
    )
    capsys.readouterr()
    assert FakeExtractor.calls == 6
    sidecar = next(output_root.glob("prompts/*/shards/qwen3_vl_8b/vt/M1/*.json"))
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["provenance"]["prefill_strategy"] == "qwen_vl_prompt_kv"
    assert payload["entry"]["metadata"]["prefill_strategy_version"] == "v1"
    assert payload["entry"]["metadata"]["prefix_identity"] == "a" * 64

    assert (
        main(
            argv,
            wrapper_factory=FakeWrapper,
            prompt_kv_extractor_factory=extractor_factory,
        )
        == 0
    )
    capsys.readouterr()
    assert FakeExtractor.calls == 6

    full_argv = [
        "full_prefill" if item == "qwen_vl_prompt_kv" else item
        for item in argv
    ]
    with pytest.raises(ValueError, match="ledger signature does not match"):
        main(full_argv, wrapper_factory=FakeWrapper)


def test_batch_resume_uses_checksums_and_records_full_identity(tmp_path, capsys) -> None:
    class FakeWrapper:
        extracted = 0

        def __init__(
            self,
            *,
            model_key,
            model_path,
            device,
            dtype,
            attn_implementation,
            min_pixels,
            max_pixels,
        ):
            self.kwargs = {
                "model_key": model_key,
                "model_path": model_path,
                "device": device,
                "dtype": dtype,
                "attn_implementation": attn_implementation,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }

        def load(self):
            return None

        def extract_prefill(self, request):
            type(self).extracted += 1
            return PrefillResult(
                request=request,
                trajectory=np.ones((2, 3), dtype=np.float32),
                token_count=4,
                t0_token_index=3,
                provenance={"elapsed_seconds": 0.1, "peak_gpu_memory_bytes": 1234},
            )

        def close(self):
            return None

    args = _args(tmp_path, question="judge emotion")
    argv = [
        "--manifest",
        str(args.manifest),
        "--prompt-set",
        str(args.prompt_set),
        "--prompt-variable",
        "question=judge emotion",
        "--model-path",
        str(args.model_path),
        "--output-root",
        str(args.output_root),
        "--device",
        "cpu",
    ]

    assert main(argv, wrapper_factory=FakeWrapper) == 0
    capsys.readouterr()
    assert FakeWrapper.extracted == 12
    combined = [
        json.loads(line)
        for line in (args.output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(combined) == 12
    assert all(entry["prompt_set_key"] == "va" for entry in combined)
    assert {entry["prompt_id"] for entry in combined} == {"p1", "p2"}
    assert all(entry["t0_token_index"] == 3 for entry in combined)
    assert all(entry["elapsed_seconds"] == 0.1 for entry in combined)
    assert all(entry["peak_gpu_memory_bytes"] == 1234 for entry in combined)

    ledger = BatchLedger(args.output_root / "batch_state.sqlite3")
    columns = {row[1] for row in ledger.connection.execute("PRAGMA table_info(tasks)")}
    identity_columns = {
        "sample_id",
        "model_key",
        "protocol",
        "prompt_set_key",
        "prompt_id",
        "condition",
    }
    assert identity_columns <= columns
    row = ledger.connection.execute(
        """SELECT layer_count,hidden_dim,token_count,t0_token_index,elapsed_seconds,
                  peak_gpu_memory_bytes,checksum
           FROM tasks WHERE status='completed' LIMIT 1"""
    ).fetchone()
    assert tuple(row[:6]) == (2, 3, 4, 3, 0.1, 1234)
    assert len(row[6]) == 64
    ledger.close()

    assert main(argv, wrapper_factory=FakeWrapper) == 0
    capsys.readouterr()
    assert FakeWrapper.extracted == 12

    first = combined[0]
    shard = Path(first["cache_root"]) / first["shard_path"]
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        main(argv, wrapper_factory=FakeWrapper)


def test_rematerialize_completed_batch_repairs_stale_manifest_without_model_load(
    tmp_path, capsys
) -> None:
    class FakeWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            return None

        def extract_prefill(self, request):
            return PrefillResult(
                request=request,
                trajectory=np.ones((2, 3), dtype=np.float32),
                token_count=4,
                t0_token_index=3,
                provenance={"elapsed_seconds": 0.1, "peak_gpu_memory_bytes": 1234},
            )

        def close(self):
            return None

    args = _args(tmp_path, question="judge emotion")
    argv = [
        "--manifest",
        str(args.manifest),
        "--prompt-set",
        str(args.prompt_set),
        "--prompt-variable",
        "question=judge emotion",
        "--model-path",
        str(args.model_path),
        "--output-root",
        str(args.output_root),
        "--device",
        "cpu",
    ]
    assert main(argv, wrapper_factory=FakeWrapper) == 0
    capsys.readouterr()

    expected = [
        json.loads(line)
        for line in (args.output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected_identities = {
        (
            row["sample_id"],
            row["model_key"],
            row["protocol"],
            row["prompt_set_key"],
            row["prompt_id"],
            row["condition"],
        )
        for row in expected
    }
    expected_checksums = {row["checksum"] for row in expected}
    (args.output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in expected[:-2]), encoding="utf-8"
    )

    report = rematerialize_completed_batch(args.output_root)

    repaired = [
        json.loads(line)
        for line in (args.output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    repaired_identities = {
        (
            row["sample_id"],
            row["model_key"],
            row["protocol"],
            row["prompt_set_key"],
            row["prompt_id"],
            row["condition"],
        )
        for row in repaired
    }
    assert report["manifest_rows"] == 12
    assert report["summary"] == {
        "total": 12,
        "pending": 0,
        "running": 0,
        "completed": 12,
        "failed": 0,
    }
    assert repaired_identities == expected_identities
    assert {row["checksum"] for row in repaired} == expected_checksums
    assert repaired == expected


def test_ledger_rejects_signature_changes(tmp_path) -> None:
    first = build_batch_plan(_args(tmp_path, question="first question"))
    ledger = BatchLedger(tmp_path / "output" / "batch_state.sqlite3")
    ledger.prepare(first, retry_failed=False)
    second = build_batch_plan(_args(tmp_path, question="different question"))

    with pytest.raises(ValueError, match="signature does not match"):
        ledger.prepare(second, retry_failed=False)
    ledger.close()
