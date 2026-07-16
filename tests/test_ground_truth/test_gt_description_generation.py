from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mprisk.ground_truth.description_generation import (
    GTDescriptionGenerationLedger,
    GTDescriptionValidationError,
    load_config,
    prepare_tasks,
    run_gt_description_generation,
    validate_gt_description_content,
    verify_gt_description_generation,
)
from mprisk.ground_truth.providers.deepseek import load_api_key


def _annotation_row(sample_id: str, sample_type: str) -> dict[str, Any]:
    source_code = "A" if sample_type == "Conflict" else "C"
    return {
        "schema_name": "mprisk_gt_annotation_input_v1",
        "gt_input_schema_version": "gt_annotation_input_v1",
        "sample_id": sample_id,
        "sample_type": sample_type,
        "protocol": "VT",
        "archetype": {
            "id": f"archetype-{source_code}",
            "name": "guarded affect",
            "canonical_meaning": "The person's outward display does not define the full affect.",
        },
        "dialogue": "I am doing fine today.",
        "scenario_context": "The person is discussing a stressful event.",
        "scenario_context_source": "source_prompt",
        "surface_emotion": "calm",
        "media": {"path": "/tmp/source.mp4", "sha256": "a" * 64},
        "source_provenance": {
            "source_archive": f"archive-{source_code}",
            "source_class_code": source_code,
            "source_row_sha256": "b" * 64,
            "source_assignment": {
                "path": "/tmp/assignment.jsonl",
                "schema_name": "assignment-v1",
                "dictionary_id": "dictionary-v1",
                "assignment_source": "canonical",
                "source_row_sha256": "b" * 64,
                "assignment_sha256": "c" * 64,
            },
        },
    }


def _write_config(tmp_path: Path) -> Path:
    rows = [
        _annotation_row("conflict-sample", "Conflict"),
        _annotation_row("aligned-sample", "Aligned"),
    ]
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (tmp_path / "conflict.txt").write_text("Conflict GT prompt", encoding="utf-8")
    (tmp_path / "aligned.txt").write_text("Aligned GT prompt", encoding="utf-8")
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=test\n", encoding="utf-8")
    config = {
        "schema_name": "mprisk_gt_description_generation_config_v1",
        "run_id": "test_gt_description_generation_v1",
        "gt_generator_model": "deepseek-v4-flash",
        "api_url": "https://api.deepseek.com/chat/completions",
        "env_file": ".env",
        "api_key_variable": "DEEPSEEK_API_KEY",
        "temperature": 0,
        "max_tokens": 128,
        "thinking": "disabled",
        "concurrency": 2,
        "retry_delays_seconds": [0.0],
        "request_timeout_seconds": 10.0,
        "min_words": 6,
        "max_words": 80,
        "gt_input_schema_version": "gt_annotation_input_v1",
        "input_manifest": "inputs.jsonl",
        "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "expected_count": 2,
        "output_root": "outputs/new_gt",
        "conflict_prompt_path": "conflict.txt",
        "aligned_prompt_path": "aligned.txt",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class FakeClient:
    def __init__(self) -> None:
        self.tasks: list[Any] = []

    async def complete(self, task: Any) -> dict[str, Any]:
        self.tasks.append(task)
        description = (
            "The person feels guarded and distressed despite the calm outward presentation."
            if task.sample_type == "Conflict"
            else "The person consistently expresses a calm and settled emotional state."
        )
        return {
            "response_id": f"response-{task.sample_id}",
            "response_model": "deepseek-v4-flash",
            "system_fingerprint": None,
            "finish_reason": "stop",
            "content": json.dumps({"GT_DESCRIPTION": description}),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class InvalidClient:
    async def complete(self, task: Any) -> dict[str, Any]:
        return {
            "response_id": f"invalid-{task.sample_id}",
            "response_model": "deepseek-v4-flash",
            "system_fingerprint": None,
            "finish_reason": "stop",
            "content": json.dumps({"GT_DESCRIPTION": "The person sounds angry!"}),
            "usage": {},
        }


async def _no_sleep(_: float) -> None:
    return None


def test_config_and_tasks_use_canonical_gt_description_names(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    tasks = prepare_tasks(tmp_path, config)

    assert config.gt_generator_model == "deepseek-v4-flash"
    assert config.conflict_prompt_path == Path("conflict.txt")
    assert config.aligned_prompt_path == Path("aligned.txt")
    assert len(tasks) == 2
    for task in tasks:
        assert set(task.model_input) == {
            "archetype",
            "dialogue",
            "scenario_context",
            "surface_emotion",
        }
        serialized = json.dumps(task.model_input, ensure_ascii=False)
        for forbidden in ("source_class_code", "sample_type", "protocol", "media"):
            assert forbidden not in serialized
        assert task.ledger_signature["gt_input_schema_version"] == (
            "gt_annotation_input_v1"
        )


def test_legacy_config_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text("schema_name: mprisk_deepseek_gt_config_v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_config(path)


def test_ledger_rejects_changed_generation_identity(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    tasks = prepare_tasks(tmp_path, config)
    ledger = GTDescriptionGenerationLedger(tmp_path / "ledger.sqlite3")
    try:
        ledger.prepare(tasks)
        changed = [
            type(task)(
                **{
                    **task.__dict__,
                    "ledger_signature": {
                        **task.ledger_signature,
                        "gt_generator_model": "different-model",
                    },
                }
            )
            for task in tasks
        ]
        with pytest.raises(ValueError, match="signature"):
            ledger.prepare(changed)
    finally:
        ledger.close()


def test_mock_generation_adds_only_gt_description_and_verifies(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    client = FakeClient()
    result = asyncio.run(
        run_gt_description_generation(
            repo_root=tmp_path,
            config_path=config_path,
            client=client,
            sleep=_no_sleep,
        )
    )

    assert (result.total, result.completed, result.failed, result.pending) == (2, 2, 0, 0)
    assert len(client.tasks) == 2
    verified = verify_gt_description_generation(
        tmp_path, config_path, require_complete=True
    )
    assert verified.completed == 2
    original = {
        row["sample_id"]: row
        for row in _read_jsonl(tmp_path / "inputs.jsonl")
    }
    generated = _read_jsonl(result.output_root / "gt_manifest.jsonl")
    for row in generated:
        expected = original[row["sample_id"]]
        assert set(row) == set(expected) | {"GT_DESCRIPTION"}
        assert {key: row[key] for key in expected} == expected


def test_invalid_response_is_preserved_as_failure(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = asyncio.run(
        run_gt_description_generation(
            repo_root=tmp_path,
            config_path=config_path,
            client=InvalidClient(),
            sleep=_no_sleep,
        )
    )
    assert (result.completed, result.failed) == (0, 2)
    failures = _read_jsonl(result.output_root / "failures.jsonl")
    attempts = _read_jsonl(result.output_root / "attempts.jsonl")
    assert len(failures) == len(attempts) == 2
    assert {row["error_type"] for row in failures} == {"GTDescriptionValidationError"}


def test_failed_rows_require_explicit_retry(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    asyncio.run(
        run_gt_description_generation(
            repo_root=tmp_path,
            config_path=config_path,
            client=InvalidClient(),
            sleep=_no_sleep,
        )
    )
    client = FakeClient()
    unchanged = asyncio.run(
        run_gt_description_generation(
            repo_root=tmp_path,
            config_path=config_path,
            client=client,
            sleep=_no_sleep,
        )
    )
    assert unchanged.failed == 2
    assert client.tasks == []
    completed = asyncio.run(
        run_gt_description_generation(
            repo_root=tmp_path,
            config_path=config_path,
            retry_failed=True,
            client=client,
            sleep=_no_sleep,
        )
    )
    assert (completed.completed, completed.failed) == (2, 0)


def test_api_key_has_no_alternate_provider_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(_write_config(tmp_path))
    config = config.model_copy(update={"env_file": tmp_path / config.env_file})
    config.env_file.write_text("OTHER_API_KEY=forbidden\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_api_key(config)


def test_gt_description_validation_accepts_exact_contract() -> None:
    value = "The person feels worried despite maintaining a calm outward expression."
    assert (
        validate_gt_description_content(
            json.dumps({"GT_DESCRIPTION": value}), min_words=6, max_words=80
        )
        == value
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"GT_DESCRIPTION": "Too short."}),
        json.dumps({"GT_DESCRIPTION": "This sentence is valid and complete.", "extra": 1}),
        json.dumps({"GT_DESCRIPTION": "This is one sentence. This is another sentence."}),
    ],
)
def test_gt_description_validation_rejects_non_contract_content(payload: str) -> None:
    with pytest.raises(GTDescriptionValidationError):
        validate_gt_description_content(payload, min_words=6, max_words=80)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
