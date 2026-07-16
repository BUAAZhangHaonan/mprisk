from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
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
from mprisk.ground_truth.providers.base import (
    GTDescriptionProviderRequest,
    GTDescriptionProviderResponse,
)
from mprisk.ground_truth.providers.deepseek import (
    DeepSeekProvider,
    DeepSeekProviderSettings,
    load_api_key,
)
from mprisk.ground_truth.providers.registry import (
    get_provider,
    validate_provider_settings,
)


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
        "schema_name": "mprisk_gt_description_generation_config_v3",
        "run_id": "test_gt_description_generation_v3",
        "provider_key": "deepseek",
        "gt_generator_model": "deepseek-v4-flash",
        "provider_settings": {
            "api_url": "https://api.deepseek.com/chat/completions",
            "env_file": ".env",
            "api_key_env": "DEEPSEEK_API_KEY",
            "temperature": 0,
            "max_tokens": 128,
            "thinking": "disabled",
            "request_timeout_seconds": 10.0,
        },
        "concurrency": 2,
        "retry_delays_seconds": [0.0],
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


class FakeProvider:
    def __init__(self) -> None:
        self.tasks: list[Any] = []

    async def complete(
        self, request: GTDescriptionProviderRequest
    ) -> GTDescriptionProviderResponse:
        self.tasks.append(request)
        description = (
            "The person feels guarded and distressed despite the calm outward presentation."
            if request.system_prompt == "Conflict GT prompt"
            else "The person consistently expresses a calm and settled emotional state."
        )
        return GTDescriptionProviderResponse(
            response_id="fake-response",
            response_model=request.model,
            finish_reason="stop",
            content=json.dumps({"GT_DESCRIPTION": description}),
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            provider_metadata={},
        )

    async def close(self) -> None:
        raise AssertionError("Injected providers are not owned by the task")


class InvalidProvider:
    async def complete(
        self, request: GTDescriptionProviderRequest
    ) -> GTDescriptionProviderResponse:
        return GTDescriptionProviderResponse(
            response_id="invalid-response",
            response_model=request.model,
            finish_reason="stop",
            content=json.dumps({"GT_DESCRIPTION": "The person sounds angry!"}),
            usage={},
            provider_metadata={},
        )

    async def close(self) -> None:
        raise AssertionError("Injected providers are not owned by the task")


async def _no_sleep(_: float) -> None:
    return None


def test_config_and_tasks_use_canonical_gt_description_names(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    tasks = prepare_tasks(tmp_path, config)

    assert config.gt_generator_model == "deepseek-v4-flash"
    assert config.provider_key == "deepseek"
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
    provider = FakeProvider()
    result = asyncio.run(
        run_gt_description_generation(
            repo_root=tmp_path,
            config_path=config_path,
            provider=provider,
            sleep=_no_sleep,
        )
    )

    assert (result.total, result.completed, result.failed, result.pending) == (2, 2, 0, 0)
    assert len(provider.tasks) == 2
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
        assert set(row) == set(expected) | {"run_id", "GT_DESCRIPTION"}
        assert row["schema_name"] == "mprisk_gt_description_v1"
        assert row["gt_input_schema_version"] == "gt_annotation_input_v1"
        assert row["run_id"] == "test_gt_description_generation_v3"
        for key in set(expected) - {"schema_name"}:
            assert row[key] == expected[key]
    provenance = json.loads(
        (result.output_root / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["gt_description_schema_name"] == "mprisk_gt_description_v1"


def test_invalid_response_is_preserved_as_failure(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = asyncio.run(
        run_gt_description_generation(
            repo_root=tmp_path,
            config_path=config_path,
            provider=InvalidProvider(),
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
            provider=InvalidProvider(),
            sleep=_no_sleep,
        )
    )
    provider = FakeProvider()
    unchanged = asyncio.run(
        run_gt_description_generation(
            repo_root=tmp_path,
            config_path=config_path,
            provider=provider,
            sleep=_no_sleep,
        )
    )
    assert unchanged.failed == 2
    assert provider.tasks == []
    completed = asyncio.run(
        run_gt_description_generation(
            repo_root=tmp_path,
            config_path=config_path,
            retry_failed=True,
            provider=provider,
            sleep=_no_sleep,
        )
    )
    assert (completed.completed, completed.failed) == (2, 0)


def test_api_key_has_no_alternate_provider_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(_write_config(tmp_path))
    settings = DeepSeekProviderSettings.model_validate(config.provider_settings)
    settings = settings.model_copy(update={"env_file": tmp_path / settings.env_file})
    settings.env_file.write_text("OTHER_API_KEY=forbidden\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_api_key(settings)


def test_unknown_provider_hard_fails_without_fallback(tmp_path: Path) -> None:
    config_payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    config_payload["provider_key"] = "unknown-provider"
    config_path = tmp_path / "unknown.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown GT Description provider"):
        load_config(config_path)
    with pytest.raises(ValueError, match="Unknown GT Description provider"):
        get_provider("unknown-provider", "model", {})
    with pytest.raises(ValueError, match="Unknown GT Description provider"):
        get_provider("DeepSeek", "model", config_payload["provider_settings"])


def test_deepseek_adapter_rejects_unknown_or_invalid_settings(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    with pytest.raises(ValueError, match="extra_forbidden"):
        validate_provider_settings(
            "deepseek", {**config.provider_settings, "fallback_provider": "forbidden"}
        )
    with pytest.raises(ValueError, match="thinking"):
        validate_provider_settings(
            "deepseek", {**config.provider_settings, "thinking": "enabled"}
        )


def test_deepseek_adapter_builds_and_validates_exact_request(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    settings = DeepSeekProviderSettings.model_validate(config.provider_settings)
    observed: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["authorization"] = request.headers["Authorization"]
        observed["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": config.gt_generator_model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "GT_DESCRIPTION": (
                                        "The person feels calm and secure throughout the exchange."
                                    )
                                }
                            ),
                            "reasoning_content": None,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    async def exercise() -> GTDescriptionProviderResponse:
        provider = DeepSeekProvider(
            config.gt_generator_model,
            settings,
            "secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await provider.complete(
                GTDescriptionProviderRequest(
                    model=config.gt_generator_model,
                    system_prompt="System prompt",
                    model_input={"archetype": {"id": "x"}},
                )
            )
        finally:
            await provider.close()

    response = asyncio.run(exercise())
    assert response.response_id == "response-1"
    assert observed["authorization"] == "Bearer secret"
    assert observed["body"] == {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": '{"archetype":{"id":"x"}}'},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
    }


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
