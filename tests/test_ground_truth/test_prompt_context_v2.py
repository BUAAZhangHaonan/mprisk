from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mprisk.ground_truth.deepseek_gt import (
    Ledger,
    load_config,
    prepare_tasks,
    run_batch,
    validate_gt_content,
    verify_outputs,
)
from mprisk.ground_truth.prompt_context_v2 import (
    CONTEXT_PRIORITY,
    PILOT_SAMPLE_IDS,
    build_prompt_context_v2_pilot,
    resolve_context,
)

ROOT = Path(__file__).resolve().parents[2]
V1_CONFIG = ROOT / "configs/ground_truth/deepseek_gt_v1.yaml"
V2_CONFIG = ROOT / "configs/ground_truth/deepseek_gt_prompt_context_v2_pilot.yaml"
V2_MANIFEST = (
    ROOT
    / "data/frozen/generated_round1_v1/ground_truth_inputs/prompt_context_v2_pilot.jsonl"
)


class FakeClient:
    def __init__(self) -> None:
        self.tasks: list[Any] = []

    async def complete(self, task: Any) -> dict[str, Any]:
        self.tasks.append(task)
        description = (
            "The person's guarded surface contrasts with their true fear, supported by the "
            "tense posture and uneasy words."
            if task.data_type == "A"
            else "The person's expression, posture, and words consistently communicate clear anger."
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
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def test_context_priority_is_setting_then_natural_trigger_then_ltx2_prompt() -> None:
    assert CONTEXT_PRIORITY == ("setting", "trigger", "ltx2_prompt")
    assert resolve_context(
        {"setting": " recorded setting ", "trigger": "natural trigger", "ltx2_prompt": "raw"}
    ) == ("recorded setting", "setting")
    assert resolve_context(
        {"setting": None, "trigger": " natural trigger ", "ltx2_prompt": "raw"}
    ) == ("natural trigger", "trigger")
    assert resolve_context(
        {"setting": None, "trigger": "T3", "ltx2_prompt": " raw prompt "}
    ) == ("raw prompt", "ltx2_prompt")


def test_full_archive_recovers_490_raw_prompts_and_selects_exact_pilot() -> None:
    rows, provenance = build_prompt_context_v2_pilot(ROOT)

    assert provenance["context_source_counts"] == {
        "setting": 126,
        "trigger": 36,
        "ltx2_prompt": 490,
    }
    assert [row["sample_id"] for row in rows] == list(PILOT_SAMPLE_IDS)
    assert Counter((row["data_type"], row["protocol"]) for row in rows) == {
        ("A", "VT"): 2,
        ("A", "VA"): 2,
        ("C", "VT"): 2,
        ("C", "VA"): 2,
    }
    assert all(row["context_source"] == "ltx2_prompt" for row in rows)


def test_frozen_v2_manifest_is_strict_hashed_and_reproducible() -> None:
    expected_rows, provenance = build_prompt_context_v2_pilot(ROOT)
    actual_rows = _read_jsonl(V2_MANIFEST)

    assert actual_rows == expected_rows
    assert len(actual_rows) == 8
    expected_fields = {
        "schema_name",
        "protocol_version",
        "sample_id",
        "source_archive",
        "data_type",
        "protocol",
        "archetype",
        "dialogue",
        "context_text",
        "context_source",
        "surface_emotion",
        "media",
        "source_assignment",
        "source_row_sha256",
    }
    assert all(set(row) == expected_fields for row in actual_rows)
    for row in actual_rows:
        assert Path(row["media"]["path"]).is_file()
        assert _sha256(Path(row["media"]["path"])) == row["media"]["sha256"]
        assert row["source_assignment"]["source_row_sha256"] == row["source_row_sha256"]
    assert provenance["pilot_count"] == 8


def test_v2_tasks_are_manifest_sized_signed_and_do_not_leak_labels(tmp_path: Path) -> None:
    config = load_config(V2_CONFIG)
    tasks = prepare_tasks(ROOT, config)

    assert len(tasks) == config.expected_count == 8
    for task in tasks:
        assert set(task.model_input) == {"archetype", "dialogue", "context", "surface_emotion"}
        serialized = json.dumps(task.model_input, ensure_ascii=False).lower()
        for forbidden in (
            "gt_description",
            "gt_emotion",
            "sample_type",
            "protocol",
            "context_source",
            "media",
            "m1",
            "m2",
            "m12",
        ):
            assert forbidden not in serialized
        assert task.ledger_signature == {
            "schema_name": config.schema_name,
            "run_id": config.run_id,
            "protocol_version": config.protocol_version,
            "input_manifest_sha256": config.input_manifest_sha256,
            "expected_count": config.expected_count,
        }

    ledger = Ledger(tmp_path / "ledger.sqlite3")
    try:
        ledger.prepare(tasks)
        changed = [
            replace(
                task,
                ledger_signature={**task.ledger_signature, "expected_count": 9},
            )
            for task in tasks
        ]
        try:
            ledger.prepare(changed)
        except ValueError as exc:
            assert "signature" in str(exc).lower()
        else:
            raise AssertionError("Changed v2 run signature must be rejected")
    finally:
        ledger.close()


def test_v2_mock_run_is_exactly_eight_one_sentence_rows(tmp_path: Path) -> None:
    config_path = _v2_test_config(tmp_path)
    client = FakeClient()

    result = asyncio.run(
        run_batch(
            repo_root=ROOT,
            config_path=config_path,
            mode="pilot",
            client=client,
            sleep=_no_sleep,
        )
    )

    assert (result.total, result.completed, result.failed, result.pending) == (8, 8, 0, 0)
    assert len(client.tasks) == 8
    verified = verify_outputs(ROOT, config_path, require_complete=True)
    assert (verified.total, verified.completed, verified.failed, verified.pending) == (8, 8, 0, 0)
    manifest = _read_jsonl(result.output_root / "gt_manifest.jsonl")
    assert len(manifest) == 8
    for row in manifest:
        assert set(row) == set(_manifest_by_id()[row["sample_id"]]) | {"GT_DESCRIPTION"}
        validate_gt_content(
            json.dumps({"GT_DESCRIPTION": row["GT_DESCRIPTION"]}),
            min_words=6,
            max_words=80,
        )


def test_invalid_model_response_is_preserved_in_attempt_evidence(tmp_path: Path) -> None:
    config_path = _v2_test_config(tmp_path)

    result = asyncio.run(
        run_batch(
            repo_root=ROOT,
            config_path=config_path,
            mode="pilot",
            client=InvalidClient(),
            sleep=_no_sleep,
        )
    )

    assert (result.completed, result.failed) == (0, 8)
    attempts = _read_jsonl(result.output_root / "attempts.jsonl")
    assert len(attempts) == 8
    for attempt in attempts:
        response = json.loads(attempt["response_json"])
        assert response["finish_reason"] == "stop"
        assert response["content"] == '{"GT_DESCRIPTION": "The person sounds angry!"}'


def test_v2_accepts_quoted_dialogue_punctuation_but_v1_remains_strict() -> None:
    content = json.dumps(
        {
            "GT_DESCRIPTION": (
                "The woman expresses anger through her tense posture and the words "
                "'I cannot believe this happened!' in the workshop."
            )
        }
    )

    with pytest.raises(ValueError, match="declarative sentence"):
        validate_gt_content(content, min_words=6, max_words=80)
    assert validate_gt_content(
        content,
        min_words=6,
        max_words=80,
        allow_quoted_terminal_marks=True,
    ).endswith("workshop.")


def test_v1_task_count_and_model_input_contract_remain_unchanged() -> None:
    tasks = prepare_tasks(ROOT, load_config(V1_CONFIG))

    assert len(tasks) == 162
    assert len([task for task in tasks if task.ledger_signature is None]) == 162
    assert all(
        set(task.model_input) == {
            "archetype",
            "trigger_context",
            "dialogue",
            "surface_emotion",
        }
        for task in tasks
    )


def test_v2_aligned_prompt_prevents_dialogue_punctuation_from_breaking_one_sentence() -> None:
    prompt = (
        ROOT / "configs/prompts/ground_truth/c_aligned_gt_prompt_context_v2.txt"
    ).read_text(encoding="utf-8")

    assert "Paraphrase dialogue rather than quoting it" in prompt
    assert "do not use question marks or exclamation marks anywhere" in prompt


def _v2_test_config(tmp_path: Path) -> Path:
    import yaml

    payload = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    payload["output_root"] = (tmp_path / "deepseek_gt_prompt_context_v2_pilot").as_posix()
    payload["env_file"] = (tmp_path / "unused.env").as_posix()
    path = tmp_path / "deepseek_gt_prompt_context_v2_pilot.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _manifest_by_id() -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in _read_jsonl(V2_MANIFEST)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def _no_sleep(_: float) -> None:
    return None
