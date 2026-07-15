from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from mprisk.ground_truth.deepseek_gt import (
    GTValidationError,
    Ledger,
    PermanentAPIError,
    load_api_key,
    load_config,
    prepare_tasks,
    run_batch,
    select_pilot,
    validate_gt_content,
    verify_outputs,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs/ground_truth/deepseek_gt_v1.yaml"


class FakeClient:
    def __init__(self, *, fail_once: str | None = None):
        self.fail_once = fail_once
        self.failed: set[str] = set()
        self.calls: list[str] = []

    async def complete(self, task: Any) -> dict[str, Any]:
        self.calls.append(task.sample_id)
        if task.sample_id == self.fail_once and task.sample_id not in self.failed:
            self.failed.add(task.sample_id)
            raise PermanentAPIError("synthetic permanent failure")
        description = (
            "The person clearly experiences the emotion described by the supplied context."
        )
        return {
            "response_id": f"response-{task.sample_id}",
            "response_model": "deepseek-v4-flash",
            "system_fingerprint": None,
            "finish_reason": "stop",
            "content": json.dumps({"GT_DESCRIPTION": description}),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def _write_test_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload["output_root"] = (tmp_path / "deepseek_gt_v1").as_posix()
    payload["env_file"] = (tmp_path / "unused.env").as_posix()
    path = tmp_path / "deepseek_gt_v1.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_prepare_tasks_uses_exact_eligible_contract_without_gt_leakage() -> None:
    tasks = prepare_tasks(ROOT, load_config(BASE_CONFIG))

    assert len(tasks) == 162
    assert len({task.sample_id for task in tasks}) == 162
    assert Counter(task.source_archive for task in tasks) == {
        "accept_a_svt": 64,
        "accept_a_va": 8,
        "accept_c_svt": 77,
        "accept_c_va": 13,
    }
    assert len(select_pilot(tasks)) == 16
    for task in tasks:
        assert set(task.model_input) == {
            "archetype",
            "trigger_context",
            "dialogue",
            "surface_emotion",
        }
        assert "gt_emotion" not in json.dumps(task.model_input).lower()


def test_ledger_insert_matches_schema_and_failed_rows_require_opt_in(tmp_path: Path) -> None:
    tasks = prepare_tasks(ROOT, load_config(BASE_CONFIG))[:2]
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    try:
        ledger.prepare(tasks)
        assert len(ledger.rows()) == 2
        ledger.fail(tasks[0].sample_id, PermanentAPIError("failed"))

        selected = {task.sample_id for task in tasks}
        assert ledger.pending_ids(selected) == [tasks[1].sample_id]
        assert ledger.pending_ids(selected, include_failed=True) == [
            tasks[0].sample_id,
            tasks[1].sample_id,
        ]

        with pytest.raises(ValueError, match="Ledger task set mismatch"):
            ledger.prepare(tasks[:1])
    finally:
        ledger.close()


def test_ground_truth_requires_deepseek_key_without_glm_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_test_config(tmp_path)
    config = load_config(config_path)
    config.env_file.write_text("GLM_API_KEY=forbidden\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("GLM_API_KEY", "forbidden")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_api_key(config)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "accepted")
    assert load_api_key(config) == "accepted"


def test_pilot_resume_full_export_and_explicit_failed_retry(tmp_path: Path) -> None:
    config_path = _write_test_config(tmp_path)
    tasks = prepare_tasks(ROOT, load_config(config_path))
    failed_id = select_pilot(tasks)[0].sample_id

    first = FakeClient(fail_once=failed_id)
    pilot = asyncio.run(
        run_batch(
            repo_root=ROOT,
            config_path=config_path,
            mode="pilot",
            client=first,
            sleep=_no_sleep,
        )
    )
    assert (pilot.completed, pilot.failed, pilot.pending) == (15, 1, 146)
    assert len(first.calls) == 16

    default_resume = FakeClient()
    unchanged = asyncio.run(
        run_batch(
            repo_root=ROOT,
            config_path=config_path,
            mode="pilot",
            client=default_resume,
            sleep=_no_sleep,
        )
    )
    assert (unchanged.completed, unchanged.failed, unchanged.pending) == (15, 1, 146)
    assert default_resume.calls == []

    explicit_retry = FakeClient()
    retried = asyncio.run(
        run_batch(
            repo_root=ROOT,
            config_path=config_path,
            mode="pilot",
            retry_failed=True,
            client=explicit_retry,
            sleep=_no_sleep,
        )
    )
    assert (retried.completed, retried.failed, retried.pending) == (16, 0, 146)
    assert explicit_retry.calls == [failed_id]

    full_client = FakeClient()
    full = asyncio.run(
        run_batch(
            repo_root=ROOT,
            config_path=config_path,
            mode="full",
            client=full_client,
            sleep=_no_sleep,
        )
    )
    assert (full.total, full.completed, full.failed, full.pending) == (162, 162, 0, 0)
    assert len(full_client.calls) == 146

    verified = verify_outputs(ROOT, config_path, require_complete=True)
    assert (verified.completed, verified.failed, verified.pending) == (162, 0, 0)

    output_root = Path(load_config(config_path).output_root)
    eligible = {
        row["sample_id"]: row
        for row in _read_jsonl(ROOT / "data/frozen/generated_round1_v1/gt_eligible.jsonl")
    }
    manifest = _read_jsonl(output_root / "gt_manifest.jsonl")
    sidecar = _read_jsonl(output_root / "review_status.jsonl")
    assert len(manifest) == len(sidecar) == 162
    for row in manifest:
        expected = eligible[row["sample_id"]]
        assert set(row) == set(expected) | {"GT_DESCRIPTION"}
        assert {key: row[key] for key in expected} == expected


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"GT_DESCRIPTION": "Too short."}),
        json.dumps({"GT_DESCRIPTION": "This sentence is valid and complete.", "extra": 1}),
        json.dumps({"GT_DESCRIPTION": "This is one sentence. This is another sentence."}),
    ],
)
def test_gt_content_rejects_non_contract_responses(payload: str) -> None:
    with pytest.raises(GTValidationError):
        validate_gt_content(payload, min_words=6, max_words=60)


async def _no_sleep(_: float) -> None:
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
