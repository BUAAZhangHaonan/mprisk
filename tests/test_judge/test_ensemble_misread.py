from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mprisk.judge.ensemble_misread import (
    EnsembleLedger,
    EnsembleMisreadConfig,
    HttpResponseReceipt,
    _execute_pending,
    build_flash_calls,
    build_pro_call,
    build_sample_tasks,
    dry_run,
    offline_replay,
    run_ensemble,
)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _receipt(call, raw: str, request_id: str, number: int) -> HttpResponseReceipt:
    envelope = {
        "id": request_id,
        "model": call.model,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": f"tool-{number}",
                            "type": "function",
                            "function": {
                                "name": "submit_misread_judgment",
                                "arguments": raw,
                            },
                        }
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "prompt_cache_hit_tokens": 4,
            "prompt_cache_miss_tokens": 6,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    body = json.dumps(envelope, sort_keys=True).encode()
    return HttpResponseReceipt(
        status_code=200,
        response_body=body,
        response_sha256=hashlib.sha256(body).hexdigest(),
        provider_request_id=request_id,
        received_at="2026-08-04T00:00:00+00:00",
    )


def _config(tmp_path: Path) -> EnsembleMisreadConfig:
    gt = tmp_path / "gt.jsonl"
    diagnostic = tmp_path / "diagnostic.jsonl"
    coverage = tmp_path / "gt_coverage.json"
    forbidden = tmp_path / "v2_started_calls.jsonl"
    prompt_sha256 = "a" * 64
    generation_policy_sha256 = "b" * 64
    request_protocol_signature_sha256 = "c" * 64
    _jsonl(
        gt,
        [
            {"sample_id": "a", "GT_DESCRIPTION": "The overall emotional valence is positive."},
            {"sample_id": "b", "GT_DESCRIPTION": "The overall emotional valence is negative."},
        ],
    )
    _jsonl(
        diagnostic,
        [
            {
                "schema_name": "mprisk_diagnostic_affect_description_v3",
                "run_id": "diag",
                "sample_id": sample_id,
                "subject_model_key": "model",
                "protocol": "VT",
                "condition": "M12",
                "split": "train",
                "DIAGNOSTIC_AFFECT_DESCRIPTION": description,
                "prompt_sha256": prompt_sha256,
                "generation_policy_sha256": generation_policy_sha256,
                "request_protocol_signature_sha256": request_protocol_signature_sha256,
            }
            for sample_id, description in (
                ("a", "The person appears happy."),
                ("b", "The person appears calm."),
            )
        ],
    )
    sample_ids = ["a", "b"]
    sample_digest = hashlib.sha256(
        json.dumps(
            sample_ids,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    coverage.write_text(
        json.dumps(
            {
                "schema_name": "mprisk_target_gt_coverage_v1",
                "status": "PASS",
                "protocols": {
                    "VT": {
                        "complete": True,
                        "expected_rows": 2,
                        "observed_rows": 2,
                        "unique_sample_ids": 2,
                        "blank_sample_ids": 0,
                        "duplicate_sample_ids": 0,
                        "protocol_mismatches": 0,
                        "nonempty_gt_descriptions": 2,
                        "missing_gt_descriptions": 0,
                        "sample_id_set_sha256": sample_digest,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _jsonl(forbidden, [{"call_id": "retired-paid-call"}])
    return EnsembleMisreadConfig(
        schema_name="mprisk_ensemble_misread_judgment_config_v3",
        run_id="run",
        status="ready",
        subject_model_key="model",
        protocol="VT",
        split="train",
        api_url="https://api.deepseek.com/beta/chat/completions",
        temperature=0,
        thinking="disabled",
        max_tokens=256,
        confidence_threshold=0.5,
        flash_model="deepseek-v4-flash",
        pro_model="deepseek-v4-pro",
        flash_replicates=3,
        gt_coverage_receipt_path=coverage,
        gt_description_manifest_path=gt,
        diagnostic_affect_description_manifest_path=diagnostic,
        diagnostic_run_id="diag",
        diagnostic_manifest_sha256=hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
        diagnostic_prompt_sha256=prompt_sha256,
        diagnostic_generation_policy_sha256=generation_policy_sha256,
        diagnostic_request_protocol_signature_sha256=request_protocol_signature_sha256,
        output_root=tmp_path / "out",
        forbidden_started_calls_path=forbidden,
        forbidden_started_calls_sha256=hashlib.sha256(forbidden.read_bytes()).hexdigest(),
        request_timeout_seconds=1.0,
        max_concurrency=2,
        pricing={
            "deepseek-v4-flash": {
                "input_usd_per_million": None,
                "output_usd_per_million": None,
            },
            "deepseek-v4-pro": {
                "input_usd_per_million": None,
                "output_usd_per_million": None,
            },
        },
    )


class FakeClient:
    def __init__(self, *, request_offset: int = 0) -> None:
        self.calls = 0
        self.request_offset = request_offset

    async def complete(self, call):
        self.calls += 1
        if call.role == "pro":
            result = {
                "decision": "UNCERTAIN",
                "confidence": 0.4,
                "rationale": "The evidence remains too ambiguous to decide.",
            }
        elif call.sample_id == "a":
            result = {
                "decision": "NON_MISREAD",
                "confidence": 0.9,
                "rationale": "The positive affect agrees with the reference.",
            }
        else:
            decision = "MISREAD" if call.slot < 2 else "NON_MISREAD"
            result = {
                "decision": decision,
                "confidence": 0.9,
                "rationale": "The preliminary comparison yields this decision.",
            }
        raw = json.dumps(result)
        number = self.request_offset + self.calls
        return _receipt(call, raw, f"request-{number}", number)


def test_dry_run_never_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = dry_run(_config(tmp_path))
    assert result["sample_count"] == 2
    assert result["flash_request_count"] == 6
    assert result["pro_request_upper_bound"] == 2
    assert result["max_api_request_count"] == 8
    assert result["planned_call_id_count"] == 6
    assert result["unique_planned_call_id_count"] == 6
    assert result["unique_request_payload_sha256_count"] == 2
    assert result["api_requests_issued"] == 0
    assert result["api_key_accessed"] is False


def test_judgment_rejects_changed_or_old_diagnostic_binding(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.diagnostic_affect_description_manifest_path.write_text(
        config.diagnostic_affect_description_manifest_path.read_text(encoding="utf-8")
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Diagnostic manifest SHA-256"):
        dry_run(config)


def test_dry_run_and_execute_block_non_pass_gt_coverage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    receipt = json.loads(config.gt_coverage_receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "BLOCKED"
    config.gt_coverage_receipt_path.write_text(
        json.dumps(receipt) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="coverage receipt is not PASS"):
        dry_run(config)
    client = FakeClient()
    with pytest.raises(ValueError, match="coverage receipt is not PASS"):
        asyncio.run(run_ensemble(config, client=client))
    assert client.calls == 0


def test_legacy_config_without_gt_coverage_receipt_is_rejected(tmp_path: Path) -> None:
    payload = _config(tmp_path).model_dump()
    payload.pop("gt_coverage_receipt_path")
    with pytest.raises(ValueError, match="gt_coverage_receipt_path"):
        EnsembleMisreadConfig.model_validate(payload)


def test_ensemble_is_resumable_and_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = FakeClient()
    result = asyncio.run(run_ensemble(config, client=client))
    assert client.calls == 7
    assert result["completed"] == 1
    assert result["human_review"] == 1
    assert result["unresolved"] == 0
    queue = [
        json.loads(line)
        for line in (config.output_root / "human_review_queue.jsonl").read_text().splitlines()
    ]
    assert [row["sample_id"] for row in queue] == ["b"]
    requests = [
        json.loads(line)
        for line in (config.output_root / "requests.jsonl").read_text().splitlines()
    ]
    assert len(requests) == 7
    assert all(row["request_id"] and row["response_sha256"] for row in requests)
    assert all(row["estimated_cost_usd"] is None for row in requests)
    judgments = [
        json.loads(line)
        for line in (config.output_root / "judgments.jsonl").read_text().splitlines()
    ]
    assert all(
        row["diagnostic_generation_policy_sha256"]
        == config.diagnostic_generation_policy_sha256
        for row in judgments
    )

    second = FakeClient()
    repeated = asyncio.run(run_ensemble(config, client=second))
    assert second.calls == 0
    assert repeated == result


class FailingClient:
    async def complete(self, call):
        raise RuntimeError(f"external failure for {call.call_id}")


def test_ensemble_external_failures_are_not_silent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = FailingClient()
    with pytest.raises(RuntimeError, match="non-repeatable calls"):
        asyncio.run(run_ensemble(config, client=client))
    summary_path = next((config.output_root / "audit_snapshots").glob("*/audit_summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["calls_ambiguous"] >= 1
    assert summary["calls_ambiguous"] <= config.max_concurrency
    assert summary["unresolved"] == 2
    assert not (config.output_root / "summary.json").exists()
    assert not (config.output_root / "judgments.jsonl").exists()


def test_confidence_threshold_is_frozen_at_half(tmp_path: Path) -> None:
    payload = _config(tmp_path).model_dump()
    payload["confidence_threshold"] = 0.6
    with pytest.raises(ValueError, match="frozen confidence threshold is 0.5"):
        EnsembleMisreadConfig.model_validate(payload)


def test_strict_tool_request_is_forced_and_schema_bounded(tmp_path: Path) -> None:
    config = _config(tmp_path)
    request = build_flash_calls(config, build_sample_tasks(config))[0].request

    assert request["thinking"] == {"type": "disabled"}
    assert request["max_tokens"] == 256
    assert "response_format" not in request
    assert request["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_misread_judgment"},
    }
    function = request["tools"][0]["function"]
    assert function["strict"] is True
    assert function["parameters"]["additionalProperties"] is False
    assert function["parameters"]["required"] == [
        "decision",
        "confidence",
        "rationale",
    ]
    assert function["parameters"]["properties"]["confidence"] == {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    }


class InvalidResponseClient:
    def __init__(self, raw: str, *, request_id: str = "provider-invalid") -> None:
        self.raw = raw
        self.request_id = request_id
        self.calls = 0

    async def complete(self, call):
        self.calls += 1
        return _receipt(
            call,
            self.raw,
            f"{self.request_id}-{self.calls}",
            1000 + self.calls,
        )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            '{"decision":"MISREAD","confidence":90,"rationale":"It conflicts."}',
            "confidence must be in [0,1]",
        ),
        (
            '{"decision":"MISREAD","confidence":0.9,"rationale":"It conflicts. It reverses."}',
            "one short sentence",
        ),
        ("not-json", "exact JSON"),
    ],
)
def test_invalid_response_keeps_full_receipt_and_stops_new_dispatch(
    tmp_path: Path, raw: str, message: str
) -> None:
    config = _config(tmp_path)
    client = InvalidResponseClient(raw)

    with pytest.raises(RuntimeError, match="invalid_response"):
        asyncio.run(run_ensemble(config, client=client))

    assert 1 <= client.calls <= config.max_concurrency
    db = sqlite3.connect(config.output_root / "request_ledger.sqlite3")
    db.row_factory = sqlite3.Row
    invalid = list(db.execute("SELECT * FROM calls WHERE status='invalid_response'"))
    pending = db.execute("SELECT COUNT(*) FROM calls WHERE status='pending'").fetchone()[0]
    db.close()
    assert invalid
    assert pending >= 6 - config.max_concurrency
    assert all(row["attempts"] == 1 for row in invalid)
    assert all(row["request_id"] and row["tool_call_id"] for row in invalid)
    assert all(row["response_body_base64"] and row["response_sha256"] for row in invalid)
    assert all(row["started_at"] and row["response_received_at"] for row in invalid)
    assert all(row["validated_at"] and row["terminal_at"] for row in invalid)
    assert all(row["raw_response"] == raw for row in invalid)
    assert all(message in row["error_message"] for row in invalid)
    assert not (config.output_root / "judgments.jsonl").exists()


class MalformedEnvelopeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, call):
        self.calls += 1
        body = b"not-an-api-envelope"
        return HttpResponseReceipt(
            status_code=200,
            response_body=body,
            response_sha256=hashlib.sha256(body).hexdigest(),
            provider_request_id=None,
            received_at="2026-08-04T00:00:01+00:00",
        )


def test_malformed_http_envelope_is_durable_before_parse(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = MalformedEnvelopeClient()

    with pytest.raises(RuntimeError, match="invalid_response"):
        asyncio.run(run_ensemble(config, client=client))

    db = sqlite3.connect(config.output_root / "request_ledger.sqlite3")
    db.row_factory = sqlite3.Row
    rows = list(db.execute("SELECT * FROM calls WHERE status='invalid_response'"))
    db.close()
    assert rows
    assert all(row["request_id"] is None for row in rows)
    assert all(row["response_status_code"] == 200 for row in rows)
    assert all(row["response_body_base64"] == "bm90LWFuLWFwaS1lbnZlbG9wZQ==" for row in rows)
    assert all(row["error_message"] == "API envelope is not JSON" for row in rows)
    provenance_path = next(
        (config.output_root / "audit_snapshots").glob("*/audit_provenance.json")
    )
    provenance = json.loads(provenance_path.read_text())
    for artifact in provenance["artifacts"].values():
        assert (config.output_root / artifact["path"]).is_file()


class DrainInflightClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, call):
        self.calls += 1
        number = self.calls
        if number == 1:
            await asyncio.sleep(0)
            raw = '{"decision":"MISREAD","confidence":90,"rationale":"Invalid."}'
        else:
            await asyncio.sleep(0.01)
            raw = (
                '{"decision":"MISREAD","confidence":0.9,'
                '"rationale":"The descriptions conflict."}'
            )
        return _receipt(call, raw, f"drain-provider-{number}", number)


def test_first_error_stops_claims_but_drains_inflight_receipts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = DrainInflightClient()

    with pytest.raises(RuntimeError, match="invalid_response"):
        asyncio.run(run_ensemble(config, client=client))

    db = sqlite3.connect(config.output_root / "request_ledger.sqlite3")
    counts = dict(db.execute("SELECT status,COUNT(*) FROM calls GROUP BY status"))
    db.close()
    assert client.calls == config.max_concurrency
    assert counts == {"completed": 1, "invalid_response": 1, "pending": 4}


class SlowValidClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, call):
        self.calls += 1
        await asyncio.sleep(0.01)
        raw = (
            '{"decision":"MISREAD","confidence":0.9,'
            '"rationale":"The descriptions conflict."}'
        )
        return _receipt(call, raw, f"slow-provider-{self.calls}", self.calls)


def test_claim_failure_stops_dispatch_and_drains_already_claimed_call(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    client = SlowValidClient()
    original_start = EnsembleLedger.start
    starts = 0

    def fail_second_start(self, call_id):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise RuntimeError("injected start failure")
        return original_start(self, call_id)

    monkeypatch.setattr(EnsembleLedger, "start", fail_second_start)

    with pytest.raises(RuntimeError, match="workers failed after draining"):
        asyncio.run(run_ensemble(config, client=client))

    db = sqlite3.connect(config.output_root / "request_ledger.sqlite3")
    counts = dict(db.execute("SELECT status,COUNT(*) FROM calls GROUP BY status"))
    db.close()
    assert client.calls == 1
    assert counts == {"completed": 1, "pending": 5}


def test_resume_with_completed_flash_and_pending_pro_dispatches_only_pro(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    tasks = build_sample_tasks(config)
    flash_calls = build_flash_calls(config, tasks)
    ledger = EnsembleLedger(config.output_root / "request_ledger.sqlite3")
    ledger.prepare(dry_run(config)["signature"])
    ledger.add_calls(flash_calls)
    ledger.assert_role_plan("flash", flash_calls)
    first_client = FakeClient()
    asyncio.run(
        _execute_pending(
            config,
            ledger,
            {call.call_id: call for call in flash_calls},
            first_client,
            call_ids=[call.call_id for call in flash_calls],
        )
    )
    assert first_client.calls == 6
    pro_call = build_pro_call(config, tasks[1], ledger.results("b", "flash"))
    ledger.add_calls([pro_call])
    ledger.assert_role_plan("pro", [pro_call])
    ledger.close()

    resumed_client = FakeClient(request_offset=6)
    summary = asyncio.run(run_ensemble(config, client=resumed_client))

    assert resumed_client.calls == 1
    assert summary["unresolved"] == 0
    assert summary["calls_completed"] == 7


def test_started_call_cannot_return_to_pending_or_dispatch_twice(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tasks = build_sample_tasks(config)
    calls = build_flash_calls(config, tasks)
    ledger = EnsembleLedger(config.output_root / "request_ledger.sqlite3")
    signature = dry_run(config)["signature"]
    ledger.prepare(signature)
    ledger.add_calls(calls)
    ledger.start(calls[0].call_id)

    with pytest.raises(RuntimeError, match="exactly once"):
        ledger.start(calls[0].call_id)
    with pytest.raises(RuntimeError, match="non-repeatable calls"):
        ledger.assert_dispatch_safe()
    row = ledger.db.execute(
        "SELECT status,attempts FROM calls WHERE call_id=?", (calls[0].call_id,)
    ).fetchone()
    assert tuple(row) == ("started", 1)
    ledger.close()


def test_offline_replay_never_reads_key_or_constructs_client(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    tasks = build_sample_tasks(config)
    calls = build_flash_calls(config, tasks)
    ledger = EnsembleLedger(config.output_root / "request_ledger.sqlite3")
    signature = dry_run(config)["signature"]
    ledger.prepare(signature)
    ledger.add_calls(calls)
    ledger.start(calls[0].call_id)
    raw = json.dumps(
        {
            "decision": "MISREAD",
            "confidence": 0.8,
            "rationale": "The diagnostic contradicts the reference.",
        }
    )
    ledger.record_response(
        calls[0].call_id,
        _receipt(calls[0], raw, "offline-provider-id", 77),
    )
    ledger.close()
    monkeypatch.setattr(
        "mprisk.judge.ensemble_misread.load_api_key",
        lambda: (_ for _ in ()).throw(AssertionError("API key was read")),
    )
    monkeypatch.setattr(
        "mprisk.judge.ensemble_misread.DeepSeekEnsembleClient",
        lambda *_: (_ for _ in ()).throw(AssertionError("client was constructed")),
    )

    summary = offline_replay(config)

    assert summary["calls_completed"] == 1
    assert summary["calls_pending"] == 5
    assert list((config.output_root / "audit_snapshots").glob("*/audit_summary.json"))
    assert not (config.output_root / "summary.json").exists()
    assert not (config.output_root / "judgments.jsonl").exists()


def test_forbidden_started_call_overlap_blocks_dry_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    call_id = build_flash_calls(config, build_sample_tasks(config))[0].call_id
    _jsonl(config.forbidden_started_calls_path, [{"call_id": call_id}])
    config.forbidden_started_calls_sha256 = hashlib.sha256(
        config.forbidden_started_calls_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="previously started call IDs"):
        dry_run(config)
