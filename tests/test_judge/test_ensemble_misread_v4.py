from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mprisk.judge import ensemble_misread as v3
from mprisk.judge import ensemble_misread_v4 as v4
from mprisk.judge.misread_judgment import MisreadJudgmentValidationError


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b"".join(
        (
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        for row in rows
    )
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _config(tmp_path: Path, bindings: list[dict[str, str]]) -> v4.EnsembleMisreadConfig:
    return v4.EnsembleMisreadConfig.model_validate(
        {
            "schema_name": v4.CONFIG_SCHEMA,
            "run_id": "v4-test",
            "status": "ready",
            "subject_model_key": "subject",
            "protocol": "VT",
            "split": "recovery_all",
            "api_url": v4.STRICT_API_URL,
            "temperature": 0,
            "thinking": "disabled",
            "max_tokens": 256,
            "confidence_threshold": 0.5,
            "flash_model": "deepseek-v4-flash",
            "pro_model": "deepseek-v4-pro",
            "flash_replicates": 3,
            "gt_coverage_receipt_path": tmp_path / "coverage.json",
            "gt_description_manifest_path": tmp_path / "gt.jsonl",
            "diagnostic_affect_description_manifest_path": tmp_path / "diag.jsonl",
            "diagnostic_run_id": "diag-run",
            "diagnostic_manifest_sha256": "a" * 64,
            "diagnostic_prompt_sha256": "b" * 64,
            "diagnostic_generation_policy_sha256": "c" * 64,
            "diagnostic_request_protocol_signature_sha256": "d" * 64,
            "output_root": tmp_path / "output",
            "forbidden_started_call_ledgers": bindings,
            "request_timeout_seconds": 120,
            "max_concurrency": 16,
            "pricing": {
                "deepseek-v4-flash": {
                    "input_usd_per_million": None,
                    "output_usd_per_million": None,
                },
                "deepseek-v4-pro": {
                    "input_usd_per_million": None,
                    "output_usd_per_million": None,
                },
            },
        }
    )


def _runtime_config(tmp_path: Path) -> v4.EnsembleMisreadConfig:
    forbidden = tmp_path / "forbidden.jsonl"
    forbidden_sha = _write_jsonl(forbidden, [{"call_id": "retired"}])
    gt = tmp_path / "gt.jsonl"
    diagnostic = tmp_path / "diag.jsonl"
    coverage = tmp_path / "coverage.json"
    _write_jsonl(
        gt,
        [
            {"sample_id": "a", "GT_DESCRIPTION": "positive relief"},
            {"sample_id": "b", "GT_DESCRIPTION": "negative worry"},
        ],
    )
    base = _config(
        tmp_path,
        [{"path": str(forbidden), "sha256": forbidden_sha}],
    )
    _write_jsonl(
        diagnostic,
        [
            {
                "schema_name": "mprisk_diagnostic_affect_description_v3",
                "run_id": base.diagnostic_run_id,
                "sample_id": sample_id,
                "subject_model_key": base.subject_model_key,
                "protocol": base.protocol,
                "condition": "M12",
                "split": base.split,
                "DIAGNOSTIC_AFFECT_DESCRIPTION": description,
                "prompt_sha256": base.diagnostic_prompt_sha256,
                "generation_policy_sha256": (
                    base.diagnostic_generation_policy_sha256
                ),
                "request_protocol_signature_sha256": (
                    base.diagnostic_request_protocol_signature_sha256
                ),
            }
            for sample_id, description in (
                ("a", "relief"),
                ("b", "calm"),
            )
        ],
    )
    sample_ids = ["a", "b"]
    sample_id_set_sha256 = hashlib.sha256(
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
                        "sample_id_set_sha256": sample_id_set_sha256,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return base.model_copy(
        update={
            "gt_coverage_receipt_path": coverage,
            "gt_description_manifest_path": gt,
            "diagnostic_affect_description_manifest_path": diagnostic,
            "diagnostic_manifest_sha256": hashlib.sha256(
                diagnostic.read_bytes()
            ).hexdigest(),
            "max_concurrency": 2,
        }
    )


def _receipt(
    call: v4.CallSpec, result: dict[str, object], number: int
) -> v4.HttpResponseReceipt:
    body = json.dumps(
        {
            "id": f"provider-{number}",
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
                                    "name": v4.JUDGMENT_TOOL_NAME,
                                    "arguments": json.dumps(result),
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
        },
        sort_keys=True,
    ).encode()
    return v4.HttpResponseReceipt(
        status_code=200,
        response_body=body,
        response_sha256=hashlib.sha256(body).hexdigest(),
        provider_request_id=f"provider-{number}",
        received_at="2026-08-04T00:00:00+00:00",
    )


class _FiniteReasonClient:
    def __init__(self, *, request_offset: int = 0) -> None:
        self.calls = 0
        self.request_offset = request_offset

    async def complete(self, call: v4.CallSpec) -> v4.HttpResponseReceipt:
        self.calls += 1
        if call.role == "pro":
            result = {"reason_code": "INSUFFICIENT_EVIDENCE", "confidence": 0.4}
        elif call.sample_id == "a":
            result = {"reason_code": "CORE_AFFECT_COMPATIBLE", "confidence": 0.9}
        else:
            reason_code = (
                "PRIMARY_AFFECT_CONTRADICTION"
                if call.slot < 2
                else "CORE_AFFECT_COMPATIBLE"
            )
            result = {"reason_code": reason_code, "confidence": 0.9}
        return _receipt(call, result, self.request_offset + self.calls)


def test_strict_schema_has_only_finite_reason_code_and_confidence() -> None:
    protocol = v4._strict_request_protocol()
    function = protocol["tools"][0]["function"]
    parameters = function["parameters"]

    assert function["strict"] is True
    assert parameters["required"] == ["reason_code", "confidence"]
    assert set(parameters["properties"]) == {"reason_code", "confidence"}
    assert parameters["properties"]["reason_code"]["enum"] == list(v4.REASON_CODES)
    assert "pattern" not in json.dumps(parameters)
    assert "rationale" not in json.dumps(protocol)
    assert "Return no decision field" in v4.STRICT_MISREAD_JUDGMENT_PROMPT
    assert "Return no decision field" in v4.ARBITRATION_PROMPT
    for reason_code, decision in v4.REASON_TO_DECISION.items():
        assert reason_code in v4.STRICT_MISREAD_JUDGMENT_PROMPT
        assert decision in v4.STRICT_MISREAD_JUDGMENT_PROMPT


@pytest.mark.parametrize(("reason_code", "decision"), v4.REASON_TO_DECISION.items())
def test_reason_code_deterministically_derives_decision(
    reason_code: str, decision: str
) -> None:
    result = v4.validate_reason_code_judgment_response(
        json.dumps({"reason_code": reason_code, "confidence": 0.75})
    )
    assert result == {
        "decision": decision,
        "confidence": 0.75,
        "reason_code": reason_code,
    }


@pytest.mark.parametrize(
    "raw",
    [
        '{"decision":"NON_MISREAD","confidence":0.9,'
        '"rationale":"First sentence. Second sentence."}',
        '{"reason_code":"UNKNOWN","confidence":0.9}',
        '{"reason_code":"CORE_AFFECT_COMPATIBLE","confidence":true}',
        '{"reason_code":"CORE_AFFECT_COMPATIBLE","confidence":1.1}',
        '{"reason_code":"CORE_AFFECT_COMPATIBLE","confidence":0.9,"extra":1}',
    ],
)
def test_validator_rejects_old_free_text_and_invalid_finite_payloads(raw: str) -> None:
    with pytest.raises(MisreadJudgmentValidationError):
        v4.validate_reason_code_judgment_response(raw)


def test_multiple_immutable_started_ledgers_are_disjoint_and_isolate_plan(
    tmp_path: Path,
) -> None:
    first = tmp_path / "v2.jsonl"
    second = tmp_path / "v3.jsonl"
    first_sha = _write_jsonl(first, [{"call_id": "old-v2"}])
    second_sha = _write_jsonl(second, [{"call_id": "old-v3"}])
    config = _config(
        tmp_path,
        [
            {"path": str(first), "sha256": first_sha},
            {"path": str(second), "sha256": second_sha},
        ],
    )
    task = v4.SampleTask(
        sample_id="s1",
        reference="relieved",
        diagnostic="relief",
        input_sha256="e" * 64,
    )
    calls = v4.build_flash_calls(config, [task])

    assert v4.validate_forbidden_call_isolation(config, calls) == {
        "forbidden_started_call_ledger_count": 2,
        "forbidden_started_call_count": 2,
        "planned_forbidden_call_id_overlap": 0,
    }

    v3_request = v3._request(
        config.flash_model,
        v3.STRICT_MISREAD_JUDGMENT_PROMPT,
        {
            "GT_DESCRIPTION": task.reference,
            "DIAGNOSTIC_AFFECT_DESCRIPTION": task.diagnostic,
        },
    )
    assert calls[0].request_sha256 != v3._hash(v3._canonical(v3_request))


def test_overlapping_started_ledgers_are_rejected(tmp_path: Path) -> None:
    first = tmp_path / "v2.jsonl"
    second = tmp_path / "v3.jsonl"
    first_sha = _write_jsonl(first, [{"call_id": "duplicate"}])
    second_sha = _write_jsonl(second, [{"call_id": "duplicate"}])
    config = _config(
        tmp_path,
        [
            {"path": str(first), "sha256": first_sha},
            {"path": str(second), "sha256": second_sha},
        ],
    )

    with pytest.raises(ValueError, match="overlap each other"):
        v4.validate_forbidden_call_isolation(config, [])


def test_duplicate_ids_inside_one_started_ledger_are_rejected(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "duplicate.jsonl"
    digest = _write_jsonl(
        ledger,
        [{"call_id": "duplicate"}, {"call_id": "duplicate"}],
    )
    config = _config(
        tmp_path,
        [{"path": str(ledger), "sha256": digest}],
    )

    with pytest.raises(ValueError, match="duplicate call IDs"):
        v4.validate_forbidden_call_isolation(config, [])


def test_full_v4_ensemble_materializes_finite_reason_evidence(
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path)
    client = _FiniteReasonClient()

    summary = asyncio.run(v4.run_ensemble(config, client=client))

    assert client.calls == 7
    assert summary["completed"] == 1
    assert summary["human_review"] == 1
    assert summary["unresolved"] == 0
    judgments = [
        json.loads(line)
        for line in (config.output_root / "judgments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(judgments) == 2
    assert all("rationale" not in json.dumps(row) for row in judgments)
    assert {row["final_reason_code"] for row in judgments} == {
        None,
        "INSUFFICIENT_EVIDENCE",
    }
    assert {row["finalization_basis"] for row in judgments} == {
        "FLASH_UNANIMOUS_CONFIDENT",
        "PRO_ARBITRATION",
    }
    requests = [
        json.loads(line)
        for line in (config.output_root / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["estimated_cost_usd"] is None for row in requests)
    assert summary["estimated_cost_usd"] is None


class _MalformedEnvelopeClient:
    async def complete(self, call: v4.CallSpec) -> v4.HttpResponseReceipt:
        body = b"not-an-api-envelope"
        return v4.HttpResponseReceipt(
            status_code=200,
            response_body=body,
            response_sha256=hashlib.sha256(body).hexdigest(),
            provider_request_id=None,
            received_at="2026-08-04T00:00:00+00:00",
        )


def test_http_receipt_is_durable_before_malformed_envelope_parse(
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path).model_copy(update={"max_concurrency": 1})

    with pytest.raises(RuntimeError, match="invalid_response"):
        asyncio.run(v4.run_ensemble(config, client=_MalformedEnvelopeClient()))

    database = sqlite3.connect(config.output_root / "request_ledger.sqlite3")
    database.row_factory = sqlite3.Row
    row = database.execute(
        "SELECT * FROM calls WHERE status='invalid_response'"
    ).fetchone()
    counts = dict(database.execute("SELECT status,COUNT(*) FROM calls GROUP BY status"))
    database.close()
    assert counts == {"invalid_response": 1, "pending": 5}
    assert row["response_status_code"] == 200
    assert base64.b64decode(row["response_body_base64"]) == b"not-an-api-envelope"
    assert row["response_sha256"] == hashlib.sha256(b"not-an-api-envelope").hexdigest()


class _DrainInflightClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, call: v4.CallSpec) -> v4.HttpResponseReceipt:
        self.calls += 1
        number = self.calls
        if number == 1:
            await asyncio.sleep(0)
            result = {
                "decision": "MISREAD",
                "confidence": 0.9,
                "rationale": "Old payload is forbidden.",
            }
        else:
            await asyncio.sleep(0.01)
            result = {
                "reason_code": "PRIMARY_AFFECT_CONTRADICTION",
                "confidence": 0.9,
            }
        return _receipt(call, result, number)


def test_first_invalid_stops_claims_and_drains_inflight_receipts(
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path)
    client = _DrainInflightClient()

    with pytest.raises(RuntimeError, match="invalid_response"):
        asyncio.run(v4.run_ensemble(config, client=client))

    database = sqlite3.connect(config.output_root / "request_ledger.sqlite3")
    counts = dict(database.execute("SELECT status,COUNT(*) FROM calls GROUP BY status"))
    database.close()
    assert client.calls == config.max_concurrency
    assert counts == {"completed": 1, "invalid_response": 1, "pending": 4}


class _SlowValidClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, call: v4.CallSpec) -> v4.HttpResponseReceipt:
        self.calls += 1
        await asyncio.sleep(0.01)
        return _receipt(
            call,
            {
                "reason_code": "PRIMARY_AFFECT_CONTRADICTION",
                "confidence": 0.9,
            },
            self.calls,
        )


def test_claim_failure_stops_dispatch_and_drains_claimed_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_config(tmp_path)
    client = _SlowValidClient()
    original_start = v4.EnsembleLedger.start
    starts = 0

    def fail_second_start(self: v4.EnsembleLedger, call_id: str) -> None:
        nonlocal starts
        starts += 1
        if starts == 2:
            raise RuntimeError("injected start failure")
        original_start(self, call_id)

    monkeypatch.setattr(v4.EnsembleLedger, "start", fail_second_start)

    with pytest.raises(RuntimeError, match="workers failed after draining"):
        asyncio.run(v4.run_ensemble(config, client=client))

    database = sqlite3.connect(config.output_root / "request_ledger.sqlite3")
    counts = dict(database.execute("SELECT status,COUNT(*) FROM calls GROUP BY status"))
    database.close()
    assert client.calls == 1
    assert counts == {"completed": 1, "pending": 5}


def test_resume_with_completed_flash_dispatches_only_pending_pro(
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path)
    tasks = v4.build_sample_tasks(config)
    flash_calls = v4.build_flash_calls(config, tasks)
    ledger = v4.EnsembleLedger(config.output_root / "request_ledger.sqlite3")
    ledger.prepare(v4.dry_run(config)["signature"])
    ledger.add_calls(flash_calls)
    first_client = _FiniteReasonClient()
    asyncio.run(
        v4._execute_pending(
            config,
            ledger,
            {call.call_id: call for call in flash_calls},
            first_client,
            call_ids=[call.call_id for call in flash_calls],
        )
    )
    pro_call = v4.build_pro_call(config, tasks[1], ledger.results("b", "flash"))
    ledger.add_calls([pro_call])
    ledger.close()

    resumed_client = _FiniteReasonClient(request_offset=6)
    summary = asyncio.run(v4.run_ensemble(config, client=resumed_client))

    assert first_client.calls == 6
    assert resumed_client.calls == 1
    assert summary["calls_completed"] == 7
    assert summary["unresolved"] == 0


def test_offline_replay_never_reads_key_or_constructs_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_config(tmp_path)
    tasks = v4.build_sample_tasks(config)
    calls = v4.build_flash_calls(config, tasks)
    ledger = v4.EnsembleLedger(config.output_root / "request_ledger.sqlite3")
    ledger.prepare(v4.dry_run(config)["signature"])
    ledger.add_calls(calls)
    ledger.start(calls[0].call_id)
    ledger.record_response(
        calls[0].call_id,
        _receipt(
            calls[0],
            {"reason_code": "PRIMARY_AFFECT_CONTRADICTION", "confidence": 0.8},
            77,
        ),
    )
    ledger.close()
    monkeypatch.setattr(
        v4,
        "load_api_key",
        lambda: (_ for _ in ()).throw(AssertionError("API key was read")),
    )
    monkeypatch.setattr(
        v4,
        "DeepSeekEnsembleClient",
        lambda *_: (_ for _ in ()).throw(AssertionError("client was constructed")),
    )

    summary = v4.offline_replay(config)

    assert summary["calls_completed"] == 1
    assert summary["calls_pending"] == 5
    assert not (config.output_root / "judgments.jsonl").exists()


class _DuplicateProviderClient(_FiniteReasonClient):
    async def complete(self, call: v4.CallSpec) -> v4.HttpResponseReceipt:
        self.calls += 1
        result = {"reason_code": "CORE_AFFECT_COMPATIBLE", "confidence": 0.9}
        return _receipt(call, result, 1)


def test_duplicate_provider_request_id_fails_closed(tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)

    with pytest.raises(RuntimeError, match="invalid_response"):
        asyncio.run(v4.run_ensemble(config, client=_DuplicateProviderClient()))

    database = sqlite3.connect(config.output_root / "request_ledger.sqlite3")
    counts = dict(database.execute("SELECT status,COUNT(*) FROM calls GROUP BY status"))
    database.close()
    assert counts == {"completed": 1, "invalid_response": 1, "pending": 4}


def test_final_materialization_refuses_unresolved_calls(tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    tasks = v4.build_sample_tasks(config)
    calls = v4.build_flash_calls(config, tasks)
    ledger = v4.EnsembleLedger(config.output_root / "request_ledger.sqlite3")
    signature = v4.dry_run(config)["signature"]
    ledger.prepare(signature)
    ledger.add_calls(calls)

    with pytest.raises(RuntimeError, match="Incomplete judgment state"):
        v4._materialize(config, signature, tasks, ledger, final=True)

    assert not (config.output_root / "judgments.jsonl").exists()
    ledger.close()


def test_final_cross_field_invariants_reject_inconsistent_states(
    tmp_path: Path,
) -> None:
    ledger = v4.EnsembleLedger(tmp_path / "ledger.sqlite3")
    with pytest.raises(ValueError, match="unanimous-Flash"):
        ledger.set_final(
            "s1",
            status="completed",
            decision="MISREAD",
            confidence=0.9,
            arbitrator_used=False,
            finalization_basis="FLASH_UNANIMOUS_CONFIDENT",
            reason_code="PRIMARY_AFFECT_CONTRADICTION",
        )
    with pytest.raises(ValueError, match="differs from reason_code"):
        ledger.set_final(
            "s1",
            status="completed",
            decision="NON_MISREAD",
            confidence=0.9,
            arbitrator_used=True,
            finalization_basis="PRO_ARBITRATION",
            reason_code="PRIMARY_AFFECT_CONTRADICTION",
        )
    with pytest.raises(ValueError, match="Unknown final status"):
        ledger.set_final(
            "s1",
            status="bogus",
            decision=None,
            confidence=0.1,
            arbitrator_used=True,
            finalization_basis="PRO_ARBITRATION",
            reason_code="INSUFFICIENT_EVIDENCE",
        )
    with pytest.raises(ValueError, match="completed Pro"):
        ledger.set_final(
            "s1",
            status="completed",
            decision="UNCERTAIN",
            confidence=0.9,
            arbitrator_used=True,
            finalization_basis="PRO_ARBITRATION",
            reason_code="INSUFFICIENT_EVIDENCE",
        )
    with pytest.raises(ValueError, match="human-review Pro"):
        ledger.set_final(
            "s1",
            status="human_review",
            decision=None,
            confidence=0.9,
            arbitrator_used=True,
            finalization_basis="PRO_ARBITRATION",
            reason_code="CORE_AFFECT_COMPATIBLE",
        )
    with pytest.raises(ValueError, match="unanimous-Flash"):
        ledger.set_final(
            "s1",
            status="completed",
            decision="MISREAD",
            confidence=0.49,
            arbitrator_used=False,
            finalization_basis="FLASH_UNANIMOUS_CONFIDENT",
            reason_code=None,
        )
    with pytest.raises(ValueError, match="finite"):
        ledger.set_final(
            "s1",
            status="human_review",
            decision=None,
            confidence=float("nan"),
            arbitrator_used=True,
            finalization_basis="PRO_ARBITRATION",
            reason_code="INSUFFICIENT_EVIDENCE",
        )
    with pytest.raises(ValueError, match="finite"):
        ledger.set_final(
            "s1",
            status="human_review",
            decision=None,
            confidence=float("inf"),
            arbitrator_used=True,
            finalization_basis="PRO_ARBITRATION",
            reason_code="INSUFFICIENT_EVIDENCE",
        )
    ledger.close()
