from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from mprisk.judge.ensemble_misread import (
    ApiCompletion,
    EnsembleMisreadConfig,
    dry_run,
    run_ensemble,
)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config(tmp_path: Path) -> EnsembleMisreadConfig:
    gt = tmp_path / "gt.jsonl"
    diagnostic = tmp_path / "diagnostic.jsonl"
    coverage = tmp_path / "gt_coverage.json"
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
                "schema_name": "mprisk_diagnostic_affect_description_v2",
                "run_id": "diag",
                "sample_id": sample_id,
                "subject_model_key": "model",
                "protocol": "VT",
                "condition": "M12",
                "split": "train",
                "DIAGNOSTIC_AFFECT_DESCRIPTION": description,
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
    return EnsembleMisreadConfig(
        schema_name="mprisk_ensemble_misread_judgment_config_v1",
        run_id="run",
        status="ready",
        subject_model_key="model",
        protocol="VT",
        split="train",
        api_url="https://invalid.example",
        temperature=0,
        confidence_threshold=0.5,
        flash_model="deepseek-v4-flash",
        pro_model="deepseek-v4-pro",
        flash_replicates=3,
        gt_coverage_receipt_path=coverage,
        gt_description_manifest_path=gt,
        diagnostic_affect_description_manifest_path=diagnostic,
        diagnostic_run_id="diag",
        output_root=tmp_path / "out",
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
    def __init__(self) -> None:
        self.calls = 0

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
        return ApiCompletion(
            raw_content=raw,
            request_id=f"request-{self.calls}",
            response_model=call.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            response_envelope_sha256=f"{self.calls:064x}",
        )


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

    second = FakeClient()
    repeated = asyncio.run(run_ensemble(config, client=second))
    assert second.calls == 0
    assert repeated == result


class FailingClient:
    async def complete(self, call):
        raise RuntimeError(f"external failure for {call.call_id}")


def test_ensemble_external_failures_are_not_silent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="Flash judgment failed for 6 request"):
        asyncio.run(run_ensemble(config, client=FailingClient()))
    summary = json.loads((config.output_root / "summary.json").read_text())
    assert summary["calls_failed"] == 6
    assert summary["unresolved"] == 2


def test_confidence_threshold_is_frozen_at_half(tmp_path: Path) -> None:
    payload = _config(tmp_path).model_dump()
    payload["confidence_threshold"] = 0.6
    with pytest.raises(ValueError, match="frozen confidence threshold is 0.5"):
        EnsembleMisreadConfig.model_validate(payload)
