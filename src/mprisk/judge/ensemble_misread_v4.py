"""Fail-closed 3x Flash + 1x Pro Misread judgment with a request ledger."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from mprisk.config.loader import load_yaml
from mprisk.judge.misread_judgment import MisreadJudgmentValidationError

CONFIG_SCHEMA = "mprisk_ensemble_misread_judgment_config_v4"
SIGNATURE_SCHEMA = "mprisk_ensemble_misread_signature_v4"
OUTPUT_SCHEMA = "mprisk_ensemble_misread_label_v4"
PROVENANCE_SCHEMA = "mprisk_ensemble_misread_provenance_v3"
GT_COVERAGE_SCHEMA = "mprisk_target_gt_coverage_v1"
STRICT_API_URL = "https://api.deepseek.com/beta/chat/completions"
JUDGMENT_TOOL_NAME = "submit_misread_judgment"
REASON_TO_DECISION = {
    "SURFACE_CUE_DOMINANCE": "MISREAD",
    "PRIMARY_AFFECT_CONTRADICTION": "MISREAD",
    "DISTINCT_AFFECT_COLLAPSE": "MISREAD",
    "DECISIVE_COMPONENT_OMISSION": "MISREAD",
    "CONFIDENT_OPPOSITE_ACCOUNT": "MISREAD",
    "CORE_AFFECT_COMPATIBLE": "NON_MISREAD",
    "SYNONYMOUS_AFFECT": "NON_MISREAD",
    "VALID_SIMPLIFICATION": "NON_MISREAD",
    "INSUFFICIENT_EVIDENCE": "UNCERTAIN",
}
REASON_CODES = tuple(REASON_TO_DECISION)
REASON_CODE_GUIDE = (
    "Return no decision field and no free text. Select exactly one reason_code: "
    "SURFACE_CUE_DOMINANCE, PRIMARY_AFFECT_CONTRADICTION, DISTINCT_AFFECT_COLLAPSE, "
    "DECISIVE_COMPONENT_OMISSION, and CONFIDENT_OPPOSITE_ACCOUNT derive MISREAD; "
    "CORE_AFFECT_COMPATIBLE, SYNONYMOUS_AFFECT, and VALID_SIMPLIFICATION derive "
    "NON_MISREAD; INSUFFICIENT_EVIDENCE derives UNCERTAIN."
)
STRICT_MISREAD_JUDGMENT_PROMPT = (
    "Compare the reference description with the diagnostic affect description. Misread applies "
    "when the diagnostic is led by surface cues, contradicts the primary affect, wrongly "
    "compresses distinct affects, omits a decisive component, or gives a confidently opposite "
    "account. Non-misread applies when the core affect is compatible, synonymous, or a valid "
    "simplification. Uncertain applies only when the comparison cannot decide. Submit exactly one "
    "judgment through the required tool. "
    + REASON_CODE_GUIDE
    + " Confidence is a number from 0 through 1."
)
ARBITRATION_PROMPT = (
    "Act as the final adjudicator for an affective Misread decision. Independently compare the "
    "reference and diagnostic descriptions, then use the three blinded preliminary assessments "
    "only as supporting evidence. Submit exactly one judgment through the required tool. "
    + REASON_CODE_GUIDE
    + " Confidence is a number from 0 through 1. Use INSUFFICIENT_EVIDENCE when the evidence "
    "cannot decide."
)


class StartedCallLedgerBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    sha256: str

    @field_validator("sha256")
    @classmethod
    def sha256_digest(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("Started-call ledger binding must be a SHA-256 digest")
        return value


class EnsembleMisreadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_name: Literal["mprisk_ensemble_misread_judgment_config_v4"]
    run_id: str
    status: Literal["pending", "ready"]
    subject_model_key: str
    protocol: Literal["VT", "VA"]
    split: str
    api_url: str
    temperature: Literal[0]
    thinking: Literal["disabled"]
    max_tokens: Literal[256]
    confidence_threshold: float
    flash_model: Literal["deepseek-v4-flash"]
    pro_model: Literal["deepseek-v4-pro"]
    flash_replicates: Literal[3]
    gt_coverage_receipt_path: Path
    gt_description_manifest_path: Path
    diagnostic_affect_description_manifest_path: Path
    diagnostic_run_id: str
    diagnostic_manifest_sha256: str | None
    diagnostic_prompt_sha256: str
    diagnostic_generation_policy_sha256: str
    diagnostic_request_protocol_signature_sha256: str
    output_root: Path
    forbidden_started_call_ledgers: list[StartedCallLedgerBinding]
    request_timeout_seconds: float
    max_concurrency: int
    pricing: dict[str, dict[str, float | None]]

    @field_validator("run_id", "subject_model_key", "split", "diagnostic_run_id")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identity fields must be non-empty")
        return value

    @field_validator("confidence_threshold")
    @classmethod
    def threshold(cls, value: float) -> float:
        if value != 0.5:
            raise ValueError("The frozen confidence threshold is 0.5")
        return value

    @field_validator("request_timeout_seconds", "max_concurrency")
    @classmethod
    def positive(cls, value: Any) -> Any:
        if value <= 0:
            raise ValueError("timeout and concurrency must be positive")
        return value

    @field_validator("api_url")
    @classmethod
    def strict_api_url(cls, value: str) -> str:
        if value != STRICT_API_URL:
            raise ValueError(f"Strict tool mode requires {STRICT_API_URL}")
        return value

    @model_validator(mode="after")
    def pricing_contract(self) -> EnsembleMisreadConfig:
        for model in (self.flash_model, self.pro_model):
            rates = self.pricing.get(model)
            if not isinstance(rates, dict) or set(rates) != {
                "input_usd_per_million",
                "output_usd_per_million",
            }:
                raise ValueError(f"Missing explicit pricing contract for {model}")
            for rate in rates.values():
                if rate is not None:
                    raise ValueError(
                        "v4 ledger pricing must remain null until cache-hit and cache-miss "
                        "rates are represented separately"
                    )
        digests = (
            self.diagnostic_prompt_sha256,
            self.diagnostic_generation_policy_sha256,
            self.diagnostic_request_protocol_signature_sha256,
        )
        if any(len(value) != 64 for value in digests):
            raise ValueError("Diagnostic binding fields must be SHA-256 digests")
        if self.status == "ready" and (
            self.diagnostic_manifest_sha256 is None
            or len(self.diagnostic_manifest_sha256) != 64
        ):
            raise ValueError("Ready judgment config requires diagnostic_manifest_sha256")
        if not self.forbidden_started_call_ledgers:
            raise ValueError("At least one immutable started-call ledger is required")
        paths = [binding.path.resolve() for binding in self.forbidden_started_call_ledgers]
        if len(paths) != len(set(paths)):
            raise ValueError("Started-call ledger paths must be unique")
        return self


@dataclass(frozen=True)
class SampleTask:
    sample_id: str
    reference: str
    diagnostic: str
    input_sha256: str


@dataclass(frozen=True)
class CallSpec:
    call_id: str
    sample_id: str
    role: Literal["flash", "pro"]
    slot: int
    model: str
    request: dict[str, Any]
    request_sha256: str


@dataclass(frozen=True)
class ApiCompletion:
    raw_content: str
    request_id: str
    response_model: str
    usage: dict[str, int]
    tool_call_id: str


@dataclass(frozen=True)
class HttpResponseReceipt:
    status_code: int
    response_body: bytes
    response_sha256: str
    provider_request_id: str | None
    received_at: str


def load_config(path: Path) -> EnsembleMisreadConfig:
    return EnsembleMisreadConfig.model_validate(load_yaml(path))


def load_api_key() -> str:
    value = os.environ.get("DEEPSEEK_API_KEY")
    if not value:
        raise ValueError("DEEPSEEK_API_KEY is required")
    return value


def build_sample_tasks(config: EnsembleMisreadConfig) -> list[SampleTask]:
    if config.status != "ready":
        raise ValueError("Misread judgment is blocked until diagnostic status is ready")
    if (
        config.diagnostic_manifest_sha256
        != _sha256(config.diagnostic_affect_description_manifest_path)
    ):
        raise ValueError("Diagnostic manifest SHA-256 does not match the judgment config")
    references = _index(_read_jsonl(config.gt_description_manifest_path), "GT_DESCRIPTION")
    _validate_gt_coverage_receipt(config, set(references))
    diagnostics = _index(
        _read_jsonl(config.diagnostic_affect_description_manifest_path),
        "DIAGNOSTIC_AFFECT_DESCRIPTION",
    )
    if not references or set(references) != set(diagnostics):
        raise ValueError("GT and diagnostic manifests must cover identical non-empty IDs")
    tasks: list[SampleTask] = []
    for sample_id in sorted(references):
        diag_row = diagnostics[sample_id]
        expected = {
            "schema_name": "mprisk_diagnostic_affect_description_v3",
            "run_id": config.diagnostic_run_id,
            "subject_model_key": config.subject_model_key,
            "protocol": config.protocol,
            "condition": "M12",
            "split": config.split,
            "prompt_sha256": config.diagnostic_prompt_sha256,
            "generation_policy_sha256": config.diagnostic_generation_policy_sha256,
            "request_protocol_signature_sha256": (
                config.diagnostic_request_protocol_signature_sha256
            ),
        }
        if any(diag_row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Diagnostic identity mismatch: {sample_id}")
        reference = _required_text(references[sample_id], "GT_DESCRIPTION")
        diagnostic = _required_text(diag_row, "DIAGNOSTIC_AFFECT_DESCRIPTION")
        blind = {"GT_DESCRIPTION": reference, "DIAGNOSTIC_AFFECT_DESCRIPTION": diagnostic}
        tasks.append(
            SampleTask(
                sample_id=sample_id,
                reference=reference,
                diagnostic=diagnostic,
                input_sha256=_hash(_canonical(blind)),
            )
        )
    return tasks


def _validate_gt_coverage_receipt(
    config: EnsembleMisreadConfig, sample_ids: set[str]
) -> None:
    if not config.gt_coverage_receipt_path.is_file():
        raise FileNotFoundError(config.gt_coverage_receipt_path)
    receipt = json.loads(config.gt_coverage_receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_name") != GT_COVERAGE_SCHEMA
        or receipt.get("status") != "PASS"
    ):
        raise ValueError("Target GT coverage receipt is not PASS")
    protocols = receipt.get("protocols")
    record = protocols.get(config.protocol) if isinstance(protocols, dict) else None
    if not isinstance(record, dict) or record.get("complete") is not True:
        raise ValueError(f"Target GT coverage is incomplete for {config.protocol}")
    expected = len(sample_ids)
    checks = {
        "expected_rows": expected,
        "observed_rows": expected,
        "unique_sample_ids": expected,
        "blank_sample_ids": 0,
        "duplicate_sample_ids": 0,
        "protocol_mismatches": 0,
        "nonempty_gt_descriptions": expected,
        "missing_gt_descriptions": 0,
        "sample_id_set_sha256": _hash(_canonical(sorted(sample_ids))),
    }
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in checks.items()
        if record.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Target GT coverage receipt does not match the judgment input: "
            + _canonical(mismatches)
        )


def build_flash_calls(config: EnsembleMisreadConfig, tasks: Sequence[SampleTask]) -> list[CallSpec]:
    calls: list[CallSpec] = []
    for task in tasks:
        request = _request(
            config.flash_model,
            STRICT_MISREAD_JUDGMENT_PROMPT,
            {
                "GT_DESCRIPTION": task.reference,
                "DIAGNOSTIC_AFFECT_DESCRIPTION": task.diagnostic,
            },
        )
        request_sha = _hash(_canonical(request))
        for slot in range(3):
            calls.append(
                CallSpec(
                    call_id=_hash(
                        _canonical(
                            {
                                "sample_id": task.sample_id,
                                "role": "flash",
                                "slot": slot,
                                "request": request_sha,
                            }
                        )
                    ),
                    sample_id=task.sample_id,
                    role="flash",
                    slot=slot,
                    model=config.flash_model,
                    request=request,
                    request_sha256=request_sha,
                )
            )
    return calls


def validate_forbidden_call_isolation(
    config: EnsembleMisreadConfig, calls: Sequence[CallSpec]
) -> dict[str, int]:
    forbidden: set[str] = set()
    per_ledger_counts: list[int] = []
    for binding in config.forbidden_started_call_ledgers:
        path = binding.path
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != binding.sha256:
            raise ValueError(f"Forbidden started-call ledger SHA-256 mismatch: {path}")
        ledger_rows = _read_jsonl(path)
        ledger_call_ids = [_required_text(row, "call_id") for row in ledger_rows]
        if len(ledger_call_ids) != len(set(ledger_call_ids)):
            raise ValueError(f"Immutable started-call ledger has duplicate call IDs: {path}")
        ledger_call_id_set = set(ledger_call_ids)
        if forbidden & ledger_call_id_set:
            raise ValueError("Immutable started-call ledgers overlap each other")
        forbidden.update(ledger_call_id_set)
        per_ledger_counts.append(len(ledger_call_id_set))
    planned = {call.call_id for call in calls}
    overlap = sorted(forbidden & planned)
    if overlap:
        raise ValueError(f"Planned calls overlap {len(overlap)} previously started call IDs")
    return {
        "forbidden_started_call_ledger_count": len(per_ledger_counts),
        "forbidden_started_call_count": len(forbidden),
        "planned_forbidden_call_id_overlap": 0,
    }


def build_pro_call(
    config: EnsembleMisreadConfig, task: SampleTask, flash_results: list[dict[str, Any]]
) -> CallSpec:
    if len(flash_results) != 3:
        raise ValueError("Pro arbitration requires exactly three Flash results")
    payload = {
        "GT_DESCRIPTION": task.reference,
        "DIAGNOSTIC_AFFECT_DESCRIPTION": task.diagnostic,
        "PRELIMINARY_ASSESSMENTS": [
            {
                key: result[key]
                for key in ("decision", "confidence", "reason_code")
            }
            for result in flash_results
        ],
    }
    request = _request(config.pro_model, ARBITRATION_PROMPT, payload)
    request_sha = _hash(_canonical(request))
    return CallSpec(
        call_id=_hash(
            _canonical(
                {"sample_id": task.sample_id, "role": "pro", "slot": 0, "request": request_sha}
            )
        ),
        sample_id=task.sample_id,
        role="pro",
        slot=0,
        model=config.pro_model,
        request=request,
        request_sha256=request_sha,
    )


class DeepSeekEnsembleClient:
    def __init__(self, config: EnsembleMisreadConfig, api_key: str) -> None:
        self.config = config
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(config.request_timeout_seconds))
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def complete(self, call: CallSpec) -> HttpResponseReceipt:
        try:
            response = await self.client.post(
                self.config.api_url, headers=self.headers, json=call.request
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RuntimeError(type(exc).__name__) from exc
        body = response.content
        return HttpResponseReceipt(
            status_code=response.status_code,
            response_body=body,
            response_sha256=hashlib.sha256(body).hexdigest(),
            provider_request_id=_extract_provider_request_id(body),
            received_at=_now(),
        )

    async def close(self) -> None:
        await self.client.aclose()


def _extract_provider_request_id(body: bytes) -> str | None:
    try:
        envelope = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    request_id = envelope.get("id") if isinstance(envelope, dict) else None
    return request_id if isinstance(request_id, str) and request_id.strip() else None


def _parse_api_completion(receipt: HttpResponseReceipt, expected_model: str) -> ApiCompletion:
    if receipt.status_code >= 400:
        raise ValueError(f"API HTTP status is {receipt.status_code}")
    try:
        envelope = json.loads(receipt.response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("API envelope is not JSON") from exc
    if not isinstance(envelope, dict):
        raise ValueError("API envelope must be an object")
    if envelope.get("model") != expected_model:
        raise ValueError("API model differs from requested model")
    request_id = envelope.get("id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("API response has no request ID")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("API response must contain one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "tool_calls":
        raise ValueError("API response did not finish with a tool call")
    message = choice.get("message")
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("API response must contain exactly one tool call")
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict) or tool_call.get("type") != "function":
        raise ValueError("API response tool call type is invalid")
    tool_call_id = tool_call.get("id")
    function = tool_call.get("function")
    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
        raise ValueError("API response tool call has no ID")
    if not isinstance(function, dict) or function.get("name") != JUDGMENT_TOOL_NAME:
        raise ValueError("API response called an unexpected function")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("API response tool arguments are missing")
    usage_raw = envelope.get("usage")
    if not isinstance(usage_raw, dict):
        raise ValueError("API response usage is missing")
    usage = {}
    for key in (
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        value = usage_raw.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"API usage {key} is invalid")
        usage[key] = value
    if usage["prompt_tokens"] != (
        usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
    ):
        raise ValueError("API prompt-token usage breakdown is inconsistent")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError("API total-token usage is inconsistent")
    return ApiCompletion(
        raw_content=arguments,
        request_id=request_id,
        response_model=expected_model,
        usage=usage,
        tool_call_id=tool_call_id,
    )


class EnsembleLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls(
              call_id TEXT PRIMARY KEY,sample_id TEXT NOT NULL,
              role TEXT NOT NULL,slot INTEGER NOT NULL,
              model TEXT NOT NULL,request_sha256 TEXT NOT NULL,request_json TEXT NOT NULL,
              status TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,request_id TEXT,
              tool_call_id TEXT,response_status_code INTEGER,response_sha256 TEXT,
              response_body_base64 TEXT,
              raw_response TEXT,result_json TEXT,usage_json TEXT,
              estimated_cost_usd REAL,error_type TEXT,error_message TEXT,updated_at TEXT NOT NULL,
              started_at TEXT,response_received_at TEXT,validated_at TEXT,terminal_at TEXT,
              UNIQUE(sample_id,role,slot));
            CREATE TABLE IF NOT EXISTS final(
              sample_id TEXT PRIMARY KEY,status TEXT NOT NULL,decision TEXT,confidence REAL,
              arbitrator_used INTEGER NOT NULL,finalization_basis TEXT NOT NULL,
              reason_code TEXT,updated_at TEXT NOT NULL);
            """
        )

    def prepare(self, signature: dict[str, Any]) -> None:
        encoded = _canonical(signature)
        with self.db:
            current = self.db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            if current is not None and current[0] != encoded:
                raise ValueError("Existing ensemble ledger signature differs")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('signature',?)", (encoded,))
        bad_attempts = self.db.execute(
            """SELECT COUNT(*) FROM calls WHERE attempts NOT IN (0,1)
            OR (attempts=1 AND status='pending') OR (attempts=0 AND status!='pending')"""
        ).fetchone()[0]
        if bad_attempts:
            raise RuntimeError("Ledger violates the attempt-zero-only dispatch contract")
        bad_receipts = self.db.execute(
            """SELECT COUNT(*) FROM calls
            WHERE status IN ('response_received','completed','invalid_response')
            AND (response_status_code IS NULL OR response_sha256 IS NULL
                 OR response_body_base64 IS NULL OR response_received_at IS NULL)"""
        ).fetchone()[0]
        if bad_receipts:
            raise RuntimeError("Ledger contains a response state without a durable receipt")
        bad_phases = self.db.execute(
            """SELECT COUNT(*) FROM calls WHERE
            (attempts=1 AND started_at IS NULL)
            OR (status='completed' AND (tool_call_id IS NULL OR raw_response IS NULL
                OR result_json IS NULL OR usage_json IS NULL OR validated_at IS NULL
                OR terminal_at IS NULL))
            OR (status='invalid_response' AND (validated_at IS NULL OR terminal_at IS NULL))
            OR (status='ambiguous' AND terminal_at IS NULL)"""
        ).fetchone()[0]
        if bad_phases:
            raise RuntimeError("Ledger contains an incomplete phase timeline")
        bad_domains = self.db.execute(
            """SELECT COUNT(*) FROM calls
            WHERE role NOT IN ('flash','pro') OR status NOT IN
            ('pending','started','response_received','completed','invalid_response','ambiguous')"""
        ).fetchone()[0]
        if bad_domains:
            raise RuntimeError("Ledger contains an unknown role or state")

    def assert_dispatch_safe(self) -> None:
        blocked = self.db.execute(
            """SELECT status,COUNT(*) FROM calls
            WHERE status IN ('started','ambiguous','invalid_response')
            GROUP BY status ORDER BY status"""
        ).fetchall()
        if blocked:
            details = ", ".join(f"{row[0]}={row[1]}" for row in blocked)
            raise RuntimeError(f"Ledger contains non-repeatable calls: {details}")
        duplicate_provider_ids = self.db.execute(
            """SELECT COUNT(*) FROM (
            SELECT request_id FROM calls WHERE request_id IS NOT NULL
            GROUP BY request_id HAVING COUNT(*)>1)"""
        ).fetchone()[0]
        if duplicate_provider_ids:
            raise RuntimeError("Ledger contains duplicate provider request IDs")

    def add_calls(self, calls: Sequence[CallSpec]) -> None:
        with self.db:
            for call in calls:
                observed = self.db.execute(
                    "SELECT request_sha256,request_json FROM calls WHERE call_id=?", (call.call_id,)
                ).fetchone()
                expected = (call.request_sha256, _canonical(call.request))
                if observed is not None:
                    if tuple(observed) != expected:
                        raise ValueError(f"Call signature mismatch: {call.call_id}")
                    continue
                self.db.execute(
                    """INSERT INTO calls(
                    call_id,sample_id,role,slot,model,request_sha256,
                    request_json,status,updated_at)
                    VALUES(?,?,?,?,?,?,?,'pending',?)""",
                    (
                        call.call_id,
                        call.sample_id,
                        call.role,
                        call.slot,
                        call.model,
                        call.request_sha256,
                        expected[1],
                        _now(),
                    ),
                )

    def assert_role_plan(self, role: str, calls: Sequence[CallSpec]) -> None:
        expected = {call.call_id for call in calls}
        observed = {
            row[0]
            for row in self.db.execute("SELECT call_id FROM calls WHERE role=?", (role,))
        }
        if observed != expected:
            raise RuntimeError(f"Ledger {role} call set differs from the immutable plan")

    def assert_stage_boundary(self) -> None:
        pro_count = self.db.execute(
            "SELECT COUNT(*) FROM calls WHERE role='pro'"
        ).fetchone()[0]
        incomplete_flash = self.db.execute(
            "SELECT COUNT(*) FROM calls WHERE role='flash' AND status!='completed'"
        ).fetchone()[0]
        if pro_count and incomplete_flash:
            raise RuntimeError("Pro calls exist before the Flash stage is complete")

    def pending_calls(self, allowed_call_ids: Sequence[str]) -> list[str]:
        allowed = set(allowed_call_ids)
        return [
            row[0]
            for row in self.db.execute(
                "SELECT call_id FROM calls WHERE status='pending' ORDER BY role,sample_id,slot"
            )
            if row[0] in allowed
        ]

    def start(self, call_id: str) -> None:
        with self.db:
            cursor = self.db.execute(
                """UPDATE calls SET status='started',attempts=1,
                updated_at=?,started_at=? WHERE call_id=? AND status='pending' AND attempts=0""",
                (_now(), _now(), call_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Call cannot be dispatched exactly once: {call_id}")

    def record_response(self, call_id: str, receipt: HttpResponseReceipt) -> None:
        with self.db:
            cursor = self.db.execute(
                """UPDATE calls SET status='response_received',request_id=?,
                response_status_code=?,response_sha256=?,response_body_base64=?,
                response_received_at=?,error_type=NULL,error_message=NULL,updated_at=?
                WHERE call_id=? AND status='started' AND attempts=1""",
                (
                    receipt.provider_request_id,
                    receipt.status_code,
                    receipt.response_sha256,
                    base64.b64encode(receipt.response_body).decode("ascii"),
                    receipt.received_at,
                    receipt.received_at,
                    call_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Response cannot be recorded from current state: {call_id}")

    def record_parsed_completion(
        self, call_id: str, completion: ApiCompletion, cost: float | None
    ) -> None:
        with self.db:
            cursor = self.db.execute(
                """UPDATE calls SET request_id=?,tool_call_id=?,raw_response=?,usage_json=?,
                estimated_cost_usd=?,updated_at=?
                WHERE call_id=? AND status='response_received'
                AND (request_id IS NULL OR request_id=?)""",
                (
                    completion.request_id,
                    completion.tool_call_id,
                    completion.raw_content,
                    _canonical(completion.usage),
                    cost,
                    _now(),
                    call_id,
                    completion.request_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Parsed response conflicts with durable receipt: {call_id}")

    def provider_request_id_is_unique(self, call_id: str) -> bool:
        row = self.db.execute(
            "SELECT request_id FROM calls WHERE call_id=?", (call_id,)
        ).fetchone()
        if row is None or not row[0]:
            return False
        return (
            self.db.execute(
                "SELECT COUNT(*) FROM calls WHERE request_id=?", (row[0],)
            ).fetchone()[0]
            == 1
        )

    def complete_validation(self, call_id: str, result: dict[str, Any]) -> None:
        with self.db:
            cursor = self.db.execute(
                """UPDATE calls SET status='completed',result_json=?,
                error_type=NULL,error_message=NULL,updated_at=?,validated_at=?,terminal_at=?
                WHERE call_id=? AND status='response_received'""",
                (_canonical(result), _now(), _now(), _now(), call_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Response cannot be completed from current state: {call_id}")

    def mark_invalid_response(self, call_id: str, exc: Exception) -> None:
        with self.db:
            cursor = self.db.execute(
                """UPDATE calls SET status='invalid_response',error_type=?,error_message=?,
                updated_at=?,validated_at=?,terminal_at=?
                WHERE call_id=? AND status='response_received'""",
                (type(exc).__name__, str(exc), _now(), _now(), _now(), call_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Invalid response cannot be recorded: {call_id}")

    def mark_ambiguous(self, call_id: str, exc: Exception) -> None:
        with self.db:
            cursor = self.db.execute(
                """UPDATE calls SET status='ambiguous',error_type=?,error_message=?,
                updated_at=?,terminal_at=? WHERE call_id=? AND status='started'""",
                (type(exc).__name__, str(exc), _now(), _now(), call_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Ambiguous dispatch cannot be recorded: {call_id}")

    def response_received_rows(self) -> list[sqlite3.Row]:
        return list(
            self.db.execute(
                "SELECT * FROM calls WHERE status='response_received' ORDER BY role,sample_id,slot"
            )
        )

    def call_rows(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM calls ORDER BY sample_id,role,slot"))

    def results(self, sample_id: str, role: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """SELECT result_json FROM calls WHERE sample_id=? AND role=?
            AND status='completed' ORDER BY slot""",
            (sample_id, role),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def set_final(
        self,
        sample_id: str,
        *,
        status: str,
        decision: str | None,
        confidence: float | None,
        arbitrator_used: bool,
        finalization_basis: str,
        reason_code: str | None,
    ) -> None:
        if status not in {"completed", "human_review"}:
            raise ValueError("Unknown final status")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Final confidence must be finite and in [0,1]")
        if finalization_basis == "FLASH_UNANIMOUS_CONFIDENT":
            if (
                arbitrator_used
                or status != "completed"
                or decision not in {"MISREAD", "NON_MISREAD"}
                or confidence < 0.5
                or reason_code is not None
            ):
                raise ValueError("Invalid unanimous-Flash final state")
        elif finalization_basis == "PRO_ARBITRATION":
            if not arbitrator_used or reason_code not in REASON_TO_DECISION:
                raise ValueError("Invalid Pro-arbitrated final state")
            derived = REASON_TO_DECISION[reason_code]
            if decision is not None and decision != derived:
                raise ValueError("Final decision differs from reason_code")
            if status == "completed" and (
                decision not in {"MISREAD", "NON_MISREAD"}
                or confidence < 0.5
            ):
                raise ValueError("Invalid completed Pro final state")
            if status == "human_review" and (
                decision is not None
                or not (derived == "UNCERTAIN" or confidence < 0.5)
            ):
                raise ValueError("Invalid human-review Pro final state")
        else:
            raise ValueError("Unknown finalization basis")
        with self.db:
            self.db.execute(
                """INSERT INTO final VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(sample_id) DO UPDATE SET
                status=excluded.status,decision=excluded.decision,
                confidence=excluded.confidence,
                arbitrator_used=excluded.arbitrator_used,
                finalization_basis=excluded.finalization_basis,
                reason_code=excluded.reason_code,updated_at=excluded.updated_at""",
                (
                    sample_id,
                    status,
                    decision,
                    confidence,
                    int(arbitrator_used),
                    finalization_basis,
                    reason_code,
                    _now(),
                ),
            )

    def final_rows(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM final ORDER BY sample_id"))

    def close(self) -> None:
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()


async def run_ensemble(
    config: EnsembleMisreadConfig, *, client: Any | None = None
) -> dict[str, int | float | None]:
    if config.status != "ready":
        raise ValueError("Ensemble config is not ready")
    tasks = build_sample_tasks(config)
    flash_calls = build_flash_calls(config, tasks)
    validate_forbidden_call_isolation(config, flash_calls)
    by_call = {call.call_id: call for call in flash_calls}
    signature = _signature(config, tasks)
    ledger = EnsembleLedger(config.output_root / "request_ledger.sqlite3")
    owns_client = client is None
    try:
        ledger.prepare(signature)
        ledger.add_calls(flash_calls)
        ledger.assert_role_plan("flash", flash_calls)
        _replay_recorded_responses(ledger)
        ledger.assert_dispatch_safe()
        ledger.assert_stage_boundary()
        if client is None:
            client = DeepSeekEnsembleClient(config, load_api_key())
        await _execute_pending(
            config, ledger, by_call, client, call_ids=[call.call_id for call in flash_calls]
        )
        ledger.assert_dispatch_safe()
        pro_calls: list[CallSpec] = []
        for task in tasks:
            flashes = ledger.results(task.sample_id, "flash")
            if len(flashes) != 3:
                continue
            unanimous = len({row["decision"] for row in flashes}) == 1
            decision = flashes[0]["decision"] if unanimous else None
            confident = all(row["confidence"] >= config.confidence_threshold for row in flashes)
            if unanimous and confident and decision in {"MISREAD", "NON_MISREAD"}:
                ledger.set_final(
                    task.sample_id,
                    status="completed",
                    decision=decision,
                    confidence=min(row["confidence"] for row in flashes),
                    arbitrator_used=False,
                    finalization_basis="FLASH_UNANIMOUS_CONFIDENT",
                    reason_code=None,
                )
            else:
                call = build_pro_call(config, task, flashes)
                pro_calls.append(call)
                by_call[call.call_id] = call
        validate_forbidden_call_isolation(config, pro_calls)
        ledger.add_calls(pro_calls)
        ledger.assert_role_plan("pro", pro_calls)
        await _execute_pending(
            config, ledger, by_call, client, call_ids=[call.call_id for call in pro_calls]
        )
        ledger.assert_dispatch_safe()
        for task in tasks:
            pro = ledger.results(task.sample_id, "pro")
            if not pro:
                continue
            result = pro[0]
            review = (
                result["decision"] == "UNCERTAIN"
                or result["confidence"] < config.confidence_threshold
            )
            ledger.set_final(
                task.sample_id,
                status="human_review" if review else "completed",
                decision=None if review else result["decision"],
                confidence=result["confidence"],
                arbitrator_used=True,
                finalization_basis="PRO_ARBITRATION",
                reason_code=result["reason_code"],
            )
        _materialize(config, signature, tasks, ledger, final=True)
        return _summary(tasks, ledger)
    except Exception:
        _materialize(config, signature, tasks, ledger, final=False)
        raise
    finally:
        if owns_client and client is not None:
            await client.close()
        ledger.close()


async def _execute_pending(
    config: EnsembleMisreadConfig,
    ledger: EnsembleLedger,
    by_call: dict[str, CallSpec],
    client: Any,
    *,
    call_ids: Sequence[str],
) -> None:
    pending_call_ids = ledger.pending_calls(call_ids)
    pending = iter(pending_call_ids)
    dispatch_lock = asyncio.Lock()
    stop_dispatch = asyncio.Event()
    worker_errors: list[Exception] = []

    async def claim() -> tuple[str, CallSpec] | None:
        async with dispatch_lock:
            if stop_dispatch.is_set():
                return None
            try:
                call_id = next(pending)
            except StopIteration:
                return None
            call = by_call.get(call_id)
            if call is None:
                raise ValueError(f"Pending call is absent from the immutable plan: {call_id}")
            ledger.start(call_id)
            return call_id, call

    async def worker() -> None:
        while True:
            try:
                claimed = await claim()
            except Exception as exc:
                stop_dispatch.set()
                worker_errors.append(exc)
                return
            if claimed is None:
                return
            call_id, call = claimed
            try:
                receipt = await client.complete(call)
            except Exception as exc:
                stop_dispatch.set()
                try:
                    ledger.mark_ambiguous(call_id, exc)
                except Exception as ledger_exc:
                    worker_errors.append(ledger_exc)
                return
            try:
                ledger.record_response(call_id, receipt)
                completion = _parse_api_completion(receipt, call.model)
                ledger.record_parsed_completion(
                    call_id,
                    completion,
                    _estimate_cost(config, call.model, completion.usage),
                )
                if not ledger.provider_request_id_is_unique(call_id):
                    raise MisreadJudgmentValidationError(
                        "provider request ID must be globally unique"
                    )
                result = validate_reason_code_judgment_response(
                    completion.raw_content
                )
                ledger.complete_validation(call_id, result)
            except Exception as exc:
                stop_dispatch.set()
                try:
                    ledger.mark_invalid_response(call_id, exc)
                except Exception as ledger_exc:
                    worker_errors.append(ledger_exc)
                return

    pending_count = len(pending_call_ids)
    results = await asyncio.gather(
        *(worker() for _ in range(min(config.max_concurrency, pending_count))),
        return_exceptions=True,
    )
    worker_errors.extend(result for result in results if isinstance(result, Exception))
    if worker_errors:
        raise RuntimeError(
            f"Judgment workers failed after draining {len(worker_errors)} internal error(s)"
        ) from worker_errors[0]


def _replay_recorded_responses(ledger: EnsembleLedger) -> None:
    for row in ledger.response_received_rows():
        call_id = row["call_id"]
        try:
            body = base64.b64decode(row["response_body_base64"], validate=True)
            if hashlib.sha256(body).hexdigest() != row["response_sha256"]:
                raise MisreadJudgmentValidationError("durable response body hash differs")
            receipt = HttpResponseReceipt(
                status_code=row["response_status_code"],
                response_body=body,
                response_sha256=row["response_sha256"],
                provider_request_id=row["request_id"],
                received_at=row["response_received_at"],
            )
            completion = _parse_api_completion(receipt, row["model"])
            ledger.record_parsed_completion(call_id, completion, None)
            if not ledger.provider_request_id_is_unique(call_id):
                raise MisreadJudgmentValidationError(
                    "provider request ID must be globally unique"
                )
            result = validate_reason_code_judgment_response(completion.raw_content)
            ledger.complete_validation(call_id, result)
        except Exception as exc:
            ledger.mark_invalid_response(call_id, exc)


def offline_replay(config: EnsembleMisreadConfig) -> dict[str, int | float | None]:
    """Validate recorded responses without constructing a client or reading an API key."""
    tasks = build_sample_tasks(config)
    calls = build_flash_calls(config, tasks)
    validate_forbidden_call_isolation(config, calls)
    signature = _signature(config, tasks)
    ledger_path = config.output_root / "request_ledger.sqlite3"
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    ledger = EnsembleLedger(ledger_path)
    try:
        ledger.prepare(signature)
        _replay_recorded_responses(ledger)
        _materialize(config, signature, tasks, ledger, final=False)
        return _summary(tasks, ledger)
    finally:
        ledger.close()


def dry_run(config: EnsembleMisreadConfig) -> dict[str, Any]:
    """Validate frozen inputs and requests without reading the API key."""
    tasks = build_sample_tasks(config)
    calls = build_flash_calls(config, tasks)
    isolation = validate_forbidden_call_isolation(config, calls)
    return {
        "sample_count": len(tasks),
        "flash_request_count": len(calls),
        "pro_request_upper_bound": len(tasks),
        "max_api_request_count": len(calls) + len(tasks),
        "planned_call_id_count": len(calls),
        "unique_planned_call_id_count": len({call.call_id for call in calls}),
        "unique_request_payload_sha256_count": len({call.request_sha256 for call in calls}),
        "api_requests_issued": 0,
        "api_key_accessed": False,
        **isolation,
        "signature": _signature(config, tasks),
    }


def _estimate_cost(
    config: EnsembleMisreadConfig, model: str, usage: dict[str, int]
) -> float | None:
    rates = config.pricing[model]
    input_rate = rates["input_usd_per_million"]
    output_rate = rates["output_usd_per_million"]
    if input_rate is None or output_rate is None:
        return None
    return (
        usage["prompt_tokens"] * input_rate + usage["completion_tokens"] * output_rate
    ) / 1_000_000


def _materialize(
    config: EnsembleMisreadConfig,
    signature: dict[str, Any],
    tasks: Sequence[SampleTask],
    ledger: EnsembleLedger,
    *,
    final: bool,
) -> None:
    call_rows = [dict(row) for row in ledger.call_rows()]
    finals = [dict(row) for row in ledger.final_rows()]
    final_by_id = {row["sample_id"]: row for row in finals}
    judgments = []
    queue = []
    for task in tasks:
        row = final_by_id.get(task.sample_id)
        if row is None:
            continue
        flashes = ledger.results(task.sample_id, "flash")
        pro = ledger.results(task.sample_id, "pro")
        record = {
            "schema_name": OUTPUT_SCHEMA,
            "sample_id": task.sample_id,
            "subject_model_key": config.subject_model_key,
            "protocol": config.protocol,
            "status": row["status"],
            "final_label": row["decision"],
            "confidence": row["confidence"],
            "arbitrator_used": bool(row["arbitrator_used"]),
            "finalization_basis": row["finalization_basis"],
            "final_reason_code": row["reason_code"],
            "flash": flashes,
            "pro_arbitration": pro[0] if pro else None,
            "diagnostic_manifest_sha256": config.diagnostic_manifest_sha256,
            "diagnostic_prompt_sha256": config.diagnostic_prompt_sha256,
            "diagnostic_generation_policy_sha256": (
                config.diagnostic_generation_policy_sha256
            ),
            "diagnostic_request_protocol_signature_sha256": (
                config.diagnostic_request_protocol_signature_sha256
            ),
        }
        judgments.append(record)
        if row["status"] == "human_review":
            queue.append(record)
    failures = [
        {
            key: row[key]
            for key in (
                "call_id",
                "sample_id",
                "role",
                "slot",
                "model",
                "error_type",
                "error_message",
            )
        }
        for row in call_rows
        if row["status"] in {"started", "ambiguous", "invalid_response"}
    ]
    request_records = [
        {
            key: row[key]
            for key in (
                "call_id",
                "sample_id",
                "role",
                "slot",
                "model",
                "request_sha256",
                "status",
                "attempts",
                "request_id",
                "tool_call_id",
                "response_status_code",
                "response_sha256",
                "usage_json",
                "estimated_cost_usd",
                "error_type",
                "error_message",
                "started_at",
                "response_received_at",
                "validated_at",
                "terminal_at",
            )
        }
        for row in call_rows
    ]
    summary = _summary(tasks, ledger)
    if final:
        non_completed = [row for row in call_rows if row["status"] != "completed"]
        if summary["unresolved"] or non_completed or failures:
            raise RuntimeError("Incomplete judgment state cannot be materialized as final")
        payloads = {
            "judgments.jsonl": _jsonl(judgments),
            "human_review_queue.jsonl": _jsonl(queue),
            "failures.jsonl": _jsonl(failures),
            "requests.jsonl": _jsonl(request_records),
            "summary.json": (
                json.dumps(summary, sort_keys=True, indent=2) + "\n"
            ).encode(),
        }
        provenance_name = "provenance.json"
        artifact_root = config.output_root
        artifact_prefix = Path()
    else:
        payloads = {
            "audit_failures.jsonl": _jsonl(failures),
            "audit_requests.jsonl": _jsonl(request_records),
            "audit_summary.json": (
                json.dumps(summary, sort_keys=True, indent=2) + "\n"
            ).encode(),
        }
        provenance_name = "audit_provenance.json"
        snapshot_id = _hash(
            _canonical({"summary": summary, "requests": request_records})
        )
        artifact_root = config.output_root / "audit_snapshots" / snapshot_id
        artifact_prefix = Path("audit_snapshots") / snapshot_id
    for name, content in payloads.items():
        _atomic_bytes(artifact_root / name, content)
    provenance = {
        "schema_name": PROVENANCE_SCHEMA,
        "run_id": config.run_id,
        "status": "complete" if final else "incomplete",
        "signature": signature,
        "policy": {
            "flash_replicates": 3,
            "pro_trigger": "not unanimous confident binary",
            "human_review": "final Pro UNCERTAIN or confidence below threshold",
            "no_binary_fallback": True,
        },
        "pricing": config.pricing,
        "artifacts": {
            name: {
                "path": str(artifact_prefix / name),
                "sha256": _sha256(artifact_root / name),
            }
            for name in payloads
        },
    }
    _atomic_bytes(
        artifact_root / provenance_name,
        (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode(),
    )


def _summary(tasks: Sequence[SampleTask], ledger: EnsembleLedger) -> dict[str, int | float | None]:
    calls = [dict(row) for row in ledger.call_rows()]
    finals = [dict(row) for row in ledger.final_rows()]
    status = Counter(row["status"] for row in finals)
    call_status = Counter(row["status"] for row in calls)
    costs = [row["estimated_cost_usd"] for row in calls if row["estimated_cost_usd"] is not None]
    return {
        "samples": len(tasks),
        "completed": status["completed"],
        "human_review": status["human_review"],
        "unresolved": len(tasks) - len(finals),
        "calls_completed": call_status["completed"],
        "calls_pending": call_status["pending"],
        "calls_started": call_status["started"],
        "calls_response_received": call_status["response_received"],
        "calls_invalid_response": call_status["invalid_response"],
        "calls_ambiguous": call_status["ambiguous"],
        "calls_failed": call_status["invalid_response"] + call_status["ambiguous"],
        "estimated_cost_usd": sum(costs)
        if len(costs) == sum(row["attempts"] > 0 for row in calls)
        else None,
    }


def _signature(config: EnsembleMisreadConfig, tasks: Sequence[SampleTask]) -> dict[str, Any]:
    return {
        "schema_name": SIGNATURE_SCHEMA,
        "run_id": config.run_id,
        "config_sha256": _hash(_canonical(config.model_dump(mode="json"))),
        "subject_model_key": config.subject_model_key,
        "protocol": config.protocol,
        "split": config.split,
        "flash_model": config.flash_model,
        "pro_model": config.pro_model,
        "flash_replicates": 3,
        "temperature": 0,
        "confidence_threshold": config.confidence_threshold,
        "prompt_sha256": _hash(STRICT_MISREAD_JUDGMENT_PROMPT),
        "arbitration_prompt_sha256": _hash(ARBITRATION_PROMPT),
        "gt_coverage_receipt_sha256": _sha256(config.gt_coverage_receipt_path),
        "gt_manifest_sha256": _sha256(config.gt_description_manifest_path),
        "diagnostic_manifest_sha256": _sha256(config.diagnostic_affect_description_manifest_path),
        "diagnostic_prompt_sha256": config.diagnostic_prompt_sha256,
        "diagnostic_generation_policy_sha256": (
            config.diagnostic_generation_policy_sha256
        ),
        "diagnostic_request_protocol_signature_sha256": (
            config.diagnostic_request_protocol_signature_sha256
        ),
        "strict_api_url": config.api_url,
        "thinking": config.thinking,
        "max_tokens": config.max_tokens,
        "tool_name": JUDGMENT_TOOL_NAME,
        "request_protocol_sha256": _hash(_canonical(_strict_request_protocol())),
        "validator_contract_sha256": _hash(_canonical(_validator_contract())),
        "finalization_contract_sha256": _hash(
            _canonical(_finalization_contract())
        ),
        "forbidden_started_call_ledgers": [
            {
                "path": str(binding.path),
                "sha256": binding.sha256,
            }
            for binding in config.forbidden_started_call_ledgers
        ],
        "sample_count": len(tasks),
    }


def _request(model: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _canonical(payload)},
        ],
        **_strict_request_protocol(),
    }


def _strict_request_protocol() -> dict[str, Any]:
    return {
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "max_tokens": 256,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": JUDGMENT_TOOL_NAME,
                    "description": "Submit one blinded affective Misread judgment.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason_code": {
                                "type": "string",
                                "enum": list(REASON_CODES),
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": ["reason_code", "confidence"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": JUDGMENT_TOOL_NAME},
        },
        "stream": False,
    }


def _validator_contract() -> dict[str, Any]:
    return {
        "decoder": "json.loads",
        "exact_keys": ["confidence", "reason_code"],
        "reason_to_decision": REASON_TO_DECISION,
        "decision_derivation": "deterministic_from_reason_code",
        "confidence": {"type": "number_not_boolean", "minimum": 0, "maximum": 1},
    }


def _finalization_contract() -> dict[str, Any]:
    return {
        "confidence_threshold": 0.5,
        "confidence": {"finite": True, "minimum": 0, "maximum": 1},
        "flash_unanimous_confident": {
            "status": "completed",
            "decision": ["MISREAD", "NON_MISREAD"],
            "reason_code": None,
            "arbitrator_used": False,
        },
        "pro_completed": {
            "status": "completed",
            "decision": ["MISREAD", "NON_MISREAD"],
            "decision_derived_from_reason_code": True,
            "minimum_confidence": 0.5,
            "arbitrator_used": True,
        },
        "pro_human_review": {
            "status": "human_review",
            "decision": None,
            "trigger": "UNCERTAIN reason or confidence below 0.5",
            "arbitrator_used": True,
        },
    }


def validate_reason_code_judgment_response(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise MisreadJudgmentValidationError("judge response must be a string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MisreadJudgmentValidationError(
            "judge response must be exact JSON"
        ) from exc
    if not isinstance(value, dict) or set(value) != {"confidence", "reason_code"}:
        raise MisreadJudgmentValidationError(
            "judge response must contain exactly confidence and reason_code"
        )
    reason_code = value["reason_code"]
    confidence = value["confidence"]
    if reason_code not in REASON_TO_DECISION:
        raise MisreadJudgmentValidationError("judge reason_code is invalid")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not 0 <= confidence <= 1
    ):
        raise MisreadJudgmentValidationError(
            "judge confidence must be in [0,1]"
        )
    return {
        "decision": REASON_TO_DECISION[reason_code],
        "confidence": float(confidence),
        "reason_code": reason_code,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict 3xFlash+Pro Misread judgment.")
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--offline-replay", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = (
        dry_run(config)
        if args.dry_run
        else offline_replay(config)
        if args.offline_replay
        else asyncio.run(run_ensemble(config))
    )
    print(_canonical(result))
    return 0


def _index(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _required_text(row, "sample_id")
        _required_text(row, field)
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL contains non-object rows: {path}")
    return rows


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty {key}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(_canonical(row) + "\n" for row in rows).encode()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
