"""Fail-closed 3x Flash + 1x Pro Misread judgment with a request ledger."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
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
from mprisk.judge.misread_judgment import (
    MISREAD_JUDGMENT_PROMPT,
    validate_misread_judgment_response,
)

CONFIG_SCHEMA = "mprisk_ensemble_misread_judgment_config_v1"
SIGNATURE_SCHEMA = "mprisk_ensemble_misread_signature_v1"
OUTPUT_SCHEMA = "mprisk_ensemble_misread_label_v1"
PROVENANCE_SCHEMA = "mprisk_ensemble_misread_provenance_v1"
ARBITRATION_PROMPT = (
    "Act as the final adjudicator for an affective Misread decision. Independently compare the "
    "reference and diagnostic descriptions, then use the three blinded preliminary assessments "
    "only as supporting evidence. Return exact JSON with decision, confidence, and one short "
    "rationale sentence. Use MISREAD, NON_MISREAD, or UNCERTAIN; do not force a binary decision."
)


class EnsembleMisreadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_name: Literal["mprisk_ensemble_misread_judgment_config_v1"]
    run_id: str
    status: Literal["pending", "ready"]
    subject_model_key: str
    protocol: Literal["VT", "VA"]
    split: str
    api_url: str
    temperature: Literal[0]
    confidence_threshold: float
    flash_model: Literal["deepseek-v4-flash"]
    pro_model: Literal["deepseek-v4-pro"]
    flash_replicates: Literal[3]
    gt_description_manifest_path: Path
    diagnostic_affect_description_manifest_path: Path
    diagnostic_run_id: str
    output_root: Path
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
                if rate is not None and rate < 0:
                    raise ValueError("pricing rates must be nonnegative or null")
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
    response_envelope_sha256: str


def load_config(path: Path) -> EnsembleMisreadConfig:
    return EnsembleMisreadConfig.model_validate(load_yaml(path))


def load_api_key() -> str:
    value = os.environ.get("DEEPSEEK_API_KEY")
    if not value:
        raise ValueError("DEEPSEEK_API_KEY is required")
    return value


def build_sample_tasks(config: EnsembleMisreadConfig) -> list[SampleTask]:
    references = _index(_read_jsonl(config.gt_description_manifest_path), "GT_DESCRIPTION")
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
            "schema_name": "mprisk_diagnostic_affect_description_v2",
            "run_id": config.diagnostic_run_id,
            "subject_model_key": config.subject_model_key,
            "protocol": config.protocol,
            "condition": "M12",
            "split": config.split,
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


def build_flash_calls(config: EnsembleMisreadConfig, tasks: Sequence[SampleTask]) -> list[CallSpec]:
    calls: list[CallSpec] = []
    for task in tasks:
        request = _request(
            config.flash_model,
            MISREAD_JUDGMENT_PROMPT,
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


def build_pro_call(
    config: EnsembleMisreadConfig, task: SampleTask, flash_results: list[dict[str, Any]]
) -> CallSpec:
    if len(flash_results) != 3:
        raise ValueError("Pro arbitration requires exactly three Flash results")
    payload = {
        "GT_DESCRIPTION": task.reference,
        "DIAGNOSTIC_AFFECT_DESCRIPTION": task.diagnostic,
        "PRELIMINARY_ASSESSMENTS": [
            {key: result[key] for key in ("decision", "confidence", "rationale")}
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

    async def complete(self, call: CallSpec) -> ApiCompletion:
        try:
            response = await self.client.post(
                self.config.api_url, headers=self.headers, json=call.request
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RuntimeError(type(exc).__name__) from exc
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}")
        envelope_bytes = response.content
        try:
            envelope = response.json()
        except json.JSONDecodeError as exc:
            raise ValueError("API envelope is not JSON") from exc
        if envelope.get("model") != call.model:
            raise ValueError("API model differs from requested model")
        request_id = envelope.get("id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("API response has no request ID")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("API response must contain one choice")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("API response content is missing")
        usage_raw = envelope.get("usage")
        if not isinstance(usage_raw, dict):
            raise ValueError("API response usage is missing")
        usage = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage_raw.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"API usage {key} is invalid")
            usage[key] = value
        return ApiCompletion(
            raw_content=content,
            request_id=request_id,
            response_model=call.model,
            usage=usage,
            response_envelope_sha256=hashlib.sha256(envelope_bytes).hexdigest(),
        )

    async def close(self) -> None:
        await self.client.aclose()


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
              response_sha256 TEXT,raw_response TEXT,result_json TEXT,usage_json TEXT,
              estimated_cost_usd REAL,error_type TEXT,error_message TEXT,updated_at TEXT NOT NULL,
              UNIQUE(sample_id,role,slot));
            CREATE TABLE IF NOT EXISTS final(
              sample_id TEXT PRIMARY KEY,status TEXT NOT NULL,decision TEXT,confidence REAL,
              arbitrator_used INTEGER NOT NULL,rationale TEXT,updated_at TEXT NOT NULL);
            """
        )

    def prepare(self, signature: dict[str, Any], *, retry_failed: bool) -> None:
        encoded = _canonical(signature)
        with self.db:
            current = self.db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            if current is not None and current[0] != encoded:
                raise ValueError("Existing ensemble ledger signature differs")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('signature',?)", (encoded,))
            self.db.execute("UPDATE calls SET status='pending' WHERE status='running'")
            if retry_failed:
                self.db.execute("UPDATE calls SET status='pending' WHERE status='failed'")

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

    def pending_calls(self) -> list[str]:
        return [
            row[0]
            for row in self.db.execute(
                "SELECT call_id FROM calls WHERE status='pending' ORDER BY role,sample_id,slot"
            )
        ]

    def start(self, call_id: str) -> None:
        with self.db:
            self.db.execute(
                """UPDATE calls SET status='running',attempts=attempts+1,
                updated_at=? WHERE call_id=?""",
                (_now(), call_id),
            )

    def complete(
        self, call_id: str, completion: ApiCompletion, result: dict[str, Any], cost: float | None
    ) -> None:
        with self.db:
            self.db.execute(
                """UPDATE calls SET status='completed',request_id=?,
                response_sha256=?,raw_response=?,result_json=?,usage_json=?,
                estimated_cost_usd=?,error_type=NULL,error_message=NULL,
                updated_at=? WHERE call_id=?""",
                (
                    completion.request_id,
                    completion.response_envelope_sha256,
                    completion.raw_content,
                    _canonical(result),
                    _canonical(completion.usage),
                    cost,
                    _now(),
                    call_id,
                ),
            )

    def fail(self, call_id: str, exc: Exception) -> None:
        with self.db:
            self.db.execute(
                """UPDATE calls SET status='failed',error_type=?,error_message=?,
                updated_at=? WHERE call_id=?""",
                (type(exc).__name__, str(exc), _now(), call_id),
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
        rationale: str | None,
    ) -> None:
        with self.db:
            self.db.execute(
                """INSERT INTO final VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(sample_id) DO UPDATE SET
                status=excluded.status,decision=excluded.decision,
                confidence=excluded.confidence,
                arbitrator_used=excluded.arbitrator_used,
                rationale=excluded.rationale,updated_at=excluded.updated_at""",
                (sample_id, status, decision, confidence, int(arbitrator_used), rationale, _now()),
            )

    def final_rows(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM final ORDER BY sample_id"))

    def close(self) -> None:
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()


async def run_ensemble(
    config: EnsembleMisreadConfig, *, client: Any | None = None, retry_failed: bool = False
) -> dict[str, int]:
    if config.status != "ready":
        raise ValueError("Ensemble config is not ready")
    tasks = build_sample_tasks(config)
    flash_calls = build_flash_calls(config, tasks)
    by_call = {call.call_id: call for call in flash_calls}
    signature = _signature(config, tasks)
    ledger = EnsembleLedger(config.output_root / "request_ledger.sqlite3")
    ledger.prepare(signature, retry_failed=retry_failed)
    ledger.add_calls(flash_calls)
    owns_client = client is None
    try:
        if client is None:
            client = DeepSeekEnsembleClient(config, load_api_key())
        await _execute_pending(config, ledger, by_call, client)
        _raise_failed_calls(ledger, stage="Flash")
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
                    rationale="Three independent Flash judgments were unanimous and confident.",
                )
            else:
                call = build_pro_call(config, task, flashes)
                pro_calls.append(call)
                by_call[call.call_id] = call
        ledger.add_calls(pro_calls)
        await _execute_pending(config, ledger, by_call, client)
        _raise_failed_calls(ledger, stage="Pro")
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
                rationale=result["rationale"],
            )
        _materialize(config, signature, tasks, ledger)
        return _summary(tasks, ledger)
    except Exception:
        _materialize(config, signature, tasks, ledger)
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
) -> None:
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def execute_one(call_id: str) -> None:
        call = by_call.get(call_id)
        if call is None:
            raise ValueError(f"Pending call is absent from the immutable plan: {call_id}")
        async with semaphore:
            ledger.start(call_id)
            try:
                completion = await client.complete(call)
                result = validate_misread_judgment_response(completion.raw_content)
                ledger.complete(
                    call_id,
                    completion,
                    result,
                    _estimate_cost(config, call.model, completion.usage),
                )
            except Exception as exc:
                ledger.fail(call_id, exc)

    await asyncio.gather(*(execute_one(call_id) for call_id in ledger.pending_calls()))


def dry_run(config: EnsembleMisreadConfig) -> dict[str, Any]:
    """Validate frozen inputs and requests without reading the API key."""
    tasks = build_sample_tasks(config)
    calls = build_flash_calls(config, tasks)
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
        "signature": _signature(config, tasks),
    }


def _raise_failed_calls(ledger: EnsembleLedger, *, stage: str) -> None:
    failed = [row for row in ledger.call_rows() if row["status"] == "failed"]
    if failed:
        raise RuntimeError(
            f"{stage} judgment failed for {len(failed)} request(s); "
            "resume with --retry-failed after correcting the external failure"
        )


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
            "flash": flashes,
            "pro_arbitration": pro[0] if pro else None,
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
        if row["status"] == "failed"
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
                "response_sha256",
                "usage_json",
                "estimated_cost_usd",
                "error_type",
                "error_message",
            )
        }
        for row in call_rows
    ]
    payloads = {
        "judgments.jsonl": _jsonl(judgments),
        "human_review_queue.jsonl": _jsonl(queue),
        "failures.jsonl": _jsonl(failures),
        "requests.jsonl": _jsonl(request_records),
        "summary.json": (
            json.dumps(_summary(tasks, ledger), sort_keys=True, indent=2) + "\n"
        ).encode(),
    }
    for name, content in payloads.items():
        _atomic_bytes(config.output_root / name, content)
    provenance = {
        "schema_name": PROVENANCE_SCHEMA,
        "run_id": config.run_id,
        "signature": signature,
        "policy": {
            "flash_replicates": 3,
            "pro_trigger": "not unanimous confident binary",
            "human_review": "final Pro UNCERTAIN or confidence below threshold",
            "no_binary_fallback": True,
        },
        "pricing": config.pricing,
        "artifacts": {
            name: {"path": name, "sha256": _sha256(config.output_root / name)} for name in payloads
        },
    }
    _atomic_bytes(
        config.output_root / "provenance.json",
        (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode(),
    )


def _summary(tasks: Sequence[SampleTask], ledger: EnsembleLedger) -> dict[str, int | float | None]:
    calls = [dict(row) for row in ledger.call_rows()]
    finals = [dict(row) for row in ledger.final_rows()]
    status = Counter(row["status"] for row in finals)
    costs = [row["estimated_cost_usd"] for row in calls if row["estimated_cost_usd"] is not None]
    return {
        "samples": len(tasks),
        "completed": status["completed"],
        "human_review": status["human_review"],
        "unresolved": len(tasks) - len(finals),
        "calls_completed": sum(row["status"] == "completed" for row in calls),
        "calls_failed": sum(row["status"] == "failed" for row in calls),
        "estimated_cost_usd": sum(costs)
        if len(costs) == sum(row["status"] == "completed" for row in calls)
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
        "prompt_sha256": _hash(MISREAD_JUDGMENT_PROMPT),
        "arbitration_prompt_sha256": _hash(ARBITRATION_PROMPT),
        "gt_manifest_sha256": _sha256(config.gt_description_manifest_path),
        "diagnostic_manifest_sha256": _sha256(config.diagnostic_affect_description_manifest_path),
        "sample_count": len(tasks),
    }


def _request(model: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _canonical(payload)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict 3xFlash+Pro Misread judgment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = (
        dry_run(config)
        if args.dry_run
        else asyncio.run(run_ensemble(config, retry_failed=args.retry_failed))
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
