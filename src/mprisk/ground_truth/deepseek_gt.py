"""Strict resumable DeepSeek GT-description generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from mprisk.config.loader import load_yaml
from mprisk.data.generated_archive_freeze import _canonical_json, _sha256

PROMPT_KIND = {"A": "a_conflict", "C": "c_aligned"}
ARCHIVE_ORDER = ("accept_a_svt", "accept_a_va", "accept_c_svt", "accept_c_va")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _GTConfigBase(StrictModel):
    model: Literal["deepseek-v4-flash"]
    api_url: str
    env_file: Path
    api_key_variable: Literal["DEEPSEEK_API_KEY"]
    temperature: Literal[0]
    max_tokens: Literal[128]
    thinking: Literal["disabled"]
    concurrency: int
    retry_delays_seconds: list[float]
    request_timeout_seconds: float
    min_words: int
    max_words: int
    output_root: Path
    a_prompt_path: Path
    c_prompt_path: Path

    @field_validator("concurrency")
    @classmethod
    def positive_concurrency(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("concurrency must be positive")
        return value

    @field_validator("retry_delays_seconds")
    @classmethod
    def retry_delays_must_be_non_negative(cls, value: list[float]) -> list[float]:
        if any(delay < 0 for delay in value):
            raise ValueError("retry delays must be non-negative")
        return value

    @field_validator("max_words")
    @classmethod
    def word_range_must_be_valid(cls, value: int, info: Any) -> int:
        min_words = info.data.get("min_words")
        if not isinstance(min_words, int) or min_words <= 0 or value < min_words:
            raise ValueError("word limits must satisfy 0 < min_words <= max_words")
        return value


class GTConfig(_GTConfigBase):
    schema_name: Literal["mprisk_deepseek_gt_config_v1"]
    run_id: Literal["deepseek_gt_v1"]
    input_root: Path


class GTPromptContextV2Config(_GTConfigBase):
    schema_name: Literal["mprisk_deepseek_gt_config_v2"]
    run_id: Literal["deepseek_gt_prompt_context_v2_pilot"]
    protocol_version: Literal["prompt_context_v2"]
    input_manifest: Path
    input_manifest_sha256: str
    expected_count: int

    @field_validator("expected_count")
    @classmethod
    def expected_count_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("expected_count must be positive")
        return value


AnyGTConfig = GTConfig | GTPromptContextV2Config


@dataclass(frozen=True)
class GTTask:
    order: int
    sample_id: str
    source_archive: str
    data_type: str
    input_hash: str
    prompt_hash: str
    system_prompt: str
    model_input: dict[str, Any]
    eligible_row: dict[str, Any]
    ledger_signature: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunResult:
    total: int
    completed: int
    failed: int
    pending: int
    output_root: Path


class TransientAPIError(RuntimeError):
    pass


class PermanentAPIError(RuntimeError):
    pass


class GTValidationError(ValueError):
    pass


def load_config(path: str | Path) -> AnyGTConfig:
    payload = load_yaml(path)
    if payload.get("schema_name") == "mprisk_deepseek_gt_config_v1":
        return GTConfig.model_validate(payload)
    if payload.get("schema_name") == "mprisk_deepseek_gt_config_v2":
        return GTPromptContextV2Config.model_validate(payload)
    raise ValueError(f"Unsupported DeepSeek GT config schema: {payload.get('schema_name')!r}")


def load_api_key(config: AnyGTConfig) -> str:
    value = os.environ.get(config.api_key_variable)
    if value:
        return value
    values = _read_env_file(config.env_file)
    value = values.get(config.api_key_variable)
    if value:
        return value
    raise ValueError("DEEPSEEK_API_KEY is required for ground-truth generation")


def prepare_tasks(repo_root: str | Path, config: AnyGTConfig) -> list[GTTask]:
    if isinstance(config, GTPromptContextV2Config):
        return _prepare_prompt_context_v2_tasks(repo_root, config)
    return _prepare_v1_tasks(repo_root, config)


def _prepare_v1_tasks(repo_root: str | Path, config: GTConfig) -> list[GTTask]:
    root = Path(repo_root).resolve()
    input_root = (root / config.input_root).resolve()
    eligible = _read_jsonl(input_root / "gt_eligible.jsonl")
    assignments = _index(_read_jsonl(input_root / "archetype_semantic_assignments_v1.jsonl"))
    dictionary = {
        row["archetype_semantic_id"]: row
        for row in _read_jsonl(input_root / "archetype_canonical_meanings_v1.jsonl")
    }
    prompts = {
        "A": (root / config.a_prompt_path).read_text(encoding="utf-8"),
        "C": (root / config.c_prompt_path).read_text(encoding="utf-8"),
    }
    tasks: list[GTTask] = []
    for order, row in enumerate(eligible):
        sample_id = _text(row, "sample_id")
        assignment = assignments.get(sample_id)
        if assignment is None or assignment.get("gt_eligible") is not True:
            raise ValueError(f"Missing eligible semantic assignment: {sample_id}")
        if assignment.get("source_row_sha256") != row.get("source_row_sha256"):
            raise ValueError(f"Assignment hash mismatch: {sample_id}")
        semantic_id = _text(assignment, "archetype_semantic_id")
        meaning = dictionary.get(semantic_id)
        if meaning is None or meaning.get("data_type") != row.get("data_type"):
            raise ValueError(f"Dictionary join mismatch: {sample_id}")
        data_type = _text(row, "data_type")
        model_input = {
            "archetype": {
                "id": semantic_id,
                "name": meaning["canonical_name"],
                "canonical_meaning": meaning["canonical_meaning"],
            },
            "trigger_context": row["context_text"],
            "dialogue": row["dialogue_text"],
            "surface_emotion": meaning["surface_emotion"],
        }
        _validate_model_input(model_input)
        prompt = prompts[data_type]
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        input_hash = hashlib.sha256(
            _canonical_json(
                {"model": config.model, "prompt_hash": prompt_hash, "input": model_input}
            ).encode()
        ).hexdigest()
        tasks.append(
            GTTask(
                order=order,
                sample_id=sample_id,
                source_archive=_text(row, "source_archive"),
                data_type=data_type,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                system_prompt=prompt,
                model_input=model_input,
                eligible_row=row,
            )
        )
    if len(tasks) != 162 or len({task.sample_id for task in tasks}) != 162:
        raise ValueError("DeepSeek GT input must contain exactly 162 unique rows")
    return tasks


def _prepare_prompt_context_v2_tasks(
    repo_root: str | Path,
    config: GTPromptContextV2Config,
) -> list[GTTask]:
    root = Path(repo_root).resolve()
    manifest_path = (root / config.input_manifest).resolve()
    if _sha256(manifest_path) != config.input_manifest_sha256:
        raise ValueError("Prompt-context v2 input manifest hash mismatch")
    rows = _read_jsonl(manifest_path)
    if len(rows) != config.expected_count:
        raise ValueError(
            f"Prompt-context v2 manifest count mismatch: expected {config.expected_count}, "
            f"got {len(rows)}"
        )
    prompts = {
        "A": (root / config.a_prompt_path).read_text(encoding="utf-8"),
        "C": (root / config.c_prompt_path).read_text(encoding="utf-8"),
    }
    ledger_signature = {
        "schema_name": config.schema_name,
        "run_id": config.run_id,
        "protocol_version": config.protocol_version,
        "input_manifest_sha256": config.input_manifest_sha256,
        "expected_count": config.expected_count,
    }
    tasks: list[GTTask] = []
    seen_ids: set[str] = set()
    for order, row in enumerate(rows):
        _validate_prompt_context_v2_row(row)
        sample_id = _text(row, "sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate prompt-context v2 sample_id: {sample_id}")
        seen_ids.add(sample_id)
        data_type = _text(row, "data_type")
        model_input = {
            "archetype": dict(row["archetype"]),
            "dialogue": row["dialogue"],
            "context": row["context_text"],
            "surface_emotion": row["surface_emotion"],
        }
        _validate_model_input(model_input)
        prompt = prompts[data_type]
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        input_hash = hashlib.sha256(
            _canonical_json(
                {
                    "model": config.model,
                    "prompt_hash": prompt_hash,
                    "input": model_input,
                    "ledger_signature": ledger_signature,
                }
            ).encode()
        ).hexdigest()
        tasks.append(
            GTTask(
                order=order,
                sample_id=sample_id,
                source_archive=_text(row, "source_archive"),
                data_type=data_type,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                system_prompt=prompt,
                model_input=model_input,
                eligible_row=row,
                ledger_signature=dict(ledger_signature),
            )
        )
    return tasks


def select_pilot(tasks: list[GTTask], per_archive: int = 4) -> list[GTTask]:
    selected: list[GTTask] = []
    for archive in ARCHIVE_ORDER:
        rows = [task for task in tasks if task.source_archive == archive]
        if len(rows) < per_archive:
            raise ValueError(f"Not enough pilot rows in {archive}")
        selected.extend(rows[:per_archive])
    return selected


class DeepSeekClient:
    def __init__(
        self,
        config: AnyGTConfig,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_seconds), transport=transport
        )
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def close(self) -> None:
        await self.client.aclose()

    async def complete(self, task: GTTask) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": task.system_prompt},
                {"role": "user", "content": _canonical_json(task.model_input)},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        try:
            response = await self.client.post(self.config.api_url, headers=self.headers, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientAPIError(type(exc).__name__) from exc
        if response.status_code in {408, 409, 429} or response.status_code >= 500:
            raise TransientAPIError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise PermanentAPIError(f"HTTP {response.status_code}")
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise PermanentAPIError("API response is not JSON") from exc
        return _validate_response_envelope(payload, self.config.model)


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            create table if not exists tasks (
              sample_id text primary key, task_order integer not null, source_archive text not null,
              data_type text not null, input_hash text not null, prompt_hash text not null,
              request_json text not null, eligible_json text not null, status text not null,
              attempts integer not null default 0, result_json text, error_type text,
              error_message text, created_at text not null, updated_at text not null
            );
            create table if not exists attempts (
              sample_id text not null, attempt integer not null, started_at text not null,
              ended_at text not null, outcome text not null, response_json text,
              error_type text, error_message text, primary key(sample_id, attempt)
            );
            """
        )
        self.db.execute("update tasks set status='pending' where status='running'")
        self.db.commit()

    def prepare(self, tasks: list[GTTask]) -> None:
        now = _now()
        for task in tasks:
            request_payload = {
                "system_prompt": task.system_prompt,
                "model_input": task.model_input,
            }
            if task.ledger_signature is not None:
                request_payload["ledger_signature"] = task.ledger_signature
            request_json = _canonical_json(request_payload)
            existing = self.db.execute(
                """select input_hash,prompt_hash,request_json,eligible_json
                   from tasks where sample_id=?""",
                (task.sample_id,),
            ).fetchone()
            eligible_json = _canonical_json(task.eligible_row)
            if existing is not None:
                actual = tuple(existing)
                expected = (task.input_hash, task.prompt_hash, request_json, eligible_json)
                if actual != expected:
                    raise ValueError(f"Ledger signature mismatch: {task.sample_id}")
                continue
            self.db.execute(
                "insert into tasks values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task.sample_id,
                    task.order,
                    task.source_archive,
                    task.data_type,
                    task.input_hash,
                    task.prompt_hash,
                    request_json,
                    eligible_json,
                    "pending",
                    0,
                    None,
                    None,
                    None,
                    now,
                    now,
                ),
            )
        self.db.commit()

        expected_ids = {task.sample_id for task in tasks}
        actual_ids = {
            str(row[0]) for row in self.db.execute("select sample_id from tasks")
        }
        if actual_ids != expected_ids:
            unexpected = sorted(actual_ids - expected_ids)
            missing = sorted(expected_ids - actual_ids)
            raise ValueError(
                f"Ledger task set mismatch: unexpected={unexpected[:5]}, missing={missing[:5]}"
            )

    def pending_ids(
        self, selected: set[str], *, include_failed: bool = False
    ) -> list[str]:
        if not selected:
            return []
        statuses = ("pending", "failed") if include_failed else ("pending",)
        placeholders = ",".join("?" for _ in statuses)
        return [
            row[0]
            for row in self.db.execute(
                f"select sample_id from tasks where status in ({placeholders}) order by task_order",
                statuses,
            )
            if row[0] in selected
        ]

    def start(self, sample_id: str) -> int:
        row = self.db.execute(
            "select attempts from tasks where sample_id=?", (sample_id,)
        ).fetchone()
        attempt = int(row[0]) + 1
        self.db.execute(
            """update tasks
               set status='running', attempts=?, updated_at=?, error_type=null,
                   error_message=null
               where sample_id=?""",
            (attempt, _now(), sample_id),
        )
        self.db.commit()
        return attempt

    def finish_attempt(
        self,
        sample_id: str,
        attempt: int,
        started: str,
        outcome: str,
        response: Any = None,
        exc: Exception | None = None,
    ) -> None:
        self.db.execute(
            "insert into attempts values (?,?,?,?,?,?,?,?)",
            (
                sample_id,
                attempt,
                started,
                _now(),
                outcome,
                None if response is None else _canonical_json(response),
                None if exc is None else type(exc).__name__,
                None if exc is None else str(exc),
            ),
        )
        self.db.commit()

    def complete(self, sample_id: str, result: dict[str, Any]) -> None:
        self.db.execute(
            "update tasks set status='completed',result_json=?,updated_at=? where sample_id=?",
            (_canonical_json(result), _now(), sample_id),
        )
        self.db.commit()

    def fail(self, sample_id: str, exc: Exception) -> None:
        self.db.execute(
            """update tasks
               set status='failed', error_type=?, error_message=?, updated_at=?
               where sample_id=?""",
            (type(exc).__name__, str(exc), _now(), sample_id),
        )
        self.db.commit()

    def rows(self) -> list[sqlite3.Row]:
        return list(self.db.execute("select * from tasks order by task_order"))

    def attempt_rows(self) -> list[sqlite3.Row]:
        return list(self.db.execute("select * from attempts order by sample_id,attempt"))

    def close(self) -> None:
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()


async def run_batch(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    mode: Literal["pilot", "full"],
    retry_failed: bool = False,
    client: DeepSeekClient | Any | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> RunResult:
    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    tasks = prepare_tasks(root, config)
    if isinstance(config, GTPromptContextV2Config):
        if mode != "pilot":
            raise ValueError("Prompt-context v2 pilot config only supports --mode pilot")
        selected = tasks
    else:
        selected = select_pilot(tasks) if mode == "pilot" else tasks
    task_by_id = {task.sample_id: task for task in tasks}
    output_root = (root / config.output_root).resolve()
    ledger = Ledger(output_root / "batch_state.sqlite3")
    ledger.prepare(tasks)
    owns_client = client is None
    if client is None:
        client = DeepSeekClient(config, load_api_key(config))
    semaphore = asyncio.Semaphore(config.concurrency)

    async def worker(sample_id: str) -> None:
        task = task_by_id[sample_id]
        async with semaphore:
            last_error: Exception | None = None
            for retry_index in range(len(config.retry_delays_seconds) + 1):
                attempt = ledger.start(sample_id)
                started = _now()
                try:
                    envelope = await client.complete(task)
                    description = validate_gt_content(
                        envelope["content"],
                        min_words=config.min_words,
                        max_words=config.max_words,
                    )
                    result = {**envelope, "GT_DESCRIPTION": description}
                    ledger.finish_attempt(sample_id, attempt, started, "completed", result)
                    ledger.complete(sample_id, result)
                    return
                except TransientAPIError as exc:
                    last_error = exc
                    ledger.finish_attempt(sample_id, attempt, started, "transient_error", exc=exc)
                    if retry_index < len(config.retry_delays_seconds):
                        await sleep(config.retry_delays_seconds[retry_index])
                        continue
                    break
                except Exception as exc:
                    last_error = exc
                    ledger.finish_attempt(sample_id, attempt, started, "failed", exc=exc)
                    break
            assert last_error is not None
            ledger.fail(sample_id, last_error)

    try:
        selected_ids = {task.sample_id for task in selected}
        runnable_ids = ledger.pending_ids(selected_ids, include_failed=retry_failed)
        await asyncio.gather(*(worker(sample_id) for sample_id in runnable_ids))
        _export(output_root, ledger, config, config_file)
        counts = Counter(row["status"] for row in ledger.rows())
        return RunResult(
            total=len(ledger.rows()),
            completed=counts["completed"],
            failed=counts["failed"],
            pending=counts["pending"],
            output_root=output_root,
        )
    finally:
        if owns_client:
            await client.close()
        ledger.close()


def validate_gt_content(content: Any, *, min_words: int, max_words: int) -> str:
    if not isinstance(content, str) or not content:
        raise GTValidationError("response content must be a non-empty string")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GTValidationError("response content must be exact JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"GT_DESCRIPTION"}:
        raise GTValidationError("response JSON must contain exactly GT_DESCRIPTION")
    value = payload["GT_DESCRIPTION"]
    if not isinstance(value, str) or not value or value != value.strip() or "\n" in value:
        raise GTValidationError("GT_DESCRIPTION must be one non-empty unpadded line")
    if not value.endswith(".") or any(mark in value for mark in "?!"):
        raise GTValidationError(
            "GT_DESCRIPTION must be a declarative sentence ending in a period"
        )
    if len(re.findall(r"[.!?](?=\s|$)", value)) != 1:
        raise GTValidationError("GT_DESCRIPTION must contain exactly one sentence")
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", value)
    if not min_words <= len(words) <= max_words:
        raise GTValidationError(
            f"GT_DESCRIPTION must contain {min_words}-{max_words} English words"
        )
    return value


def verify_outputs(
    repo_root: str | Path,
    config_path: str | Path,
    *,
    require_complete: bool,
) -> RunResult:
    root = Path(repo_root).resolve()
    config = load_config(config_path)
    tasks = prepare_tasks(root, config)
    output_root = (root / config.output_root).resolve()
    manifest = _read_jsonl(output_root / "gt_manifest.jsonl")
    sidecar = _read_jsonl(output_root / "review_status.jsonl")
    failures = _read_jsonl(output_root / "failures.jsonl")
    provenance = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    completed = len(manifest)
    expected_count = _expected_count(config)
    if require_complete and (
        completed != expected_count or failures or len(sidecar) != expected_count
    ):
        raise ValueError(
            f"Complete GT export must contain {expected_count} completed rows and no failures"
        )
    eligible_by_id = {task.sample_id: task.eligible_row for task in tasks}
    task_ids = set(eligible_by_id)
    manifest_ids = [_text(row, "sample_id") for row in manifest]
    sidecar_ids = [_text(row, "sample_id") for row in sidecar]
    failure_ids = [_text(row, "sample_id") for row in failures]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("GT manifest contains duplicate sample_id values")
    if len(sidecar_ids) != len(set(sidecar_ids)):
        raise ValueError("GT review sidecar contains duplicate sample_id values")
    if not set(manifest_ids + sidecar_ids + failure_ids) <= task_ids:
        raise ValueError("GT export contains unknown sample_id values")
    if set(sidecar_ids) != set(manifest_ids):
        raise ValueError("GT review sidecar must exactly match completed manifest rows")
    for row in manifest:
        sample_id = _text(row, "sample_id")
        expected = eligible_by_id[sample_id]
        if set(row) != set(expected) | {"GT_DESCRIPTION"}:
            raise ValueError(f"Final GT manifest fields differ for {sample_id}")
        if {key: row[key] for key in expected} != expected:
            raise ValueError(f"Final GT manifest changed eligible fields for {sample_id}")
        validate_gt_content(
            _canonical_json({"GT_DESCRIPTION": row["GT_DESCRIPTION"]}),
            min_words=config.min_words,
            max_words=config.max_words,
        )
    for row in sidecar:
        if set(row) != {
            "sample_id",
            "annotation_status",
            "human_review_status",
            "gt_description_sha256",
        }:
            raise ValueError(f"Unexpected GT review sidecar fields: {row.get('sample_id')}")
        if row["annotation_status"] != "preliminary_ai_draft":
            raise ValueError(f"Unexpected annotation status: {row['sample_id']}")
        if row["human_review_status"] != "pending_human":
            raise ValueError(f"Unexpected human review status: {row['sample_id']}")
    expected_provenance_schema = (
        "mprisk_deepseek_gt_provenance_v2"
        if isinstance(config, GTPromptContextV2Config)
        else "mprisk_deepseek_gt_provenance_v1"
    )
    if provenance.get("schema_name") != expected_provenance_schema:
        raise ValueError("Unexpected GT provenance schema")
    if provenance.get("run_id") != config.run_id or provenance.get("model") != config.model:
        raise ValueError("GT provenance run identity mismatch")
    for artifact in provenance["artifacts"].values():
        path = root / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"GT artifact hash mismatch: {path}")
    return RunResult(
        total=expected_count,
        completed=completed,
        failed=len(failures),
        pending=expected_count - completed - len(failures),
        output_root=output_root,
    )


def _export(
    output_root: Path,
    ledger: Ledger,
    config: AnyGTConfig,
    config_file: Path,
) -> None:
    rows = ledger.rows()
    manifest: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    ledger_export: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        ledger_export.append(
            {
                key: record[key]
                for key in record
                if key not in {"request_json", "eligible_json", "result_json"}
            }
        )
        if row["status"] == "completed":
            eligible = json.loads(row["eligible_json"])
            result = json.loads(row["result_json"])
            manifest.append({**eligible, "GT_DESCRIPTION": result["GT_DESCRIPTION"]})
            raw.append(
                {
                    "sample_id": row["sample_id"],
                    "input_hash": row["input_hash"],
                    "prompt_hash": row["prompt_hash"],
                    "request": json.loads(row["request_json"]),
                    "response": result,
                }
            )
            sidecar.append(
                {
                    "sample_id": row["sample_id"],
                    "annotation_status": "preliminary_ai_draft",
                    "human_review_status": "pending_human",
                    "gt_description_sha256": hashlib.sha256(
                        result["GT_DESCRIPTION"].encode()
                    ).hexdigest(),
                }
            )
        elif row["status"] == "failed":
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "error_type": row["error_type"],
                    "error_message": row["error_message"],
                }
            )
    payloads = {
        "gt_manifest.jsonl": _jsonl(manifest),
        "raw_responses.jsonl": _jsonl(raw),
        "review_status.jsonl": _jsonl(sidecar),
        "failures.jsonl": _jsonl(failures),
        "ledger.jsonl": _jsonl(ledger_export),
        "attempts.jsonl": _jsonl([dict(row) for row in ledger.attempt_rows()]),
    }
    artifacts = {
        name.removesuffix(".jsonl"): {
            "path": (config.output_root / name).as_posix(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in payloads.items()
    }
    provenance: dict[str, Any] = {
        "schema_name": (
            "mprisk_deepseek_gt_provenance_v2"
            if isinstance(config, GTPromptContextV2Config)
            else "mprisk_deepseek_gt_provenance_v1"
        ),
        "run_id": config.run_id,
        "model": config.model,
        "thinking": config.thinking,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "config_sha256": _sha256(config_file),
        "counts": dict(Counter(row["status"] for row in rows)),
        "artifacts": artifacts,
    }
    if isinstance(config, GTPromptContextV2Config):
        provenance["protocol_version"] = config.protocol_version
        provenance["input_manifest"] = config.input_manifest.as_posix()
        provenance["input_manifest_sha256"] = config.input_manifest_sha256
        provenance["expected_count"] = config.expected_count
    provenance_json = json.dumps(
        provenance, ensure_ascii=False, sort_keys=True, indent=2
    )
    payloads["provenance.json"] = (provenance_json + "\n").encode()
    for name, content in payloads.items():
        _atomic_write(output_root / name, content)


def _validate_response_envelope(payload: Any, model: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("model") != model:
        raise PermanentAPIError("API returned an unexpected model")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise PermanentAPIError("API must return exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise PermanentAPIError(f"Unexpected finish_reason: {choice.get('finish_reason')!r}")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("reasoning_content") not in (None, ""):
        raise PermanentAPIError("Thinking was not disabled")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise PermanentAPIError("API returned empty content")
    return {
        "response_id": payload.get("id"), "response_model": payload.get("model"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"), "content": content,
        "usage": payload.get("usage") or {},
    }


def _validate_model_input(payload: dict[str, Any]) -> None:
    allowed_fields = (
        {"archetype", "trigger_context", "dialogue", "surface_emotion"},
        {"archetype", "context", "dialogue", "surface_emotion"},
    )
    if set(payload) not in allowed_fields:
        raise ValueError("Model input contains forbidden fields")
    if set(payload["archetype"]) != {"id", "name", "canonical_meaning"}:
        raise ValueError("Archetype model input contains forbidden fields")
    for key in set(payload) - {"archetype", "surface_emotion"}:
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if payload["surface_emotion"] is not None and not isinstance(payload["surface_emotion"], str):
        raise TypeError("surface_emotion must be a string or null")


def _validate_prompt_context_v2_row(row: dict[str, Any]) -> None:
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
    if set(row) != expected_fields:
        raise ValueError("Prompt-context v2 manifest fields are not strict")
    if row.get("schema_name") != "mprisk_gt_prompt_context_v2_pilot_row":
        raise ValueError("Prompt-context v2 row schema mismatch")
    if row.get("protocol_version") != "prompt_context_v2":
        raise ValueError("Prompt-context v2 protocol version mismatch")
    if row.get("context_source") not in {"setting", "trigger", "ltx2_prompt"}:
        raise ValueError("Prompt-context v2 context_source is invalid")
    data_type = _text(row, "data_type")
    protocol = _text(row, "protocol")
    source_archive = _text(row, "source_archive")
    expected_archive = {
        ("A", "VT"): "accept_a_svt",
        ("A", "VA"): "accept_a_va",
        ("C", "VT"): "accept_c_svt",
        ("C", "VA"): "accept_c_va",
    }.get((data_type, protocol))
    if expected_archive != source_archive:
        raise ValueError("Prompt-context v2 archive/data_type/protocol mismatch")
    if not isinstance(row.get("archetype"), dict) or set(row["archetype"]) != {
        "id",
        "name",
        "canonical_meaning",
    }:
        raise ValueError("Prompt-context v2 archetype fields are not strict")
    for key in ("id", "name", "canonical_meaning"):
        _text(row["archetype"], key)
    for key in ("sample_id", "dialogue", "context_text", "source_row_sha256"):
        _text(row, key)
    media = row.get("media")
    if not isinstance(media, dict) or set(media) != {"path", "sha256"}:
        raise ValueError("Prompt-context v2 media fields are not strict")
    media_path = Path(_text(media, "path"))
    if not media_path.is_file() or _sha256(media_path) != _text(media, "sha256"):
        raise ValueError(f"Prompt-context v2 media hash mismatch: {row['sample_id']}")
    assignment = row.get("source_assignment")
    assignment_fields = {
        "path",
        "schema_name",
        "dictionary_id",
        "assignment_source",
        "source_row_sha256",
        "assignment_sha256",
    }
    if not isinstance(assignment, dict) or set(assignment) != assignment_fields:
        raise ValueError("Prompt-context v2 source assignment fields are not strict")
    if assignment.get("source_row_sha256") != row["source_row_sha256"]:
        raise ValueError("Prompt-context v2 source assignment hash mismatch")
    for key in assignment_fields:
        _text(assignment, key)
    if row["surface_emotion"] is not None and not isinstance(row["surface_emotion"], str):
        raise TypeError("Prompt-context v2 surface_emotion must be a string or null")


def _expected_count(config: AnyGTConfig) -> int:
    return config.expected_count if isinstance(config, GTPromptContextV2Config) else 162


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            raise ValueError(f"Invalid env line in {path}")
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"JSONL rows must be objects: {path}")
    return rows


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _text(row, "sample_id")
        if key in result:
            raise ValueError(f"Duplicate sample_id: {key}")
        result[key] = row
    return result


def _text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(_canonical_json(row)+"\n" for row in rows).encode()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()
