"""Provider-independent, strict, resumable GT Description generation."""

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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from mprisk.config.loader import load_yaml
from mprisk.data.generated_archive_freeze import _canonical_json, _sha256
from mprisk.ground_truth.annotation_inputs import (
    GT_INPUT_SCHEMA_VERSION,
    GTAnnotationInput,
)
from mprisk.ground_truth.providers.base import (
    GTDescriptionProvider,
    GTDescriptionProviderRequest,
    GTDescriptionProviderResponse,
    TransientProviderError,
)
from mprisk.ground_truth.providers.registry import (
    get_provider,
    validate_provider_settings,
)

CONFIG_SCHEMA = "mprisk_gt_description_generation_config_v3"
OUTPUT_SCHEMA = "mprisk_gt_description_v1"
PROVENANCE_SCHEMA = "mprisk_gt_description_generation_provenance_v3"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GTDescriptionGenerationConfig(_StrictModel):
    schema_name: Literal["mprisk_gt_description_generation_config_v3"]
    run_id: str
    provider_key: str
    gt_generator_model: str
    provider_settings: dict[str, Any]
    concurrency: int
    retry_delays_seconds: list[float]
    min_words: int
    max_words: int
    gt_input_schema_version: Literal["gt_annotation_input_v1"]
    input_manifest: Path
    input_manifest_sha256: str
    expected_count: int
    output_root: Path
    conflict_prompt_path: Path
    aligned_prompt_path: Path

    @field_validator("run_id")
    @classmethod
    def run_id_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run_id must be non-empty")
        return value

    @field_validator("provider_key", "gt_generator_model")
    @classmethod
    def provider_identity_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider_key and gt_generator_model must be non-empty")
        return value

    @field_validator("concurrency", "expected_count")
    @classmethod
    def positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("concurrency and expected_count must be positive")
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

    @field_validator("input_manifest_sha256")
    @classmethod
    def manifest_hash_must_be_sha256(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("input_manifest_sha256 must be a lowercase SHA-256 digest")
        return value


@dataclass(frozen=True)
class GTDescriptionGenerationTask:
    order: int
    sample_id: str
    sample_type: Literal["Conflict", "Aligned"]
    source_archive: str
    input_hash: str
    prompt_hash: str
    system_prompt: str
    model_input: dict[str, Any]
    annotation_input_row: dict[str, Any]
    ledger_signature: dict[str, Any]


@dataclass(frozen=True)
class GTDescriptionGenerationResult:
    total: int
    completed: int
    failed: int
    pending: int
    output_root: Path


class GTDescriptionValidationError(ValueError):
    pass


def load_config(path: str | Path) -> GTDescriptionGenerationConfig:
    payload = load_yaml(path)
    if payload.get("schema_name") != CONFIG_SCHEMA:
        raise ValueError(
            f"Unsupported GT Description generation config: {payload.get('schema_name')!r}"
        )
    config = GTDescriptionGenerationConfig.model_validate(payload)
    validate_provider_settings(config.provider_key, config.provider_settings)
    return config


def prepare_tasks(
    repo_root: str | Path,
    config: GTDescriptionGenerationConfig,
) -> list[GTDescriptionGenerationTask]:
    root = Path(repo_root).resolve()
    manifest_path = _resolve_repo_path(root, config.input_manifest)
    if _sha256(manifest_path) != config.input_manifest_sha256:
        raise ValueError("GT annotation input manifest hash mismatch")
    rows = _read_jsonl(manifest_path)
    if len(rows) != config.expected_count:
        raise ValueError(
            f"GT annotation input count mismatch: expected {config.expected_count}, "
            f"got {len(rows)}"
        )
    prompts = {
        "Conflict": _resolve_repo_path(root, config.conflict_prompt_path).read_text(
            encoding="utf-8"
        ),
        "Aligned": _resolve_repo_path(root, config.aligned_prompt_path).read_text(
            encoding="utf-8"
        ),
    }
    ledger_signature = {
        "schema_name": config.schema_name,
        "run_id": config.run_id,
        "provider_key": config.provider_key,
        "gt_generator_model": config.gt_generator_model,
        "provider_settings_sha256": hashlib.sha256(
            _canonical_json(config.provider_settings).encode()
        ).hexdigest(),
        "gt_input_schema_version": config.gt_input_schema_version,
        "input_manifest_sha256": config.input_manifest_sha256,
        "expected_count": config.expected_count,
    }
    tasks: list[GTDescriptionGenerationTask] = []
    seen_ids: set[str] = set()
    for order, raw_row in enumerate(rows):
        annotation_input = GTAnnotationInput.model_validate(raw_row)
        row = annotation_input.model_dump(mode="json")
        sample_id = annotation_input.sample_id
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate GT annotation sample_id: {sample_id}")
        seen_ids.add(sample_id)
        sample_type = annotation_input.sample_type
        model_input = {
            "archetype": annotation_input.archetype.model_dump(mode="json"),
            "dialogue": annotation_input.dialogue,
            "scenario_context": annotation_input.scenario_context,
            "surface_emotion": annotation_input.surface_emotion,
        }
        _validate_model_input(model_input)
        prompt = prompts[sample_type]
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        input_hash = hashlib.sha256(
            _canonical_json(
                {
                    "gt_generator_model": config.gt_generator_model,
                    "prompt_hash": prompt_hash,
                    "model_input": model_input,
                    "ledger_signature": ledger_signature,
                }
            ).encode()
        ).hexdigest()
        tasks.append(
            GTDescriptionGenerationTask(
                order=order,
                sample_id=sample_id,
                sample_type=sample_type,
                source_archive=annotation_input.source_provenance.source_archive,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                system_prompt=prompt,
                model_input=model_input,
                annotation_input_row=row,
                ledger_signature=dict(ledger_signature),
            )
        )
    return tasks


class GTDescriptionGenerationLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            create table if not exists tasks (
              sample_id text primary key, task_order integer not null,
              source_archive text not null, sample_type text not null,
              input_hash text not null, prompt_hash text not null,
              request_json text not null, annotation_input_json text not null,
              status text not null, attempts integer not null default 0,
              result_json text, error_type text, error_message text,
              created_at text not null, updated_at text not null
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

    def prepare(self, tasks: list[GTDescriptionGenerationTask]) -> None:
        now = _now()
        for task in tasks:
            request_json = _canonical_json(
                {
                    "system_prompt": task.system_prompt,
                    "model_input": task.model_input,
                    "ledger_signature": task.ledger_signature,
                }
            )
            annotation_input_json = _canonical_json(task.annotation_input_row)
            existing = self.db.execute(
                """select input_hash,prompt_hash,request_json,annotation_input_json
                   from tasks where sample_id=?""",
                (task.sample_id,),
            ).fetchone()
            if existing is not None:
                expected = (
                    task.input_hash,
                    task.prompt_hash,
                    request_json,
                    annotation_input_json,
                )
                if tuple(existing) != expected:
                    raise ValueError(f"Ledger signature mismatch: {task.sample_id}")
                continue
            self.db.execute(
                "insert into tasks values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task.sample_id,
                    task.order,
                    task.source_archive,
                    task.sample_type,
                    task.input_hash,
                    task.prompt_hash,
                    request_json,
                    annotation_input_json,
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
        actual_ids = {str(row[0]) for row in self.db.execute("select sample_id from tasks")}
        if actual_ids != expected_ids:
            unexpected = sorted(actual_ids - expected_ids)
            missing = sorted(expected_ids - actual_ids)
            raise ValueError(
                f"Ledger task set mismatch: unexpected={unexpected[:5]}, missing={missing[:5]}"
            )

    def pending_ids(self, *, include_failed: bool = False) -> list[str]:
        statuses = ("pending", "failed") if include_failed else ("pending",)
        placeholders = ",".join("?" for _ in statuses)
        return [
            str(row[0])
            for row in self.db.execute(
                f"select sample_id from tasks where status in ({placeholders}) "
                "order by task_order",
                statuses,
            )
        ]

    def start(self, sample_id: str) -> int:
        row = self.db.execute(
            "select attempts from tasks where sample_id=?", (sample_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown ledger sample_id: {sample_id}")
        attempt = int(row[0]) + 1
        self.db.execute(
            """update tasks set status='running',attempts=?,updated_at=?,
               error_type=null,error_message=null where sample_id=?""",
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
            """update tasks set status='failed',error_type=?,error_message=?,updated_at=?
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


async def run_gt_description_generation(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    retry_failed: bool = False,
    provider: GTDescriptionProvider | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> GTDescriptionGenerationResult:
    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    tasks = prepare_tasks(root, config)
    task_by_id = {task.sample_id: task for task in tasks}
    output_root = _resolve_repo_path(root, config.output_root)
    ledger = GTDescriptionGenerationLedger(output_root / "batch_state.sqlite3")
    ledger.prepare(tasks)
    owns_provider = provider is None
    if provider is None:
        provider = get_provider(
            config.provider_key,
            config.gt_generator_model,
            config.provider_settings,
        )
    semaphore = asyncio.Semaphore(config.concurrency)

    async def worker(sample_id: str) -> None:
        task = task_by_id[sample_id]
        async with semaphore:
            last_error: Exception | None = None
            for retry_index in range(len(config.retry_delays_seconds) + 1):
                attempt = ledger.start(sample_id)
                started = _now()
                response: GTDescriptionProviderResponse | None = None
                try:
                    request = GTDescriptionProviderRequest(
                        model=config.gt_generator_model,
                        system_prompt=task.system_prompt,
                        model_input=task.model_input,
                    )
                    response = await provider.complete(request)
                    description = validate_gt_description_content(
                        response.content,
                        min_words=config.min_words,
                        max_words=config.max_words,
                    )
                    result = {**asdict(response), "GT_DESCRIPTION": description}
                    ledger.finish_attempt(sample_id, attempt, started, "completed", result)
                    ledger.complete(sample_id, result)
                    return
                except TransientProviderError as exc:
                    last_error = exc
                    ledger.finish_attempt(
                        sample_id, attempt, started, "transient_error", exc=exc
                    )
                    if retry_index < len(config.retry_delays_seconds):
                        await sleep(config.retry_delays_seconds[retry_index])
                        continue
                    break
                except Exception as exc:
                    last_error = exc
                    ledger.finish_attempt(
                        sample_id,
                        attempt,
                        started,
                        "failed",
                        response=None if response is None else asdict(response),
                        exc=exc,
                    )
                    break
            if last_error is None:
                raise RuntimeError(f"GT task ended without a result: {sample_id}")
            ledger.fail(sample_id, last_error)

    try:
        runnable_ids = ledger.pending_ids(include_failed=retry_failed)
        await asyncio.gather(*(worker(sample_id) for sample_id in runnable_ids))
        _export(output_root, ledger, config, config_file)
        counts = Counter(row["status"] for row in ledger.rows())
        return GTDescriptionGenerationResult(
            total=len(ledger.rows()),
            completed=counts["completed"],
            failed=counts["failed"],
            pending=counts["pending"],
            output_root=output_root,
        )
    finally:
        if owns_provider:
            await provider.close()
        ledger.close()


def validate_gt_description_content(content: Any, *, min_words: int, max_words: int) -> str:
    if not isinstance(content, str) or not content:
        raise GTDescriptionValidationError("response content must be a non-empty string")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GTDescriptionValidationError("response content must be exact JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"GT_DESCRIPTION"}:
        raise GTDescriptionValidationError(
            "response JSON must contain exactly GT_DESCRIPTION"
        )
    value = payload["GT_DESCRIPTION"]
    if not isinstance(value, str) or not value or value != value.strip() or "\n" in value:
        raise GTDescriptionValidationError(
            "GT_DESCRIPTION must be one non-empty unpadded line"
        )
    if not value.endswith(".") or any(mark in value for mark in "?!"):
        raise GTDescriptionValidationError(
            "GT_DESCRIPTION must be a declarative sentence ending in a period"
        )
    if len(re.findall(r"[.!?](?=\s|$)", value)) != 1:
        raise GTDescriptionValidationError("GT_DESCRIPTION must contain exactly one sentence")
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", value)
    if not min_words <= len(words) <= max_words:
        raise GTDescriptionValidationError(
            f"GT_DESCRIPTION must contain {min_words}-{max_words} English words"
        )
    return value


def verify_gt_description_generation(
    repo_root: str | Path,
    config_path: str | Path,
    *,
    require_complete: bool,
) -> GTDescriptionGenerationResult:
    root = Path(repo_root).resolve()
    config = load_config(config_path)
    tasks = prepare_tasks(root, config)
    output_root = _resolve_repo_path(root, config.output_root)
    manifest = _read_jsonl(output_root / "gt_manifest.jsonl")
    sidecar = _read_jsonl(output_root / "review_status.jsonl")
    failures = _read_jsonl(output_root / "failures.jsonl")
    provenance = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    completed = len(manifest)
    if require_complete and (
        completed != config.expected_count
        or failures
        or len(sidecar) != config.expected_count
    ):
        raise ValueError(
            f"Complete GT export must contain {config.expected_count} rows and no failures"
        )
    input_by_id = {task.sample_id: task.annotation_input_row for task in tasks}
    task_ids = set(input_by_id)
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
        expected = input_by_id[sample_id]
        if set(row) != set(expected) | {"run_id", "GT_DESCRIPTION"}:
            raise ValueError(f"Final GT manifest fields differ for {sample_id}")
        expected_output = {
            **expected,
            "schema_name": OUTPUT_SCHEMA,
            "run_id": config.run_id,
        }
        if {key: row[key] for key in expected_output} != expected_output:
            raise ValueError(f"Final GT manifest changed annotation input fields for {sample_id}")
        validate_gt_description_content(
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
    if provenance.get("schema_name") != PROVENANCE_SCHEMA:
        raise ValueError("Unexpected GT provenance schema")
    if (
        provenance.get("run_id") != config.run_id
        or provenance.get("provider_key") != config.provider_key
        or provenance.get("gt_generator_model") != config.gt_generator_model
        or provenance.get("gt_description_schema_name") != OUTPUT_SCHEMA
        or provenance.get("gt_input_schema_version") != GT_INPUT_SCHEMA_VERSION
        or provenance.get("provider_settings_sha256")
        != hashlib.sha256(_canonical_json(config.provider_settings).encode()).hexdigest()
    ):
        raise ValueError("GT provenance run identity mismatch")
    for artifact in provenance["artifacts"].values():
        path = root / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"GT artifact hash mismatch: {path}")
    return GTDescriptionGenerationResult(
        total=config.expected_count,
        completed=completed,
        failed=len(failures),
        pending=config.expected_count - completed - len(failures),
        output_root=output_root,
    )


def _export(
    output_root: Path,
    ledger: GTDescriptionGenerationLedger,
    config: GTDescriptionGenerationConfig,
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
                if key not in {"request_json", "annotation_input_json", "result_json"}
            }
        )
        if row["status"] == "completed":
            annotation_input = json.loads(row["annotation_input_json"])
            result = json.loads(row["result_json"])
            manifest.append(
                {
                    **annotation_input,
                    "schema_name": OUTPUT_SCHEMA,
                    "run_id": config.run_id,
                    "GT_DESCRIPTION": result["GT_DESCRIPTION"],
                }
            )
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
    provenance = {
        "schema_name": PROVENANCE_SCHEMA,
        "run_id": config.run_id,
        "provider_key": config.provider_key,
        "gt_generator_model": config.gt_generator_model,
        "gt_description_schema_name": OUTPUT_SCHEMA,
        "gt_input_schema_version": config.gt_input_schema_version,
        "provider_settings_sha256": hashlib.sha256(
            _canonical_json(config.provider_settings).encode()
        ).hexdigest(),
        "config_sha256": _sha256(config_file),
        "input_manifest": config.input_manifest.as_posix(),
        "input_manifest_sha256": config.input_manifest_sha256,
        "expected_count": config.expected_count,
        "counts": dict(Counter(row["status"] for row in rows)),
        "artifacts": artifacts,
    }
    payloads["provenance.json"] = (
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    for name, content in payloads.items():
        _atomic_write(output_root / name, content)


def _validate_model_input(payload: dict[str, Any]) -> None:
    if set(payload) != {
        "archetype",
        "dialogue",
        "scenario_context",
        "surface_emotion",
    }:
        raise ValueError("GT generator model input contains forbidden fields")
    if set(payload["archetype"]) != {"id", "name", "canonical_meaning"}:
        raise ValueError("Archetype model input contains forbidden fields")
    for key in ("dialogue", "scenario_context"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if payload["surface_emotion"] is not None and not isinstance(
        payload["surface_emotion"], str
    ):
        raise TypeError("surface_emotion must be a string or null")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"JSONL rows must be objects: {path}")
    return rows


def _text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode()


def _resolve_repo_path(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"GT Description path escapes repository: {resolved}")
    return resolved


def _atomic_write(path: Path, content: bytes) -> None:
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
