"""Strict, blinded Misread judgment from two canonical descriptions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from mprisk.config.loader import load_yaml

MISREAD_JUDGMENT_PROMPT = (
    "Compare the reference description with the diagnostic affect description. Return MISREAD "
    "when the diagnostic is led by surface cues, contradicts the primary affect, wrongly "
    "compresses distinct affects, omits a decisive component, or gives a confidently opposite "
    "account. Return NON_MISREAD when the core affect is compatible, synonymous, or a valid "
    "simplification. Return UNCERTAIN only when the comparison cannot decide. Return exact JSON "
    "with decision, confidence, and one short rationale sentence."
)
CONFIG_SCHEMA = "mprisk_misread_judgment_config_v2"
PROVENANCE_SCHEMA = "mprisk_misread_judgment_provenance_v2"
SIGNATURE_SCHEMA = "mprisk_misread_judgment_signature_v2"
DECISIONS = frozenset({"MISREAD", "NON_MISREAD", "UNCERTAIN"})
FINAL_DECISIONS = frozenset({"MISREAD", "NON_MISREAD"})
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


class MisreadJudgmentValidationError(ValueError):
    """The service did not return the required strict decision object."""


class MisreadJudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_name: Literal["mprisk_misread_judgment_config_v2"]
    run_id: str
    status: Literal["pending", "ready"]
    judge_model: Literal["deepseek-v4-flash"]
    subject_model_key: str
    protocol: Literal["VT", "VA"]
    split: str
    api_url: str
    temperature: Literal[0]
    confidence_threshold: Literal[0.5]
    gt_description_manifest_path: Path
    diagnostic_affect_description_manifest_path: Path
    output_root: Path
    request_timeout_seconds: float

    @field_validator("request_timeout_seconds")
    @classmethod
    def positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        return value

    @field_validator("run_id", "subject_model_key", "split")
    @classmethod
    def identity_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Misread judgment identity fields must be non-empty")
        return value


@dataclass(frozen=True)
class MisreadJudgeTask:
    sample_id: str
    request: dict[str, Any]
    input_hash: str
    request_hash: str


def load_config(path: str | Path) -> MisreadJudgeConfig:
    return MisreadJudgeConfig.model_validate(load_yaml(path))


def load_api_key(_: MisreadJudgeConfig) -> str:
    value = os.environ.get("DEEPSEEK_API_KEY")
    if not value:
        raise ValueError("DEEPSEEK_API_KEY is required for Misread Judgment")
    return value


def build_tasks(config: MisreadJudgeConfig) -> list[MisreadJudgeTask]:
    references = _index_rows(
        _read_jsonl(config.gt_description_manifest_path), "GT_DESCRIPTION"
    )
    diagnostics = _index_rows(
        _read_jsonl(config.diagnostic_affect_description_manifest_path),
        "DIAGNOSTIC_AFFECT_DESCRIPTION",
    )
    if not references or set(references) != set(diagnostics):
        raise ValueError("GT and Diagnostic Affect Description manifests must have identical IDs")
    tasks: list[MisreadJudgeTask] = []
    for sample_id in sorted(references):
        reference = _required_text(references[sample_id], "GT_DESCRIPTION")
        diagnostic_row = diagnostics[sample_id]
        diagnostic = _required_text(
            diagnostic_row, "DIAGNOSTIC_AFFECT_DESCRIPTION"
        )
        if (
            diagnostic_row.get("subject_model_key") != config.subject_model_key
            or diagnostic_row.get("protocol") != config.protocol
            or diagnostic_row.get("split") != config.split
        ):
            raise ValueError(f"Diagnostic manifest identity mismatch: {sample_id}")
        blind_payload = {
            "GT_DESCRIPTION": reference,
            "DIAGNOSTIC_AFFECT_DESCRIPTION": diagnostic,
        }
        request = {
            "model": config.judge_model,
            "messages": [
                {"role": "system", "content": MISREAD_JUDGMENT_PROMPT},
                {"role": "user", "content": _canonical_json(blind_payload)},
            ],
            "temperature": config.temperature,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        encoded = _canonical_json(request)
        _validate_blind_request(encoded, sample_id)
        tasks.append(
            MisreadJudgeTask(
                sample_id=sample_id,
                request=request,
                input_hash=_hash_text(_canonical_json(blind_payload)),
                request_hash=_hash_text(encoded),
            )
        )
    return tasks


class MisreadJudgeClient:
    def __init__(self, config: MisreadJudgeConfig, api_key: str) -> None:
        self.config = config
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(config.request_timeout_seconds))
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def complete(self, task: MisreadJudgeTask) -> str:
        try:
            response = await self.client.post(
                self.config.api_url, headers=self.headers, json=task.request
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RuntimeError(type(exc).__name__) from exc
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}")
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise MisreadJudgmentValidationError("API response envelope is not JSON") from exc
        if payload.get("model") != self.config.judge_model:
            raise MisreadJudgmentValidationError("API response model does not match fixed model")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise MisreadJudgmentValidationError("API response must contain exactly one choice")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise MisreadJudgmentValidationError("API response has no string content")
        return content

    async def close(self) -> None:
        await self.client.aclose()


class MisreadJudgeLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks (
              sample_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL, request_hash TEXT NOT NULL,
              request_json TEXT NOT NULL, status TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0, result_json TEXT, raw_response TEXT,
              error_type TEXT, error_message TEXT, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
              sample_id TEXT NOT NULL, attempt INTEGER NOT NULL, started_at TEXT NOT NULL,
              finished_at TEXT, outcome TEXT NOT NULL, raw_response TEXT,
              error_type TEXT, error_message TEXT, PRIMARY KEY(sample_id, attempt)
            );
            """
        )

    def prepare(self, signature: dict[str, Any], *, retry_failed: bool = False) -> None:
        value = _canonical_json(signature)
        with self.connection:
            current = self.connection.execute(
                "SELECT value FROM metadata WHERE key='signature'"
            ).fetchone()
            if current is not None and current["value"] != value:
                raise ValueError("Existing judge ledger signature does not match")
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('signature',?)", (value,)
            )
            self.connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")
            if retry_failed:
                self.connection.execute("UPDATE tasks SET status='pending' WHERE status='failed'")

    def add_tasks(self, tasks: Sequence[MisreadJudgeTask]) -> None:
        with self.connection:
            for task in tasks:
                existing = self.connection.execute(
                    "SELECT input_hash,request_hash,request_json FROM tasks WHERE sample_id=?",
                    (task.sample_id,),
                ).fetchone()
                expected = (task.input_hash, task.request_hash, _canonical_json(task.request))
                if existing is not None:
                    if tuple(existing) != expected:
                        raise ValueError(f"Judge task signature mismatch: {task.sample_id}")
                    continue
                self.connection.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,?,'0',NULL,NULL,NULL,NULL,?)".replace(
                        "'0'", "0"
                    ),
                    (
                        task.sample_id,
                        task.input_hash,
                        task.request_hash,
                        _canonical_json(task.request),
                        "pending",
                        _now(),
                    ),
                )
            observed = {row[0] for row in self.connection.execute("SELECT sample_id FROM tasks")}
            expected_ids = {task.sample_id for task in tasks}
            if observed != expected_ids:
                raise ValueError("Judge ledger task set does not match frozen manifests")

    def pending(self, *, retry_failed: bool = False) -> list[str]:
        statuses = ("pending", "failed") if retry_failed else ("pending",)
        placeholders = ",".join("?" for _ in statuses)
        return [
            row[0]
            for row in self.connection.execute(
                f"SELECT sample_id FROM tasks WHERE status IN ({placeholders}) ORDER BY sample_id",
                statuses,
            )
        ]

    def start(self, sample_id: str) -> int:
        row = self.connection.execute(
            "SELECT attempts FROM tasks WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if row is None:
            raise KeyError(sample_id)
        attempt = int(row[0]) + 1
        with self.connection:
            self.connection.execute(
                "UPDATE tasks SET status='running',attempts=?,updated_at=? WHERE sample_id=?",
                (attempt, _now(), sample_id),
            )
        return attempt

    def complete(
        self, sample_id: str, attempt: int, started_at: str, raw: str, result: dict[str, Any]
    ) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE tasks SET status='completed',result_json=?,raw_response=?,"
                "updated_at=? WHERE sample_id=?",
                (_canonical_json(result), raw, _now(), sample_id),
            )
            self.connection.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)",
                (sample_id, attempt, started_at, _now(), "completed", raw, None, None),
            )

    def fail(self, sample_id: str, attempt: int, started_at: str, exc: Exception) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE tasks SET status='failed',error_type=?,error_message=?,"
                "updated_at=? WHERE sample_id=?",
                (type(exc).__name__, str(exc), _now(), sample_id),
            )
            self.connection.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)",
                (
                    sample_id,
                    attempt,
                    started_at,
                    _now(),
                    "failed",
                    None,
                    type(exc).__name__,
                    str(exc),
                ),
            )

    def rows(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM tasks ORDER BY sample_id"))

    def attempt_rows(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM attempts ORDER BY sample_id,attempt"))

    def close(self) -> None:
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


async def run_misread_judgment(
    *, config: MisreadJudgeConfig, client: Any | None = None, retry_failed: bool = False
) -> dict[str, int]:
    if config.status != "ready":
        raise ValueError("Misread judgment config is pending required manifests")
    tasks = build_tasks(config)
    signature = _signature(config, tasks)
    ledger = MisreadJudgeLedger(config.output_root / "batch_state.sqlite3")
    ledger.prepare(signature, retry_failed=retry_failed)
    ledger.add_tasks(tasks)
    by_id = {task.sample_id: task for task in tasks}
    owns_client = client is None
    if client is None:
        client = MisreadJudgeClient(config, load_api_key(config))
    try:
        for sample_id in ledger.pending(retry_failed=retry_failed):
            attempt = ledger.start(sample_id)
            started_at = _now()
            try:
                raw = await client.complete(by_id[sample_id])
                ledger.complete(
                    sample_id,
                    attempt,
                    started_at,
                    raw,
                    validate_misread_judgment_response(raw),
                )
            except Exception as exc:
                ledger.fail(sample_id, attempt, started_at, exc)
        _materialize(config, signature, ledger)
        return _summary(ledger.rows())
    finally:
        if owns_client:
            await client.close()
        ledger.close()


def validate_misread_judgment_response(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise MisreadJudgmentValidationError("judge response must be a string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MisreadJudgmentValidationError("judge response must be exact JSON") from exc
    if not isinstance(value, dict) or set(value) != {"decision", "confidence", "rationale"}:
        raise MisreadJudgmentValidationError(
            "judge response must contain exactly decision, confidence, rationale"
        )
    decision = value["decision"]
    confidence = value["confidence"]
    rationale = value["rationale"]
    if decision not in DECISIONS:
        raise MisreadJudgmentValidationError("judge decision is invalid")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise MisreadJudgmentValidationError("judge confidence must be in [0,1]")
    if not isinstance(rationale, str) or rationale != rationale.strip() or "\n" in rationale:
        raise MisreadJudgmentValidationError("judge rationale must be one short line")
    if (
        len(_SENTENCE_END.findall(rationale)) != 1
        or rationale[-1] not in ".!?"
        or len(rationale.split()) > 30
    ):
        raise MisreadJudgmentValidationError("judge rationale must be one short sentence")
    return {"decision": decision, "confidence": float(confidence), "rationale": rationale}


def verify_misread_judgment_artifacts(
    config: MisreadJudgeConfig, *, require_complete: bool
) -> dict[str, Any]:
    tasks = build_tasks(config)
    expected_ids = {task.sample_id for task in tasks}
    root = config.output_root
    records = _read_jsonl(root / "judgments.jsonl")
    queue = _read_jsonl(root / "human_review_queue.jsonl")
    failures = _read_jsonl(root / "failures.jsonl")
    if len({row.get("sample_id") for row in records}) != len(records):
        raise ValueError("Judgment export contains duplicate sample_id values")
    record_ids = {str(row.get("sample_id")) for row in records}
    if require_complete and (record_ids != expected_ids or failures):
        raise ValueError("Completed Misread judgment export must cover all tasks with no failures")
    expected_queue: set[str] = set()
    for row in records:
        validate_misread_judgment_response(
            _canonical_json(
                {
                    "decision": row.get("decision"),
                    "confidence": row.get("confidence"),
                    "rationale": row.get("rationale"),
                }
            )
        )
        if row["decision"] == "UNCERTAIN" or row["confidence"] < config.confidence_threshold:
            expected_queue.add(row["sample_id"])
    queue_ids = {str(row.get("sample_id")) for row in queue}
    if len(queue_ids) != len(queue) or queue_ids != expected_queue:
        raise ValueError(
            "Human review queue does not exactly match uncertain or low-confidence records"
        )
    for row in queue:
        if row.get("confidence_threshold") != config.confidence_threshold:
            raise ValueError("Human review queue threshold mismatch")
    provenance = _read_json(root / "provenance.json")
    if (
        provenance.get("schema_name") != PROVENANCE_SCHEMA
        or provenance.get("run_id") != config.run_id
        or provenance.get("signature") != _signature(config, tasks)
    ):
        raise ValueError("Judge provenance signature mismatch")
    if (
        provenance.get("threshold_interpretation")
        != "round-one provisional operational threshold; not a paper-validated threshold"
    ):
        raise ValueError("Judge provenance threshold interpretation mismatch")
    for artifact in provenance.get("artifacts", {}).values():
        path = root / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"Judge artifact hash mismatch: {path}")
    return {
        "count": len(records),
        "queue_count": len(queue),
        "failed": len(failures),
        "status": "passed",
    }


def import_human_decisions(config: MisreadJudgeConfig, path: str | Path) -> None:
    queue = _read_jsonl(config.output_root / "human_review_queue.jsonl")
    queued = {row["sample_id"] for row in queue}
    decisions = _read_jsonl(path)
    provided: dict[str, str] = {}
    for row in decisions:
        if set(row) != {"sample_id", "final_decision"}:
            raise ValueError("Human decisions must contain exactly sample_id and final_decision")
        sample_id = _required_text(row, "sample_id")
        final = row.get("final_decision")
        if final not in FINAL_DECISIONS or sample_id in provided:
            raise ValueError("Human decisions contain invalid or duplicate rows")
        provided[sample_id] = final
    if set(provided) != queued:
        raise ValueError("Human decisions must exactly cover the review queue")
    _atomic_jsonl(
        config.output_root / "human_decisions.jsonl",
        [
            {"sample_id": sample_id, "final_decision": provided[sample_id]}
            for sample_id in sorted(provided)
        ],
    )


def export_final_labels(config: MisreadJudgeConfig) -> list[dict[str, Any]]:
    records = _read_jsonl(config.output_root / "judgments.jsonl")
    queue = _read_jsonl(config.output_root / "human_review_queue.jsonl")
    queued = {row["sample_id"] for row in queue}
    if len(queued) != len(queue):
        raise ValueError("Human review queue contains duplicate sample_id values")
    human_path = config.output_root / "human_decisions.jsonl"
    human_rows = _read_jsonl(human_path) if human_path.is_file() else []
    human = {row.get("sample_id"): row.get("final_decision") for row in human_rows}
    if (
        len(human) != len(human_rows)
        or set(human) != queued
        or any(value not in FINAL_DECISIONS for value in human.values())
    ):
        raise ValueError(
            "Human decisions must exactly cover the review queue with binary decisions"
        )
    output: list[dict[str, Any]] = []
    for row in records:
        decision = human.get(row["sample_id"], row["decision"])
        if decision not in FINAL_DECISIONS:
            raise ValueError("Every UNCERTAIN judgment requires a human final decision")
        output.append(
            {
                "sample_id": row["sample_id"],
                "final_decision": decision,
                "binary_label": int(decision == "MISREAD"),
            }
        )
    expected_ids = {task.sample_id for task in build_tasks(config)}
    if {row["sample_id"] for row in output} != expected_ids:
        raise ValueError("Final binary labels must cover every configured sample")
    _atomic_jsonl(config.output_root / "final_binary_labels.jsonl", output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run blinded Misread judgment.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/judge/misread_judgment_v2.yaml")
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--import-human", type=Path)
    parser.add_argument("--export-final", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.import_human:
        import_human_decisions(config, args.import_human)
    if args.export_final:
        print(_canonical_json({"count": len(export_final_labels(config))}))
    elif args.verify:
        print(
            _canonical_json(
                verify_misread_judgment_artifacts(config, require_complete=True)
            )
        )
    else:
        print(
            _canonical_json(
                asyncio.run(
                    run_misread_judgment(
                        config=config, retry_failed=args.retry_failed
                    )
                )
            )
        )
    return 0


def _materialize(
    config: MisreadJudgeConfig,
    signature: dict[str, Any],
    ledger: MisreadJudgeLedger,
) -> None:
    rows = [dict(row) for row in ledger.rows()]
    judgments = []
    failures = []
    for row in rows:
        if row["status"] == "completed":
            result = json.loads(row["result_json"])
            judgments.append(
                {
                    "sample_id": row["sample_id"],
                    "input_hash": row["input_hash"],
                    "request_hash": row["request_hash"],
                    "raw_response": row["raw_response"],
                    **result,
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
    queue = [
        {
            "sample_id": row["sample_id"],
            "decision": row["decision"],
            "confidence": row["confidence"],
            "rationale": row["rationale"],
            "confidence_threshold": config.confidence_threshold,
        }
        for row in judgments
        if row["decision"] == "UNCERTAIN" or row["confidence"] < config.confidence_threshold
    ]
    payloads = {
        "judgments.jsonl": _jsonl(judgments),
        "human_review_queue.jsonl": _jsonl(queue),
        "failures.jsonl": _jsonl(failures),
        "attempts.jsonl": _jsonl([dict(row) for row in ledger.attempt_rows()]),
        "summary.json": (json.dumps(_summary(rows), sort_keys=True, indent=2) + "\n").encode(),
    }
    for name, content in payloads.items():
        _atomic_bytes(config.output_root / name, content)
    provenance = {
        "schema_name": PROVENANCE_SCHEMA,
        "run_id": config.run_id,
        "signature": signature,
        "confidence_threshold": config.confidence_threshold,
        "threshold_interpretation": (
            "round-one provisional operational threshold; not a paper-validated threshold"
        ),
        "artifacts": {
            name: {"path": name, "sha256": _sha256(config.output_root / name)} for name in payloads
        },
    }
    _atomic_bytes(
        config.output_root / "provenance.json",
        (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode(),
    )


def _signature(
    config: MisreadJudgeConfig, tasks: Sequence[MisreadJudgeTask]
) -> dict[str, Any]:
    return {
        "schema_name": SIGNATURE_SCHEMA,
        "run_id": config.run_id,
        "config_sha256": _hash_text(_canonical_json(config.model_dump(mode="json"))),
        "judge_model": config.judge_model,
        "subject_model_key": config.subject_model_key,
        "protocol": config.protocol,
        "split": config.split,
        "temperature": config.temperature,
        "confidence_threshold": config.confidence_threshold,
        "prompt_sha256": _hash_text(MISREAD_JUDGMENT_PROMPT),
        "gt_description_manifest_sha256": _sha256(
            config.gt_description_manifest_path
        ),
        "diagnostic_affect_description_manifest_sha256": _sha256(
            config.diagnostic_affect_description_manifest_path
        ),
        "task_count": len(tasks),
    }


def _summary(rows: Sequence[Any]) -> dict[str, int]:
    counts = Counter(row["status"] for row in rows)
    return {
        "total": len(rows),
        "completed": counts["completed"],
        "failed": counts["failed"],
        "pending": counts["pending"],
        "running": counts["running"],
    }


def _validate_blind_request(encoded: str, sample_id: str) -> None:
    lowered = encoded.lower()
    forbidden = (
        sample_id.lower(),
        "archetype",
        "trigger",
        "dialogue",
        "surface_emotion",
        "sample_type",
        "source_archive",
        "model_name",
    )
    if any(value in lowered for value in forbidden) or re.search(r"\\b(?:vt|va)\\b", lowered):
        raise ValueError("Judge request is not blinded")


def _index_rows(rows: list[dict[str, Any]], required_field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _required_text(row, "sample_id")
        _required_text(row, required_field)
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty {key}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    result = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in result):
        raise ValueError(f"{source} must contain JSON objects")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode()


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _atomic_bytes(path, _jsonl(rows))


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
