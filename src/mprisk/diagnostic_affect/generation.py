"""Resumable model-independent Diagnostic Affect Description generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mprisk.assets.registry import index_assets, load_model_assets
from mprisk.models.base_wrapper import GenerationRequest, GenerationResult
from mprisk.models.wrapper_registry import get_wrapper

CANONICAL_DIAGNOSTIC_AFFECT_PROMPT = (
    "Based on the complete input, describe the person's overall emotional state in one concise "
    "sentence. Do not address the person, give advice, or explain your reasoning."
)
CONFIG_SCHEMA = "mprisk_diagnostic_affect_description_config_v2"
OUTPUT_SCHEMA = "mprisk_diagnostic_affect_description_v2"
PROVENANCE_SCHEMA = "mprisk_diagnostic_affect_description_provenance_v2"
SIGNATURE_SCHEMA = "mprisk_diagnostic_affect_description_signature_v2"
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_SUPPORTED_PROTOCOLS = frozenset({"VT", "VA"})
_SUPPORTED_CONDITIONS = frozenset({"M12"})


@dataclass(frozen=True)
class DiagnosticAffectDescriptionTask:
    task_id: str
    request: GenerationRequest
    input_sha256: str
    media_sha256: str
    prompt_sha256: str


@dataclass(frozen=True)
class DiagnosticAffectDescriptionPlan:
    tasks: list[DiagnosticAffectDescriptionTask]
    signature: dict[str, Any]
    counts: dict[str, int]


def build_diagnostic_affect_description_plan(
    *,
    schema_name: str,
    run_id: str,
    manifest_path: Path,
    subject_model_key: str,
    model_family: str,
    model_path: Path,
    protocol: str,
    condition: str,
    dataset: str,
    split: str,
    max_new_tokens: int,
    video_fps: float = 1.0,
    asset_config_sha256: str = "test-asset-config",
    config_sha256: str = "test-config",
    selected_sample_ids: Iterable[str] | None = None,
) -> DiagnosticAffectDescriptionPlan:
    """Build a strict sample-level plan from one explicit dataset/protocol/split."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    protocol = protocol.upper()
    condition = condition.upper()
    if protocol not in _SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported Diagnostic Affect Description protocol: {protocol!r}")
    if condition not in _SUPPORTED_CONDITIONS:
        raise ValueError(f"Unsupported Diagnostic Affect Description condition: {condition!r}")
    if schema_name != CONFIG_SCHEMA:
        raise ValueError(f"Unsupported Diagnostic Affect Description schema: {schema_name!r}")
    for field_name, value in (
        ("run_id", run_id),
        ("subject_model_key", subject_model_key),
        ("model_family", model_family),
        ("dataset", dataset),
        ("split", split),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    selected = None if selected_sample_ids is None else set(selected_sample_ids)
    source_rows = _read_jsonl(manifest_path)
    rows = [
        row
        for row in source_rows
        if str(row.get("source_dataset", "")) == dataset
        and str(row.get("split", "")) == split
        and str(row.get("protocol", "")).upper() == protocol
    ]
    if not rows:
        raise ValueError(
            "Manifest contains no rows for "
            f"dataset={dataset!r}, split={split!r}, protocol={protocol!r}"
        )
    tasks: list[DiagnosticAffectDescriptionTask] = []
    for row in rows:
        sample_id = _required_string(row, "sample_id")
        if selected is not None and sample_id not in selected:
            continue
        media_paths = _required_media_paths(row, protocol=protocol)
        media_hashes = {name: _sha256(path) for name, path in media_paths.items()}
        vision_path = media_paths["vision"]
        content: list[dict[str, Any]] = [
            {"type": "video", "video": str(vision_path), "fps": video_fps}
        ]
        if protocol == "VT":
            dialogue = _required_string(row, "text_content")
            content.append(
                {
                    "type": "text",
                    "text": f"{dialogue}\n\n{CANONICAL_DIAGNOSTIC_AFFECT_PROMPT}",
                }
            )
            use_audio_in_video = False
        else:
            content.append({"type": "text", "text": CANONICAL_DIAGNOSTIC_AFFECT_PROMPT})
            use_audio_in_video = True
        request = GenerationRequest(
            sample_id=sample_id,
            model_key=subject_model_key,
            protocol=protocol.lower(),
            condition=condition,
            messages=({"role": "user", "content": content},),
            media_paths={name: str(path) for name, path in media_paths.items()},
            use_audio_in_video=use_audio_in_video,
            generation_kwargs={
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
            },
        )
        input_payload = {
            "sample_id": sample_id,
            "subject_model_key": subject_model_key,
            "protocol": protocol,
            "condition": condition,
            "dataset": dataset,
            "split": split,
            "messages": list(request.messages),
            "media_sha256": media_hashes,
        }
        input_sha256 = _hash_text(_canonical_json(input_payload))
        task_id = _hash_text(
            _canonical_json({"sample_id": sample_id, "input_sha256": input_sha256})
        )
        tasks.append(
            DiagnosticAffectDescriptionTask(
                task_id=task_id,
                request=request,
                input_sha256=input_sha256,
                media_sha256=_hash_text(_canonical_json(media_hashes)),
                prompt_sha256=_hash_text(CANONICAL_DIAGNOSTIC_AFFECT_PROMPT),
            )
        )
    if selected is not None and {task.request.sample_id for task in tasks} != selected:
        raise ValueError(
            "One or more selected sample IDs are absent from the selected manifest rows"
        )
    if len({task.request.sample_id for task in tasks}) != len(tasks):
        raise ValueError("Selected manifest rows contain duplicate sample_id values")
    counts = {protocol: len(tasks)}
    model_path = model_path.expanduser().resolve()
    signature = {
        "schema_name": SIGNATURE_SCHEMA,
        "run_id": run_id,
        "manifest_sha256": _sha256(manifest_path),
        "asset_config_sha256": asset_config_sha256,
        "subject_model_key": subject_model_key,
        "model_family": model_family,
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "model_weight_map_sha256": _model_weight_map_sha256(model_path),
        "protocol": protocol,
        "condition": condition,
        "dataset": dataset,
        "split": split,
        "prompt_sha256": _hash_text(CANONICAL_DIAGNOSTIC_AFFECT_PROMPT),
        "config_sha256": config_sha256,
        "max_new_tokens": max_new_tokens,
        "video_fps": video_fps,
        "generation_policy": {"do_sample": False, "num_beams": 1},
        "task_count": len(tasks),
        "counts": counts,
    }
    return DiagnosticAffectDescriptionPlan(tasks=tasks, signature=signature, counts=counts)


def validate_diagnostic_affect_description(result: GenerationResult) -> None:
    """Reject invalid model output; never rewrite, truncate, or replace it."""
    text = result.text
    if result.request.condition != "M12":
        raise ValueError("Diagnostic descriptions require the M12 condition")
    if not text or text != text.strip() or "\n" in text:
        raise ValueError("Generated description must be non-empty")
    endings = _SENTENCE_END_RE.findall(text)
    if len(endings) != 1 or text[-1] not in ".!?":
        raise ValueError("Generated description must contain exactly one sentence")


class DiagnosticAffectDescriptionLedger:
    """SQLite resume state with an immutable per-run signature."""

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
              task_id TEXT PRIMARY KEY, sample_id TEXT NOT NULL UNIQUE, protocol TEXT NOT NULL,
              input_sha256 TEXT NOT NULL, media_sha256 TEXT NOT NULL, prompt_sha256 TEXT NOT NULL,
              request_json TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
              attempts INTEGER NOT NULL DEFAULT 0, result_json TEXT, provenance_json TEXT,
              error_type TEXT, error_message TEXT, traceback TEXT, elapsed_seconds REAL
            );
            CREATE TABLE IF NOT EXISTS attempts (
              task_id TEXT NOT NULL, attempt INTEGER NOT NULL, started_at TEXT NOT NULL,
              finished_at TEXT, outcome TEXT NOT NULL, result_json TEXT,
              error_type TEXT, error_message TEXT, traceback TEXT,
              PRIMARY KEY(task_id,attempt)
            );
            """
        )

    def prepare(self, signature: dict[str, Any], *, retry_failed: bool = False) -> None:
        encoded = _canonical_json(signature)
        with self.connection:
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key='signature'"
            ).fetchone()
            if row is not None and row["value"] != encoded:
                raise ValueError("Existing description ledger signature does not match this run")
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('signature',?)", (encoded,)
            )
            self.connection.execute(
                "UPDATE attempts SET outcome='interrupted',finished_at=? WHERE outcome='running'",
                (_now(),),
            )
            self.connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")
            if retry_failed:
                self.connection.execute(
                    "UPDATE tasks SET status='pending',error_type=NULL,error_message=NULL,"
                    "traceback=NULL WHERE status='failed'"
                )

    def add_tasks(self, tasks: Sequence[DiagnosticAffectDescriptionTask]) -> None:
        with self.connection:
            self.connection.executemany(
                """INSERT OR IGNORE INTO tasks(
                task_id,sample_id,protocol,input_sha256,media_sha256,prompt_sha256,request_json,status)
                VALUES(?,?,?,?,?,?,?,'pending')""",
                [
                    (
                        task.task_id,
                        task.request.sample_id,
                        task.request.protocol,
                        task.input_sha256,
                        task.media_sha256,
                        task.prompt_sha256,
                        _canonical_json(_request_payload(task.request)),
                    )
                    for task in tasks
                ],
            )
            count = self.connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
            if count != len(tasks):
                raise ValueError("Existing description ledger task set does not match this run")

    def validate_completed(self, tasks: Sequence[DiagnosticAffectDescriptionTask]) -> None:
        """Verify resumed completed rows against the immutable task input before reuse."""
        by_id = {task.task_id: task for task in tasks}
        rows = self.connection.execute(
            """SELECT task_id,input_sha256,media_sha256,prompt_sha256,request_json,result_json
            FROM tasks WHERE status='completed'"""
        ).fetchall()
        for row in rows:
            task = by_id.get(row["task_id"])
            if task is None:
                raise ValueError(f"Completed task is absent from this plan: {row['task_id']}")
            if row["input_sha256"] != task.input_sha256:
                raise ValueError(f"Completed task input hash mismatch: {task.request.sample_id}")
            if row["media_sha256"] != task.media_sha256:
                raise ValueError(f"Completed task media hash mismatch: {task.request.sample_id}")
            if row["prompt_sha256"] != task.prompt_sha256:
                raise ValueError(f"Completed task prompt hash mismatch: {task.request.sample_id}")
            if json.loads(row["request_json"]) != _request_payload(task.request):
                raise ValueError(f"Completed task request mismatch: {task.request.sample_id}")
            result = json.loads(row["result_json"])
            validate_diagnostic_affect_description(
                GenerationResult(
                    request=task.request,
                    text=str(result["text"]),
                    token_ids=result["token_ids"],
                    eos_token_ids=result["eos_token_ids"],
                    finish_reason=str(result["finish_reason"]),
                    input_token_count=int(result["input_token_count"]),
                )
            )

    def pending_tasks(
        self, tasks: Sequence[DiagnosticAffectDescriptionTask]
    ) -> Iterable[tuple[DiagnosticAffectDescriptionTask, int]]:
        by_id = {task.task_id: task for task in tasks}
        for row in self.connection.execute(
            "SELECT task_id,attempts FROM tasks WHERE status='pending' ORDER BY rowid"
        ):
            task = by_id[row["task_id"]]
            attempt = int(row["attempts"]) + 1
            with self.connection:
                changed = self.connection.execute(
                    "UPDATE tasks SET status='running',attempts=attempts+1 "
                    "WHERE task_id=? AND status='pending'",
                    (task.task_id,),
                ).rowcount
                if changed == 1:
                    self.connection.execute(
                        "INSERT INTO attempts(task_id,attempt,started_at,outcome) VALUES(?,?,?,?)",
                        (task.task_id, attempt, _now(), "running"),
                    )
            if changed == 1:
                yield task, attempt

    def complete(
        self,
        task_id: str,
        attempt: int,
        result: GenerationResult,
        provenance: dict[str, Any],
    ) -> None:
        validate_diagnostic_affect_description(result)
        with self.connection:
            self.connection.execute(
                """UPDATE tasks SET status='completed',result_json=?,provenance_json=?,
                elapsed_seconds=?,error_type=NULL,error_message=NULL,traceback=NULL
                WHERE task_id=?""",
                (
                    _canonical_json(_result_payload(result)),
                    _canonical_json(provenance),
                    provenance.get("elapsed_seconds"),
                    task_id,
                ),
            )
            self.connection.execute(
                "UPDATE attempts SET finished_at=?,outcome='completed',result_json=? "
                "WHERE task_id=? AND attempt=?",
                (_now(), _canonical_json(_result_payload(result)), task_id, attempt),
            )

    def fail(self, task_id: str, attempt: int, error: Exception) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE tasks SET status='failed',error_type=?,error_message=?,traceback=? "
                "WHERE task_id=?",
                (type(error).__name__, str(error), traceback.format_exc(), task_id),
            )
            self.connection.execute(
                "UPDATE attempts SET finished_at=?,outcome='failed',error_type=?,error_message=?,"
                "traceback=? WHERE task_id=? AND attempt=?",
                (
                    _now(),
                    type(error).__name__,
                    str(error),
                    traceback.format_exc(),
                    task_id,
                    attempt,
                ),
            )

    def completed_records(self) -> list[dict[str, Any]]:
        signature_row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='signature'"
        ).fetchone()
        if signature_row is None:
            raise ValueError("Description ledger has no immutable signature")
        signature = json.loads(signature_row["value"])
        records = []
        for row in self.connection.execute(
            """SELECT sample_id,protocol,input_sha256,media_sha256,prompt_sha256,result_json,
            provenance_json,request_json
            FROM tasks WHERE status='completed' ORDER BY sample_id"""
        ):
            result = json.loads(row["result_json"])
            request = json.loads(row["request_json"])
            records.append(
                {
                    "schema_name": OUTPUT_SCHEMA,
                    "run_id": signature["run_id"],
                    "sample_id": row["sample_id"],
                    "subject_model_key": request["model_key"],
                    "protocol": row["protocol"].upper(),
                    "condition": request["condition"],
                    "dataset": signature["dataset"],
                    "split": signature["split"],
                    "DIAGNOSTIC_AFFECT_DESCRIPTION": result["text"],
                    "token_ids": result["token_ids"],
                    "eos_token_ids": result["eos_token_ids"],
                    "finish_reason": result["finish_reason"],
                    "input_token_count": result["input_token_count"],
                    "input_sha256": row["input_sha256"],
                    "media_sha256": row["media_sha256"],
                    "prompt_sha256": row["prompt_sha256"],
                    "provenance": json.loads(row["provenance_json"]),
                }
            )
        return records

    def failures(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """SELECT task_id,sample_id,protocol,attempts,error_type,error_message,traceback
                FROM tasks WHERE status='failed' ORDER BY rowid"""
            )
        ]

    def attempt_records(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM attempts ORDER BY task_id,attempt"
            )
        ]

    def summary(self) -> dict[str, int]:
        counts = {
            row["status"]: row["n"]
            for row in self.connection.execute(
                "SELECT status,COUNT(*) AS n FROM tasks GROUP BY status"
            )
        }
        return {
            "total": sum(counts.values()),
            **{key: counts.get(key, 0) for key in ("pending", "running", "completed", "failed")},
        }

    def close(self) -> None:
        self.connection.close()


def export_diagnostic_affect_descriptions(
    records: Sequence[dict[str, Any]], destination: Path
) -> None:
    """Write a deterministic manifest only after complete records have been validated."""
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Description manifest contains duplicate sample_id values")
    for record in records:
        request = GenerationRequest(
            sample_id=str(record["sample_id"]),
            model_key=str(record["subject_model_key"]),
            protocol=str(record["protocol"]),
            condition=str(record["condition"]),
            messages=(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CANONICAL_DIAGNOSTIC_AFFECT_PROMPT}
                    ],
                },
            ),
            media_paths={},
            use_audio_in_video=str(record["protocol"]).upper() == "VA",
            generation_kwargs={"do_sample": False, "num_beams": 1, "max_new_tokens": 1},
        )
        validate_diagnostic_affect_description(
            GenerationResult(
                request=request,
                text=str(record["DIAGNOSTIC_AFFECT_DESCRIPTION"]),
                token_ids=record["token_ids"],
                eos_token_ids=record["eos_token_ids"],
                finish_reason=str(record["finish_reason"]),
                input_token_count=int(record["input_token_count"]),
            )
        )
    _atomic_text(destination, "".join(_canonical_json(record) + "\n" for record in records))


def generate_diagnostic_affect_descriptions(
    plan: DiagnosticAffectDescriptionPlan,
    *,
    output_root: Path,
    subject_model_key: str,
    model_family: str,
    model_path: Path,
    device: str,
    dtype: str,
    attn_implementation: str,
    retry_failed: bool = False,
    wrapper_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one model process serially; resume only an identical immutable plan."""
    output_root = output_root.expanduser().resolve()
    ledger = DiagnosticAffectDescriptionLedger(output_root / "batch_state.sqlite3")
    ledger.prepare(plan.signature, retry_failed=retry_failed)
    ledger.add_tasks(plan.tasks)
    ledger.validate_completed(plan.tasks)
    if plan.signature.get("subject_model_key") != subject_model_key:
        raise ValueError("subject_model_key does not match the immutable plan")
    if plan.signature.get("model_family") != model_family:
        raise ValueError("model_family does not match the immutable plan")
    factory = wrapper_factory or get_wrapper(model_family)
    wrapper = factory(
        model_key=subject_model_key,
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    try:
        wrapper.load()
        for task, attempt in ledger.pending_tasks(plan.tasks):
            started = time.perf_counter()
            try:
                result = wrapper.generate_conditioned(task.request)
                provenance = {
                    "model_path": str(model_path.expanduser().resolve()),
                    "model_config_sha256": plan.signature["model_config_sha256"],
                    "model_weight_map_sha256": plan.signature["model_weight_map_sha256"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "generation": dict(result.provenance),
                }
                ledger.complete(task.task_id, attempt, result, provenance)
            except Exception as error:
                ledger.fail(task.task_id, attempt, error)
                _materialize(ledger, output_root, plan.signature)
    finally:
        wrapper.close()
        _materialize(ledger, output_root, plan.signature)
        summary = ledger.summary()
        ledger.close()
    return summary


def verify_diagnostic_affect_descriptions(
    *,
    manifest_path: Path,
    output_root: Path,
    subject_model_key: str,
    run_id: str,
    protocol: str,
    condition: str,
    dataset: str,
    split: str,
    strict_full: bool = True,
) -> dict[str, Any]:
    records = _read_jsonl(output_root / "manifest.jsonl")
    protocol = protocol.upper()
    condition = condition.upper()
    selected_rows = [
        row
        for row in _read_jsonl(manifest_path)
        if str(row.get("source_dataset", "")) == dataset
        and str(row.get("split", "")) == split
        and str(row.get("protocol", "")).upper() == protocol
    ]
    expected = {str(row["sample_id"]): protocol for row in selected_rows}
    observed = {str(row.get("sample_id")): str(row.get("protocol")).upper() for row in records}
    if len(observed) != len(records):
        raise ValueError("Description manifest contains duplicate sample IDs")
    if strict_full and set(observed) != set(expected):
        raise ValueError("Description manifest does not match selected manifest sample IDs")
    if not strict_full and not set(observed).issubset(expected):
        raise ValueError("Description smoke manifest contains unknown sample IDs")
    expected_fields = {
        "schema_name", "run_id", "sample_id", "subject_model_key", "protocol", "condition",
        "dataset", "split",
        "DIAGNOSTIC_AFFECT_DESCRIPTION", "token_ids", "eos_token_ids", "finish_reason",
        "input_token_count", "input_sha256", "media_sha256", "prompt_sha256", "provenance",
    }
    for record in records:
        if set(record) != expected_fields:
            raise ValueError("Description manifest fields are not strict")
        if (
            record.get("schema_name") != OUTPUT_SCHEMA
            or record.get("run_id") != run_id
            or record.get("subject_model_key") != subject_model_key
            or record.get("protocol") != protocol
            or record.get("condition") != condition
            or record.get("dataset") != dataset
            or record.get("split") != split
        ):
            raise ValueError("Description manifest schema or condition mismatch")
        if observed[str(record["sample_id"])] != expected[str(record["sample_id"])]:
            raise ValueError("Description protocol does not match frozen eligibility input")
        request = GenerationRequest(
            sample_id=str(record["sample_id"]),
            model_key=subject_model_key,
            protocol=str(record["protocol"]),
            condition=condition,
            messages=(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CANONICAL_DIAGNOSTIC_AFFECT_PROMPT}
                    ],
                },
            ),
            media_paths={},
            use_audio_in_video=str(record["protocol"]).upper() == "VA",
            generation_kwargs={"do_sample": False, "num_beams": 1, "max_new_tokens": 1},
        )
        validate_diagnostic_affect_description(
            GenerationResult(
                request=request,
                text=str(record["DIAGNOSTIC_AFFECT_DESCRIPTION"]),
                token_ids=record["token_ids"],
                eos_token_ids=record["eos_token_ids"],
                finish_reason=str(record["finish_reason"]),
                input_token_count=int(record["input_token_count"]),
            )
        )
    summary = _read_json(output_root / "summary.json")
    if (
        summary.get("failed") != 0
        or summary.get("pending") != 0
        or summary.get("running") != 0
        or summary.get("completed") != len(records)
    ):
        raise ValueError("Description ledger summary is incomplete or contains failures")
    counts = {protocol: len(records)}
    provenance = _read_json(output_root / "provenance.json")
    if (
        provenance.get("schema_name") != PROVENANCE_SCHEMA
        or provenance.get("run_id") != run_id
    ):
        raise ValueError("Description provenance schema mismatch")
    if provenance.get("canonical_prompt") != CANONICAL_DIAGNOSTIC_AFFECT_PROMPT:
        raise ValueError("Description provenance prompt mismatch")
    signature = provenance.get("signature")
    expected_identity = {
        "schema_name": SIGNATURE_SCHEMA,
        "run_id": run_id,
        "subject_model_key": subject_model_key,
        "protocol": protocol,
        "condition": condition,
        "dataset": dataset,
        "split": split,
    }
    if not isinstance(signature, dict) or any(
        signature.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("Description provenance identity mismatch")
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Description provenance artifacts are missing")
    for name in ("manifest", "failures", "attempts", "summary"):
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError(f"Description artifact metadata is invalid: {name}")
        if artifact.get("sha256") != _sha256(output_root / artifact["path"]):
            raise ValueError(f"Description artifact hash mismatch: {name}")
    return {
        "status": "passed",
        "count": len(records),
        "counts": counts,
        "manifest_sha256": _sha256(output_root / "manifest.jsonl"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate strict model-independent Diagnostic Affect Descriptions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/diagnostic_affect_description.yaml"),
    )
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _read_config(args.config)
    manifest_path = args.manifest_path or Path(config["manifest_path"])
    output_root = args.output_root or Path(config["output_root"])
    asset_config = Path(config["asset_config"])
    subject_model_key = str(config["subject_model_key"])
    assets = index_assets(load_model_assets(asset_config, require_local_paths=False))
    if subject_model_key not in assets:
        raise ValueError(f"Unknown subject_model_key: {subject_model_key!r}")
    asset = assets[subject_model_key]
    model_path = Path(config["model_path"]).expanduser().resolve()
    if model_path != asset.local_model_path.expanduser().resolve():
        raise ValueError("model_path does not match subject_model_key in asset_config")
    protocol = str(config["protocol"]).upper()
    condition = str(config["condition"]).upper()
    dataset = str(config["dataset"])
    split = str(config["split"])
    if protocol.lower() not in asset.protocols:
        raise ValueError(f"Subject model {subject_model_key!r} does not support {protocol!r}")
    max_new_tokens = int(config["max_new_tokens"])
    video_fps = float(config["video_fps"])
    device = args.device or str(config["device"])
    attn_implementation = str(config["attn_implementation"])
    if args.sample_id and not args.smoke:
        raise ValueError("--sample-id is only valid with --smoke")
    selected_sample_ids = args.sample_id
    if args.smoke and not selected_sample_ids:
        selected_sample_ids = _select_smoke_sample_ids(
            manifest_path,
            dataset=dataset,
            split=split,
            protocol=protocol,
        )
    plan = build_diagnostic_affect_description_plan(
        schema_name=str(config["schema_name"]),
        run_id=str(config["run_id"]),
        manifest_path=manifest_path,
        subject_model_key=subject_model_key,
        model_family=asset.family,
        model_path=model_path,
        protocol=protocol,
        condition=condition,
        dataset=dataset,
        split=split,
        max_new_tokens=max_new_tokens,
        video_fps=video_fps,
        asset_config_sha256=_sha256(asset_config),
        config_sha256=_sha256(args.config),
        selected_sample_ids=selected_sample_ids,
    )
    summary = generate_diagnostic_affect_descriptions(
        plan,
        output_root=output_root,
        subject_model_key=subject_model_key,
        model_family=asset.family,
        model_path=model_path,
        device=device,
        dtype=str(config["dtype"]),
        attn_implementation=attn_implementation,
        retry_failed=args.retry_failed,
    )
    if summary["failed"] or summary["pending"] or summary["running"]:
        raise RuntimeError(f"Generation did not complete cleanly: {summary}")
    verification = verify_diagnostic_affect_descriptions(
        manifest_path=manifest_path,
        output_root=output_root,
        subject_model_key=subject_model_key,
        run_id=str(config["run_id"]),
        protocol=protocol,
        condition=condition,
        dataset=dataset,
        split=split,
        strict_full=not args.smoke,
    )
    print(
        _canonical_json(
            {
                "summary": summary,
                "verification": verification,
                "output_root": str(output_root.resolve()),
            }
        )
    )
    return 0


def _materialize(
    ledger: DiagnosticAffectDescriptionLedger,
    output_root: Path,
    signature: dict[str, Any],
) -> None:
    records = ledger.completed_records()
    export_diagnostic_affect_descriptions(records, output_root / "manifest.jsonl")
    _atomic_text(
        output_root / "failures.jsonl",
        "".join(_canonical_json(row) + "\n" for row in ledger.failures()),
    )
    _atomic_text(
        output_root / "attempts.jsonl",
        "".join(_canonical_json(row) + "\n" for row in ledger.attempt_records()),
    )
    _atomic_json(output_root / "summary.json", ledger.summary())
    _atomic_json(
        output_root / "provenance.json",
        {
            "schema_name": PROVENANCE_SCHEMA,
            "run_id": signature["run_id"],
            "canonical_prompt": CANONICAL_DIAGNOSTIC_AFFECT_PROMPT,
            "signature": signature,
            "artifacts": {
                name: {"path": filename, "sha256": _sha256(output_root / filename)}
                for name, filename in {
                    "manifest": "manifest.jsonl",
                    "failures": "failures.jsonl",
                    "attempts": "attempts.jsonl",
                    "summary": "summary.json",
                }.items()
            },
        },
    )


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_name") != CONFIG_SCHEMA:
        raise ValueError(f"Unsupported diagnostic-description config: {path}")
    required = {
        "schema_name",
        "run_id",
        "asset_config",
        "manifest_path",
        "output_root",
        "subject_model_key",
        "model_path",
        "protocol",
        "condition",
        "dataset",
        "split",
        "device",
        "dtype",
        "max_new_tokens",
        "video_fps",
        "attn_implementation",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Diagnostic-description config is missing: {sorted(missing)}")
    if set(value) != required:
        raise ValueError("Diagnostic-description config contains unsupported fields")
    return value


def _request_payload(request: GenerationRequest) -> dict[str, Any]:
    return {
        "sample_id": request.sample_id,
        "model_key": request.model_key,
        "protocol": request.protocol,
        "condition": request.condition,
        "messages": list(request.messages),
        "media_paths": dict(request.media_paths),
        "use_audio_in_video": request.use_audio_in_video,
        "generation_kwargs": dict(request.generation_kwargs),
    }


def _result_payload(result: GenerationResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "token_ids": list(result.token_ids),
        "eos_token_ids": list(result.eos_token_ids),
        "finish_reason": result.finish_reason,
        "input_token_count": result.input_token_count,
    }


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest row has no non-empty {key}: {row.get('sample_id')}")
    return value


def _required_media_paths(row: dict[str, Any], *, protocol: str) -> dict[str, Path]:
    raw = row.get("media_paths")
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest row has no media_paths object: {row.get('sample_id')}")
    required_modalities = ("vision",) if protocol == "VT" else ("vision", "audio")
    paths: dict[str, Path] = {}
    for modality in required_modalities:
        value = raw.get(modality)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Manifest row has no {modality} media path: {row.get('sample_id')}"
            )
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model input media is missing: {path}")
        paths[modality] = path
    return paths


def _model_weight_map_sha256(model_path: Path) -> str:
    index_files = sorted(model_path.glob("*.index.json"))
    if index_files:
        entries = {path.name: _sha256(path) for path in index_files}
    else:
        weight_files = sorted(model_path.glob("*.safetensors")) + sorted(
            model_path.glob("*.bin")
        )
        if not weight_files:
            raise FileNotFoundError(f"No model weight files or index found in {model_path}")
        entries = {path.name: _sha256(path) for path in weight_files}
    return _hash_text(_canonical_json(entries))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} must be a JSON object")
            rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _select_smoke_sample_ids(
    manifest_path: Path,
    *,
    dataset: str,
    split: str,
    protocol: str,
) -> list[str]:
    selected: dict[str, str] = {}
    for row in _read_jsonl(manifest_path):
        if (
            str(row.get("source_dataset", "")) != dataset
            or str(row.get("split", "")) != split
            or str(row.get("protocol", "")).upper() != protocol.upper()
        ):
            continue
        sample_type = _required_string(row, "sample_type")
        if sample_type in {"Conflict", "Aligned"} and sample_type not in selected:
            selected[sample_type] = _required_string(row, "sample_id")
    if set(selected) != {"Conflict", "Aligned"}:
        raise ValueError("Smoke selection requires one Conflict and one Aligned sample")
    return [selected["Conflict"], selected["Aligned"]]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
