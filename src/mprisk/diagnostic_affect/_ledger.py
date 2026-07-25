"""Diagnostic Affect Description SQLite resume ledger."""

from __future__ import annotations

import json
import traceback
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from mprisk.models.base_wrapper import GenerationResult
from mprisk.utils.api_runner import SqliteLedgerBase
from mprisk.utils.io import (
    canonical_json as _canonical_json,
    now_iso as _now,
)

from mprisk.diagnostic_affect._plan import (
    OUTPUT_SCHEMA,
    DiagnosticAffectDescriptionTask,
    _request_payload,
    _result_payload,
)
from mprisk.diagnostic_affect._verifier import validate_diagnostic_affect_description


class DiagnosticAffectDescriptionLedger(SqliteLedgerBase):
    """SQLite resume state with an immutable per-run signature."""

    running_to_pending_sql = ""
    failed_to_pending_sql = None

    _SCHEMA_SQL = """
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

    #: When retry_failed is requested, also clear the per-failure payload so
    #: the next run sees a clean failure surface.
    _CLEAR_FAILED_PAYLOAD_SQL = (
        "UPDATE tasks SET status='pending',error_type=NULL,error_message=NULL,"
        "traceback=NULL WHERE status='failed'"
    )

    def __init__(self, path: Path) -> None:
        super().__init__(path, schema_sql=self._SCHEMA_SQL)

    def prepare(self, signature: dict[str, Any], *, retry_failed: bool = False) -> None:
        with self.connection:
            self.check_metadata_signature(
                key="signature",
                value=signature,
                retry_failed=retry_failed,
                error_message="Existing description ledger signature does not match this run",
                clear_failed_payload=retry_failed,
                clear_failed_payload_sql=self._CLEAR_FAILED_PAYLOAD_SQL,
            )
            self.connection.execute(
                "UPDATE attempts SET outcome='interrupted',finished_at=? WHERE outcome='running'",
                (_now(),),
            )
            self.connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")

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
        counts = self.status_counts()
        return {
            "total": sum(counts.values()),
            **{key: counts.get(key, 0) for key in ("pending", "running", "completed", "failed")},
        }

    def close(self) -> None:
        self.connection.close()
