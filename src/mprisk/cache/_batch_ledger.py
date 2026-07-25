"""SQLite-backed ledger for resumable prefill batch extraction."""

from __future__ import annotations

import json
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mprisk.cache._batch_plan import BatchPlan, BatchTask
from mprisk.utils.api_runner import SqliteLedgerBase
from mprisk.utils.io import canonical_json as _canonical_json


class BatchLedger(SqliteLedgerBase):
    # BatchLedger gates prepare() on the signature first; only after that
    # match does it flip running->pending and (optionally) failed->pending.
    # The base class therefore must NOT do those resets in __init__.
    running_to_pending_sql = ""
    failed_to_pending_sql = None

    _SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks (
          task_id TEXT PRIMARY KEY, sample_id TEXT NOT NULL, model_key TEXT NOT NULL,
          protocol TEXT NOT NULL, prompt_set_key TEXT NOT NULL, prompt_id TEXT NOT NULL,
          condition TEXT NOT NULL, sample_type TEXT NOT NULL, use_in_main INTEGER NOT NULL,
          annotation_count INTEGER NOT NULL, split TEXT NOT NULL, source_dataset TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
          attempts INTEGER NOT NULL DEFAULT 0, error_type TEXT, error_message TEXT,
          traceback TEXT, layer_count INTEGER, hidden_dim INTEGER, token_count INTEGER,
          t0_token_index INTEGER, elapsed_seconds REAL, peak_gpu_memory_bytes INTEGER,
          checksum TEXT, entry_json TEXT
        );
    """

    #: SQL to clear failed payload alongside the status flip on retry.
    _CLEAR_FAILED_PAYLOAD_SQL = (
        "UPDATE tasks SET status='pending', error_type=NULL, error_message=NULL, "
        "traceback=NULL WHERE status='failed'"
    )

    def __init__(self, path: Path) -> None:
        super().__init__(path, schema_sql=self._SCHEMA_SQL)

    def prepare(self, plan: BatchPlan, *, retry_failed: bool) -> None:
        with self.connection:
            self.check_metadata_signature(
                key="signature",
                value=plan.signature,
                retry_failed=False,  # we run a payload-clearing variant below
                error_message="Existing batch ledger signature does not match this run",
            )
            self.connection.executemany(
                """INSERT OR IGNORE INTO tasks(
                   task_id,sample_id,model_key,protocol,prompt_set_key,prompt_id,
                   condition,sample_type,use_in_main,
                   annotation_count,split,source_dataset,status)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,'pending')""",
                [
                    (
                        task.task_id,
                        task.sample_id,
                        str(plan.signature["model_key"]),
                        str(plan.signature["protocol"]),
                        task.prompt_set_key,
                        task.prompt_id,
                        task.condition,
                        str(task.row.get("sample_type", "")),
                        int(bool(task.row.get("use_in_main"))),
                        int(task.row.get("annotation_count", 0)),
                        str(task.row.get("split", "")),
                        str(task.row.get("source_dataset", "")),
                    )
                    for task in plan.tasks
                ],
            )
            count = self.connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
            if count != len(plan.tasks):
                raise ValueError("Existing batch ledger task set does not match this run")
            self.connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")
            if retry_failed:
                self.connection.execute(self._CLEAR_FAILED_PAYLOAD_SQL)

    def pending_tasks(self, plan: BatchPlan) -> Iterator[BatchTask]:
        by_id = {task.task_id: task for task in plan.tasks}
        rows = self.connection.execute(
            "SELECT task_id FROM tasks WHERE status='pending' ORDER BY rowid"
        ).fetchall()
        for row in rows:
            task = by_id[row["task_id"]]
            with self.connection:
                changed = self.connection.execute(
                    """UPDATE tasks SET status='running', attempts=attempts+1
                       WHERE task_id=? AND status='pending'""",
                    (task.task_id,),
                ).rowcount
            if changed == 1:
                yield task

    def completed_tasks(self, plan: BatchPlan) -> Iterator[tuple[BatchTask, dict[str, Any]]]:
        by_id = {task.task_id: task for task in plan.tasks}
        rows = self.connection.execute(
            """SELECT task_id,entry_json FROM tasks WHERE status='completed'
               ORDER BY rowid"""
        ).fetchall()
        for row in rows:
            if row["entry_json"] is None:
                raise ValueError(f"Completed task has no ledger entry: {row['task_id']}")
            yield by_id[row["task_id"]], json.loads(row["entry_json"])

    def complete(
        self,
        task_id: str,
        entry: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE tasks SET status='completed',entry_json=?,layer_count=?,hidden_dim=?,
                   token_count=?,t0_token_index=?,elapsed_seconds=?,peak_gpu_memory_bytes=?,checksum=?,
                   error_type=NULL,error_message=NULL,traceback=NULL WHERE task_id=?""",
                (
                    _canonical_json(entry),
                    int(entry["layer_count"]),
                    int(entry["hidden_dim"]),
                    int(entry["token_count"]),
                    int(entry["t0_token_index"]),
                    provenance.get("elapsed_seconds"),
                    provenance.get("peak_gpu_memory_bytes"),
                    str(entry["checksum"]),
                    task_id,
                ),
            )

    def fail(self, task_id: str, error: Exception) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE tasks SET status='failed', error_type=?, error_message=?, traceback=?
                   WHERE task_id=?""",
                (type(error).__name__, str(error), traceback.format_exc(), task_id),
            )

    def completed_entries(self, prompt_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT entry_json FROM tasks WHERE prompt_id=? AND status='completed'
               ORDER BY rowid""",
            (prompt_id,),
        ).fetchall()
        return [json.loads(row["entry_json"]) for row in rows]

    def completed_entries_all(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT entry_json FROM tasks WHERE status='completed' ORDER BY rowid"""
        ).fetchall()
        return [json.loads(row["entry_json"]) for row in rows]

    def failures(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """SELECT task_id,sample_id,model_key,protocol,prompt_set_key,prompt_id,
               condition,sample_type,use_in_main,
               annotation_count,split,source_dataset,attempts,error_type,error_message,traceback
               FROM tasks WHERE status='failed' ORDER BY rowid"""
            ).fetchall()
        ]

    def summary(self) -> dict[str, Any]:
        counts = self.status_counts()
        return {
            "total": sum(counts.values()),
            **{key: counts.get(key, 0) for key in ("pending", "running", "completed", "failed")},
        }

    def close(self) -> None:
        self.connection.close()
