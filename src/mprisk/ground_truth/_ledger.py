"""GT Description generation SQLite ledger."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from mprisk.data.generated_archive_freeze import _canonical_json
from mprisk.utils.api_runner import SqliteLedgerBase
from mprisk.utils.io import now_iso as _now

from mprisk.ground_truth._plan import GTDescriptionGenerationTask


class GTDescriptionGenerationLedger(SqliteLedgerBase):
    # GT Description generation gates a run on the in-row ledger_signature
    # columns rather than a ``metadata.signature`` row, so the base-class
    # signature helper is unused here.
    running_to_pending_sql = "update tasks set status='pending' where status='running'"
    failed_to_pending_sql = None  # stage rejects retry_failed (handled by caller)

    _SCHEMA_SQL = """
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

    def __init__(self, path: Path):
        super().__init__(path, schema_sql=self._SCHEMA_SQL, synchronous="")

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
            existing = self.connection.execute(
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
            self.connection.execute(
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
        self.connection.commit()
        expected_ids = {task.sample_id for task in tasks}
        actual_ids = {str(row[0]) for row in self.connection.execute("select sample_id from tasks")}
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
            for row in self.connection.execute(
                f"select sample_id from tasks where status in ({placeholders}) "
                "order by task_order",
                statuses,
            )
        ]

    def start(self, sample_id: str) -> int:
        row = self.connection.execute(
            "select attempts from tasks where sample_id=?", (sample_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown ledger sample_id: {sample_id}")
        attempt = int(row[0]) + 1
        self.connection.execute(
            """update tasks set status='running',attempts=?,updated_at=?,
               error_type=null,error_message=null where sample_id=?""",
            (attempt, _now(), sample_id),
        )
        self.connection.commit()
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
        self.connection.execute(
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
        self.connection.commit()

    def complete(self, sample_id: str, result: dict[str, Any]) -> None:
        self.connection.execute(
            "update tasks set status='completed',result_json=?,updated_at=? where sample_id=?",
            (_canonical_json(result), _now(), sample_id),
        )
        self.connection.commit()

    def fail(self, sample_id: str, exc: Exception) -> None:
        self.connection.execute(
            """update tasks set status='failed',error_type=?,error_message=?,updated_at=?
               where sample_id=?""",
            (type(exc).__name__, str(exc), _now(), sample_id),
        )
        self.connection.commit()

    def rows(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("select * from tasks order by task_order"))

    def attempt_rows(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("select * from attempts order by sample_id,attempt"))

    def close(self) -> None:
        self.checkpoint_and_close()
