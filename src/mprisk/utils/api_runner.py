"""Shared base for SQLite-backed resumable API batch runners.

Four pipeline stages (GT Description generation, prefill cache extraction,
Misread judgment, Diagnostic Affect Description generation) all drive a
SQLite ledger that records per-task status, attempts, signatures, and result
payloads. Their public APIs differ (different task identity, different
materialization contracts), but the boilerplate around the sqlite3
connection is identical:

- create the parent directory and open the database
- enable WAL journaling and (optionally) FULL synchronous mode
- install the per-stage schema (``CREATE TABLE IF NOT EXISTS``)
- reset ``running`` rows back to ``pending`` on startup
- gate a run on an immutable ``metadata.signature`` value
- count rows grouped by status, checkpoint WAL on close

This module factors that boilerplate into :class:`SqliteLedgerBase`.
Stage-specific ledgers subclass it and keep their own
``prepare``/``complete``/``fail``/``pending``/materialization methods.

The class intentionally does not own a ``schema`` attribute: each subclass
passes its own ``CREATE TABLE`` script via ``schema_sql`` so the on-disk
schema (and therefore external readers over ``batch_state.sqlite3``,
``manifest.jsonl`` etc.) is byte-for-byte unchanged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from mprisk.utils.io import canonical_json as _canonical_json, now_iso as _now


class SqliteLedgerBase:
    """Common SQLite setup for resumable API batch ledgers.

    Subclasses must call ``super().__init__(path, schema_sql=...)``. The
    ``connection`` attribute is intentionally public because downstream
    tests and materialization helpers read or write rows directly via SQL.
    """

    #: SQL to flip ``running`` rows back to ``pending`` on startup. Stages
    #: that do not use a ``running`` status (or use a different transition)
    #: override this to an empty string.
    running_to_pending_sql: str = "UPDATE tasks SET status='pending' WHERE status='running'"

    #: SQL to flip ``failed`` rows back to ``pending`` when ``retry_failed``
    #: is requested. ``None`` means the stage rejects retries (its
    #: ``prepare`` will raise if asked to retry).
    failed_to_pending_sql: str | None = (
        "UPDATE tasks SET status='pending' WHERE status='failed'"
    )

    def __init__(
        self,
        path: str | Path,
        *,
        schema_sql: str,
        synchronous: str = "FULL",
        retry_failed: bool = False,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.path = target
        self.connection = sqlite3.connect(target)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        if synchronous:
            self.connection.execute(f"PRAGMA synchronous={synchronous}")
        if schema_sql:
            self.connection.executescript(schema_sql)
        if self.running_to_pending_sql:
            self.connection.execute(self.running_to_pending_sql)
        if retry_failed and self.failed_to_pending_sql:
            self.connection.execute(self.failed_to_pending_sql)
        self.connection.commit()

    # ------------------------------------------------------------------
    # Metadata-signature gating (used by prefill / misread / diagnostic)
    # ------------------------------------------------------------------

    def check_metadata_signature(
        self,
        *,
        key: str,
        value: Any,
        retry_failed: bool = False,
        error_message: str = "Existing ledger signature does not match this run",
        clear_failed_payload: bool = False,
        clear_failed_payload_sql: str | None = None,
    ) -> None:
        """Persist and verify an immutable per-run signature.

        Stores ``value`` under ``metadata[key]`` if absent. If a previous
        value exists and differs, raises ``ValueError(error_message)``.

        When ``retry_failed`` is true, the configured
        :attr:`failed_to_pending_sql` is run after the signature matches.
        When ``clear_failed_payload`` is true, ``clear_failed_payload_sql``
        is run instead (use this when failed rows also carry payload
        columns like ``error_type`` / ``traceback`` that must be cleared
        alongside the status flip).
        """
        encoded = _canonical_json(value)
        with self.connection:
            current = self.connection.execute(
                "SELECT value FROM metadata WHERE key=?", (key,)
            ).fetchone()
            if current is not None and current["value"] != encoded:
                raise ValueError(error_message)
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES(?,?)",
                (key, encoded),
            )
            if retry_failed:
                if clear_failed_payload and clear_failed_payload_sql:
                    self.connection.execute(clear_failed_payload_sql)
                elif self.failed_to_pending_sql:
                    self.connection.execute(self.failed_to_pending_sql)

    def metadata_value(self, key: str) -> Any | None:
        """Return the raw stored metadata value (or ``None`` if absent)."""
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else row["value"]

    # ------------------------------------------------------------------
    # Common status queries
    # ------------------------------------------------------------------

    def ids_by_status(
        self,
        statuses: tuple[str, ...],
        *,
        id_column: str,
        order_by: str,
    ) -> list[str]:
        """Return ids whose status is in ``statuses``.

        ``id_column`` is the SQL column to project (``task_id`` or
        ``sample_id``). ``order_by`` is an arbitrary SQL ``ORDER BY``
        expression against the ``tasks`` table.
        """
        placeholders = ",".join("?" for _ in statuses)
        rows = self.connection.execute(
            f"SELECT {id_column} FROM tasks WHERE status IN ({placeholders}) "
            f"ORDER BY {order_by}",
            statuses,
        ).fetchall()
        return [row[0] for row in rows]

    def status_counts(self) -> dict[str, int]:
        """Return ``{status: count}`` over the ``tasks`` table."""
        return {
            row["status"]: int(row["n"])
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
            ).fetchall()
        }

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def checkpoint_and_close(self) -> None:
        """WAL checkpoint (TRUNCATE) and close. Safe to call once."""
        try:
            self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            self.connection.close()


#: Convenience re-export so stage modules can build ``now_iso`` /
#: ``canonical_json`` payloads without re-importing :mod:`mprisk.utils.io`.
now_iso = _now
canonical_json = _canonical_json
