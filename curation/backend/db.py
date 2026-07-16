from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def connect(path: str | Path = "curation/outputs/curation.sqlite") -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists samples (
            sample_id text primary key,
            source_dataset text,
            source_id text,
            protocol text,
            candidate_type text,
            payload_json text not null
        );
        create table if not exists candidate_labels (
            sample_id text primary key,
            payload_json text not null
        );
        create table if not exists llm_screening (
            sample_id text primary key,
            payload_json text not null
        );
        create table if not exists human_annotations (
            id integer primary key autoincrement,
            sample_id text not null,
            annotator_id text not null,
            payload_json text not null,
            created_at text default current_timestamp
        );
        create table if not exists adjudications (
            sample_id text primary key,
            payload_json text not null,
            created_at text default current_timestamp
        );
        create table if not exists exports (
            id integer primary key autoincrement,
            export_path text not null,
            created_at text default current_timestamp
        );
        """
    )
    conn.commit()


def upsert_sample(conn: sqlite3.Connection, sample: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into samples (sample_id, source_dataset, source_id, protocol, candidate_type, payload_json)
        values (?, ?, ?, ?, ?, ?)
        on conflict(sample_id) do update set
            source_dataset=excluded.source_dataset,
            source_id=excluded.source_id,
            protocol=excluded.protocol,
            candidate_type=excluded.candidate_type,
            payload_json=excluded.payload_json
        """,
        (
            sample["sample_id"],
            sample.get("source_dataset", ""),
            sample.get("source_id", ""),
            sample.get("protocol", ""),
            sample.get("candidate_type", "Ambiguous"),
            json.dumps(sample, ensure_ascii=False),
        ),
    )
    conn.execute(
        """
        insert into candidate_labels (sample_id, payload_json)
        values (?, ?)
        on conflict(sample_id) do update set payload_json=excluded.payload_json
        """,
        (sample["sample_id"], json.dumps(sample, ensure_ascii=False)),
    )
    conn.commit()


def insert_annotation(conn: sqlite3.Connection, annotation: dict[str, Any]) -> None:
    conn.execute(
        "insert into human_annotations (sample_id, annotator_id, payload_json) values (?, ?, ?)",
        (
            annotation["sample_id"],
            annotation["annotator_id"],
            json.dumps(annotation, ensure_ascii=False),
        ),
    )
    conn.commit()


def list_samples(conn: sqlite3.Connection, *, candidate_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    if candidate_type:
        rows = conn.execute(
            "select payload_json from samples where candidate_type = ? order by sample_id limit ?",
            (candidate_type, limit),
        ).fetchall()
    else:
        rows = conn.execute("select payload_json from samples order by sample_id limit ?", (limit,)).fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def get_sample(conn: sqlite3.Connection, sample_id: str) -> dict[str, Any] | None:
    row = conn.execute("select payload_json from samples where sample_id = ?", (sample_id,)).fetchone()
    return json.loads(row["payload_json"]) if row else None


def list_annotations(conn: sqlite3.Connection, sample_id: str | None = None) -> list[dict[str, Any]]:
    if sample_id:
        rows = conn.execute(
            "select payload_json from human_annotations where sample_id = ? order by id",
            (sample_id,),
        ).fetchall()
    else:
        rows = conn.execute("select payload_json from human_annotations order by id").fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def upsert_adjudication(conn: sqlite3.Connection, adjudication: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into adjudications (sample_id, payload_json) values (?, ?)
        on conflict(sample_id) do update set payload_json=excluded.payload_json
        """,
        (adjudication["sample_id"], json.dumps(adjudication, ensure_ascii=False)),
    )
    conn.commit()
