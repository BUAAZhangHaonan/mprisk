from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

# read-only GLM 3-round annotation db (never written by the curation service)
GLM_ANNOTATION_DB = Path(
    os.environ.get("MPRISK_CURATION_DATA", "/home/team/zhanghaonan/TAFFC/mprisk-data/curation")
) / "glm_5_3_flash_annotation.sqlite"


def connect(path: str | Path = "curation/outputs/curation.sqlite") -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False)
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
        create index if not exists idx_annotations_sample on human_annotations (sample_id);
        create index if not exists idx_annotations_annotator on human_annotations (annotator_id);
        create unique index if not exists idx_annotations_unique on human_annotations (sample_id, annotator_id);
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


def upsert_llm_screening(conn: sqlite3.Connection, screening: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into llm_screening (sample_id, payload_json) values (?, ?)
        on conflict(sample_id) do update set payload_json=excluded.payload_json
        """,
        (screening["sample_id"], json.dumps(screening, ensure_ascii=False)),
    )
    conn.commit()


def get_llm_screening(conn: sqlite3.Connection, sample_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "select payload_json from llm_screening where sample_id = ?", (sample_id,)
    ).fetchone()
    return json.loads(row["payload_json"]) if row else None


def insert_annotation(conn: sqlite3.Connection, annotation: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into human_annotations (sample_id, annotator_id, payload_json)
        values (?, ?, ?)
        on conflict(sample_id, annotator_id) do update set
            payload_json=excluded.payload_json,
            created_at=current_timestamp
        """,
        (
            annotation["sample_id"],
            annotation["annotator_id"],
            json.dumps(annotation, ensure_ascii=False),
        ),
    )
    conn.commit()


def list_samples(
    conn: sqlite3.Connection,
    *,
    candidate_type: str | None = None,
    llm_type: str | None = None,
    human_type: str | None = None,
    protocol: str | None = None,
    exclude_annotator: str | None = None,
    only_annotator: str | None = None,
    disagreement_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    where = ""
    clauses: list[str] = []
    params: list[Any] = []
    if candidate_type:
        clauses.append("s.candidate_type = ?")
        params.append(candidate_type)
    if llm_type:
        clauses.append("ls.suggestion = ?")
        params.append(llm_type)
    if human_type:
        clauses.append(
            "exists (select 1 from human_annotations ha where ha.sample_id = s.sample_id "
            "and json_extract(ha.payload_json, '$.sample_type') = ?)"
        )
        params.append(human_type)
    if protocol:
        clauses.append("s.protocol = ?")
        params.append(protocol)
    if exclude_annotator:
        clauses.append(
            "s.sample_id not in (select sample_id from human_annotations where annotator_id = ?)"
        )
        params.append(exclude_annotator)
    if only_annotator:
        clauses.append(
            "s.sample_id in (select sample_id from human_annotations where annotator_id = ?)"
        )
        params.append(only_annotator)
    if disagreement_only:
        clauses.append("ls.suggestion is not null and s.candidate_type != ls.suggestion")
    if clauses:
        where = " where " + " and ".join(clauses)

    ls_join = """
        left join (
            select sample_id,
                   json_extract(payload_json, '$.sample_type_suggestion') as suggestion
            from llm_screening
        ) ls on ls.sample_id = s.sample_id
    """

    total = conn.execute(f"select count(*) from samples s{ls_join}{where}", params).fetchone()[0]

    query = f"""
        select s.payload_json as payload_json,
               ls.suggestion as llm_suggestion,
               ac.annotation_count as annotation_count,
               ac.annotators as annotators,
               ac.human_types as human_types
        from samples s
        {ls_join}
        left join (
            select sample_id,
                   count(distinct annotator_id) as annotation_count,
                   group_concat(distinct annotator_id) as annotators,
                   group_concat(distinct json_extract(payload_json, '$.sample_type')) as human_types
            from human_annotations group by sample_id
        ) ac on ac.sample_id = s.sample_id
        {where}
        order by s.sample_id
        limit ? offset ?
    """
    page_params = [*params, limit, offset]
    rows = conn.execute(query, page_params).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = json.loads(row["payload_json"])
        annotators_str = row["annotators"] or ""
        annotators = [a for a in annotators_str.split(",") if a] if annotators_str else []
        human_types_str = row["human_types"] or ""
        human_types = [t for t in human_types_str.split(",") if t] if human_types_str else []
        # always prefer llm_screening table data over stale payload values
        llm_sugg = row["llm_suggestion"]
        if llm_sugg:
            item["llm_sample_type_suggestion"] = llm_sugg
            item["llm_agrees"] = item.get("candidate_type") == llm_sugg
        item["annotation_count"] = row["annotation_count"] or 0
        item["annotators"] = annotators
        item["human_types"] = human_types
        result.append(item)
    return result, total


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


def progress_stats(conn: sqlite3.Connection, annotator_id: str | None = None) -> dict[str, Any]:
    annotation_filter = ""
    annotation_params: tuple[str, ...] = ()
    if annotator_id is not None:
        annotation_filter = "where annotator_id = ?"
        annotation_params = (annotator_id,)
    groups: dict[str, dict[str, Any]] = {}
    rows = conn.execute(
        f"""
        select s.candidate_type as candidate_type,
               s.protocol as protocol,
               count(*) as total,
               sum(case when a.n >= 1 then 1 else 0 end) as annotated_once,
               sum(case when a.n >= 2 then 1 else 0 end) as annotated_twice
        from samples s
        left join (
            select sample_id, count(distinct annotator_id) as n
            from human_annotations {annotation_filter} group by sample_id
        ) a on a.sample_id = s.sample_id
        group by s.candidate_type, s.protocol
        """, annotation_params).fetchall()
    for row in rows:
        key = f"{row['protocol']}:{row['candidate_type']}"
        groups[key] = {
            "protocol": row["protocol"],
            "candidate_type": row["candidate_type"],
            "total": row["total"],
            "annotated_once": row["annotated_once"] or 0,
            "annotated_twice": row["annotated_twice"] or 0,
        }
    total_annotations = conn.execute(f"select count(distinct sample_id) from human_annotations {annotation_filter}", annotation_params).fetchone()[0]
    annotators = [
        {"annotator_id": row["annotator_id"], "count": row["n"]}
        for row in conn.execute(
            f"select annotator_id, count(distinct sample_id) as n from human_annotations {annotation_filter} group by annotator_id order by n desc", annotation_params
        ).fetchall()
    ]
    return {"groups": list(groups.values()), "total_annotations": total_annotations, "annotators": annotators}


def upsert_adjudication(conn: sqlite3.Connection, adjudication: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into adjudications (sample_id, payload_json) values (?, ?)
        on conflict(sample_id) do update set payload_json=excluded.payload_json
        """,
        (adjudication["sample_id"], json.dumps(adjudication, ensure_ascii=False)),
    )
    conn.commit()


_GLM_LABEL_NAMES = {1: "positive", 0: "neutral", -1: "negative"}


def _glm_conf_text(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "未知"


def _glm_summary(
    final: sqlite3.Row,
    rounds: list[sqlite3.Row],
    adjudication: dict[str, Any] | None,
) -> str:
    label = _GLM_LABEL_NAMES.get(final["final_label"], "unknown")
    if final["method"] == "adjudicated":
        rationale = (adjudication or {}).get("rationale") or "无记录"
        confidence = (adjudication or {}).get("confidence")
        return (
            f"三轮意见存在分歧，模型终裁为「{label}」"
            f"（置信 {_glm_conf_text(confidence)}）。裁决理由：{rationale}"
        )
    majority_rounds = [r for r in rounds if r["label"] == final["final_label"]]
    evidence = ""
    if majority_rounds:
        evidence = max(majority_rounds, key=lambda r: (r["confidence"] or 0.0))["evidence"] or ""
    return (
        f"三轮交叉标注多数一致为「{label}」"
        f"（{len(majority_rounds)}/{len(rounds)} 一致，平均置信 {_glm_conf_text(final['mean_confidence'])}）。"
        f"主要依据：{evidence}"
    )


def _glm_modality(conn: sqlite3.Connection, source_id: str, modality: str) -> dict[str, Any] | None:
    final = conn.execute(
        "select final_label, method, agreement, mean_confidence, adjudication_id "
        "from glm_final where source_id = ? and modality = ?",
        (source_id, modality),
    ).fetchone()
    if final is None or final["final_label"] is None:
        return None
    rounds = conn.execute(
        "select round, label, confidence, evidence from glm_runs "
        "where source_id = ? and modality = ? and status = 'ok' order by round",
        (source_id, modality),
    ).fetchall()
    adjudication = None
    if final["adjudication_id"] is not None:
        adj = conn.execute(
            "select rationale, confidence from glm_adjudications where id = ?",
            (final["adjudication_id"],),
        ).fetchone()
        if adj is not None:
            adjudication = {"rationale": adj["rationale"], "confidence": adj["confidence"]}
    return {
        "final_label": _GLM_LABEL_NAMES.get(final["final_label"], "unknown"),
        "final_label_raw": final["final_label"],
        "method": final["method"],
        "agreement": final["agreement"],
        "mean_confidence": final["mean_confidence"],
        "summary": _glm_summary(final, rounds, adjudication),
        "rounds": [
            {
                "round": r["round"],
                "label": _GLM_LABEL_NAMES.get(r["label"], "unknown"),
                "confidence": r["confidence"],
                "evidence": r["evidence"],
            }
            for r in rounds
        ],
        "adjudication": adjudication,
    }


def get_glm_annotation(source_id: str) -> dict[str, Any] | None:
    """Read the GLM 3-round model suggestion for one source_id ({"V": ..., "T": ...}).

    Opens the annotation sqlite strictly read-only; returns None when the source
    has no usable final label on either modality.
    """
    if not source_id:
        return None
    try:
        conn = sqlite3.connect(f"file:{GLM_ANNOTATION_DB}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None
    try:
        conn.row_factory = sqlite3.Row
        result = {m: _glm_modality(conn, source_id, m) for m in ("V", "T")}
    finally:
        conn.close()
    return result if result["V"] or result["T"] else None
