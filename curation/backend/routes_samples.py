from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from curation.backend.app_state import get_conn
from curation.backend.db import (
    get_glm_annotation,
    get_llm_screening,
    get_sample,
    list_annotations,
    list_samples,
    progress_stats,
)

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("")
def queue(
    candidate_type: str | None = None,
    llm_type: str | None = None,
    human_type: str | None = None,
    protocol: str | None = None,
    exclude_annotator: str | None = None,
    only_annotator: str | None = None,
    disagreement_only: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    conn=Depends(get_conn),
):
    items, total = list_samples(
        conn,
        candidate_type=candidate_type,
        llm_type=llm_type,
        human_type=human_type,
        protocol=protocol,
        exclude_annotator=exclude_annotator,
        only_annotator=only_annotator,
        disagreement_only=disagreement_only,
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/progress")
def progress(conn=Depends(get_conn)):
    return progress_stats(conn)


@router.get("/{sample_id}")
def detail(sample_id: str, conn=Depends(get_conn)):
    sample = get_sample(conn, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="sample not found")
    # always read from llm_screening table (real Gemini data), ignore stale payload
    screening = get_llm_screening(conn, sample_id)
    annotations = list_annotations(conn, sample_id=sample_id)
    # override top-level fields with real data
    if screening:
        sample["llm_sample_type_suggestion"] = screening.get("sample_type_suggestion")
        sample["llm_agrees"] = sample.get("candidate_type") == screening.get("sample_type_suggestion")
        sample["llm_screening"] = screening
    else:
        sample["llm_screening"] = None
        sample["llm_sample_type_suggestion"] = None
        sample["llm_agrees"] = None
    # GLM 3-round annotation suggestion (read-only side db, keyed by source_id)
    glm = get_glm_annotation(sample.get("source_id") or "")
    sample["glm_annotation"] = glm
    v_final = glm["V"]["final_label"] if glm and glm["V"] else None
    t_final = glm["T"]["final_label"] if glm and glm["T"] else None
    sample["glm_joint"] = (
        {
            "relation": "aligned" if v_final == t_final else "conflict",
            "labels": {"V": v_final, "T": t_final},
        }
        if v_final and t_final
        else None
    )
    return {
        **sample,
        "annotations": annotations,
        "annotation_count": len({a.get("annotator_id") for a in annotations}),
    }
