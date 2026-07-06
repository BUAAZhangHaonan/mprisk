from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from curation.backend.app_state import get_conn
from curation.backend.db import get_sample, list_samples

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("")
def queue(candidate_type: str | None = None, limit: int = Query(100, le=500), conn=Depends(get_conn)):
    return {"items": list_samples(conn, candidate_type=candidate_type, limit=limit)}


@router.get("/{sample_id}")
def detail(sample_id: str, conn=Depends(get_conn)):
    sample = get_sample(conn, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="sample not found")
    return sample
