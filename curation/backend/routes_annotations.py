from __future__ import annotations

from fastapi import APIRouter, Depends

from curation.backend.app_state import get_conn
from curation.backend.db import insert_annotation, list_annotations
from curation.backend.models import Annotation

router = APIRouter(prefix="/annotations", tags=["annotations"])


@router.post("")
def save_annotation(annotation: Annotation, conn=Depends(get_conn)):
    insert_annotation(conn, annotation.model_dump())
    return {"status": "ok"}


@router.get("")
def annotations(sample_id: str | None = None, conn=Depends(get_conn)):
    return {"items": list_annotations(conn, sample_id=sample_id)}
