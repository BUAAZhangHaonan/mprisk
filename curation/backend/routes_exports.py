from __future__ import annotations

from fastapi import APIRouter, Depends

from curation.backend.app_state import get_conn
from curation.backend.db import list_annotations
from curation.scripts.adjudicate_annotations import adjudicate_all

router = APIRouter(prefix="/exports", tags=["exports"])


@router.get("/adjudication-preview")
def adjudication_preview(conn=Depends(get_conn)):
    return {"items": adjudicate_all(list_annotations(conn))}
