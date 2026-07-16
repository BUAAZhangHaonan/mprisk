from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(prefix="/media", tags=["media"])


@router.get("")
def media(path: str = Query(..., min_length=1)):
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise HTTPException(status_code=404, detail="media file not found")
    return FileResponse(target)
