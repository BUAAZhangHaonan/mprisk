from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

router = APIRouter(prefix="/media", tags=["media"])


def allowed_roots() -> list[Path]:
    raw = os.environ.get(
        "MPRISK_MEDIA_ROOTS",
        "/home/team/zhanghaonan/TAFFC/mprisk-data/curation/media",
    )
    return [Path(item).resolve() for item in raw.split(":") if item.strip()]


def resolve_allowed(path: str) -> Path:
    target = Path(path).expanduser().resolve()
    for root in allowed_roots():
        if target == root or root in target.parents:
            return target
    raise HTTPException(status_code=403, detail="path outside allowed media roots")


def serve_asset(asset_id: str, audio: bool) -> Response | FileResponse:
    if not asset_id or Path(asset_id).is_absolute():
        raise HTTPException(status_code=403, detail="invalid media asset id")
    target = resolve_allowed(str(allowed_roots()[0] / asset_id))
    if not target.is_file():
        raise HTTPException(status_code=404, detail="media file not found")
    if audio:
        # extract the audio track on the fly so annotators can review audio-only
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-i", str(target), "-vn",
                    "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1",
                ],
                capture_output=True, timeout=120, check=True,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"audio extraction failed: {exc}")
        if not result.stdout:
            raise HTTPException(status_code=404, detail="no audio track")
        return Response(content=result.stdout, media_type="audio/wav")
    return FileResponse(target)


@router.get("/{asset_id:path}")
def media(asset_id: str, audio: bool = Query(False)):
    return serve_asset(asset_id, audio)
