from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from curation.backend.routes_annotations import router as annotations_router
from curation.backend.routes_exports import router as exports_router
from curation.backend.routes_media import router as media_router
from curation.backend.routes_samples import router as samples_router
from curation.backend.app_state import get_conn
from curation.backend.db import progress_stats

app = FastAPI(title="MPRisk Curation", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_html(request, call_next):
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if "text/html" in ct:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


app.include_router(samples_router)
app.include_router(annotations_router)
app.include_router(exports_router)
app.include_router(media_router)
app.include_router(samples_router,prefix="/api")
app.include_router(annotations_router,prefix="/api")
app.include_router(exports_router,prefix="/api")
app.include_router(media_router,prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/api/health")
def api_health(): return {"status":"ok"}
@app.get("/api/progress")
def api_progress(conn=Depends(get_conn)): return progress_stats(conn)
@app.get("/api/annotators/statistics")
def annotator_statistics(conn=Depends(get_conn)): return progress_stats(conn)
@app.get("/api/adjudication/preview")
def adjudication_preview(): return {"items":[]}


# serve the built frontend so annotators only need the backend port
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")

