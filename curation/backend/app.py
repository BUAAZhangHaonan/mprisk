from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from curation.backend.routes_annotations import router as annotations_router
from curation.backend.routes_exports import router as exports_router
from curation.backend.routes_samples import router as samples_router

app = FastAPI(title="MPRisk Curation", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(samples_router)
app.include_router(annotations_router)
app.include_router(exports_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
