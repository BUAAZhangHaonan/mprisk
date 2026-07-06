from __future__ import annotations


def require_annotator(annotator_id: str | None) -> str:
    if not annotator_id:
        raise ValueError("annotator_id is required")
    return annotator_id
