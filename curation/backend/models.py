from __future__ import annotations

from pydantic import BaseModel, Field


class Sample(BaseModel):
    sample_id: str
    source_dataset: str = ""
    source_id: str = ""
    protocol: str = ""
    candidate_type: str = "Ambiguous"
    payload: dict = Field(default_factory=dict)


class Annotation(BaseModel):
    sample_id: str
    annotator_id: str
    m1_label: str
    m2_label: str
    joint_label: str
    m1_specific_affect: str = ""
    m2_specific_affect: str = ""
    joint_specific_affect: str = ""
    m1_is_clear: bool = False
    m2_is_clear: bool = False
    joint_is_clear: bool = False
    m1_confidence: float = 0.0
    m2_confidence: float = 0.0
    joint_confidence: float = 0.0
    sample_type: str = "Ambiguous"
    dominant_modality: str = "unclear"
    quality_flags: list[str] = Field(default_factory=list)
    notes: str = ""


class Adjudication(BaseModel):
    sample_id: str
    final_label: dict
    adjudicator_id: str = ""
