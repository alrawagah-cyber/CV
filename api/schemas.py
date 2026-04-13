"""Pydantic models for the API request/response schemas."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DamageTypeScore(BaseModel):
    type: str = Field(..., description="Damage type label, e.g. 'dent'")
    probability: float = Field(..., ge=0.0, le=1.0)


class SeveritySchema(BaseModel):
    grade: str
    grade_index: int = Field(..., ge=0)
    grade_confidence: float = Field(..., ge=0.0, le=1.0)
    probs: dict[str, float]


class PartAssessment(BaseModel):
    part: str
    class_id: int
    detection_confidence: float
    bbox_xyxy_px: list[int]
    bbox_xyxy_norm: list[float]
    damage_types: list[DamageTypeScore]
    damage_probs_all: dict[str, float]
    primary_damage_type: Optional[str] = None
    severity: Optional[SeveritySchema] = None
    recommendation: Optional[Literal["repair", "replace"]] = None
    repair_probability: Optional[float] = None
    replace_probability: Optional[float] = None
    pretrained_baseline: bool


class ClaimReport(BaseModel):
    image_id: str
    image_width: int
    image_height: int
    parts_detected: int
    parts_damaged: int
    parts_requiring_replacement: int
    overall_assessment: str
    parts: list[PartAssessment]
    pretrained_baseline: bool
    model_versions: dict[str, str]
    warnings: list[str] = []
    inference_ms: Optional[int] = None
    schema_version: str = "1.0"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting"]
    version: str
    device: str
    models_loaded: bool
    pretrained_baseline: bool


class BatchJobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    detail: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "unknown"]
    result: Optional[list[ClaimReport]] = None
    error: Optional[str] = None
