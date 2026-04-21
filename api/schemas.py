"""Pydantic models for the API request/response schemas."""

from __future__ import annotations

from typing import Literal

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
    damaged: bool = Field(default=True, description="False if L2 V2 flagged the part as no_damage.")
    damage_types: list[DamageTypeScore]
    damage_probs_all: dict[str, float]
    primary_damage_type: str | None = None
    severity: SeveritySchema | None = None
    recommendation: Literal["repair", "replace"] | None = None
    repair_probability: float | None = None
    replace_probability: float | None = None
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
    inference_ms: int | None = None
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
    detail: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "unknown"]
    result: list[ClaimReport] | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Feedback capture (Stage 1 of the human-in-the-loop retraining system)
# --------------------------------------------------------------------------- #
class FeedbackPart(BaseModel):
    """An adjuster-corrected part assessment.

    All fields match the ``PartAssessment`` shape except that the adjuster
    may drop ``detection_confidence`` / ``damage_probs_all`` — we don't need
    model probabilities on the corrected side.

    Set ``damaged=False`` to flag that the model hallucinated damage on a
    part that is actually clean. These rows become negative examples when
    the L2 V2 classifier is retrained with the ``no_damage`` class.
    """

    part: str
    bbox_xyxy_px: list[int] | None = None
    bbox_xyxy_norm: list[float] | None = None
    damaged: bool = Field(
        default=True,
        description="False if the part is actually undamaged (false positive correction).",
    )
    damage_types: list[str] = Field(
        default_factory=list,
        description="Corrected multi-label damage types (e.g. ['dent', 'shatter']).",
    )
    primary_damage_type: str | None = None
    severity: str | None = Field(
        default=None, description="Corrected severity grade (minor|moderate|severe|total_loss)."
    )
    recommendation: Literal["repair", "replace"] | None = None
    adjuster_notes: str | None = None


class FeedbackRequest(BaseModel):
    """Payload submitted to ``POST /feedback`` when an adjuster corrects a report."""

    claim_id: str = Field(..., min_length=1, max_length=128, description="Chubb-side claim identifier.")
    adjuster_id: str = Field(
        ..., min_length=1, max_length=128, description="Identity of the human who corrected the report."
    )
    original_report: ClaimReport
    corrected_parts: list[FeedbackPart] = Field(
        default_factory=list, description="Replacement/additional per-part annotations."
    )
    corrected_overall_assessment: str | None = Field(
        default=None,
        description="E.g. 'minor_damage', 'major_damage', 'total_loss' — free-form label from the adjuster UI.",
    )
    notes: str | None = Field(default=None, max_length=4000)


class FeedbackResponse(BaseModel):
    feedback_id: str
    claim_id: str
    stored_at: str = Field(..., description="URI (file:// or gs://) of the stored bundle.")
    status: Literal["stored"]
