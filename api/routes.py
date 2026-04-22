"""HTTP routes for the FastAPI service."""

from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import metrics
from api.feedback_store import utc_now_iso
from api.middleware import limiter
from api.schemas import (
    BatchJobResponse,
    ClaimReport,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    JobStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _record_drift(request: Request, report: dict) -> None:
    """Push a report into the drift monitor and update Prometheus counters."""
    monitor = getattr(request.app.state, "drift_monitor", None)
    if monitor is not None:
        monitor.record(report)

    # Prometheus per-label counters for alerting / Grafana.
    overall = report.get("overall_assessment")
    if overall:
        metrics.DRIFT_OVERALL.labels(assessment=overall).inc()
    for part in report.get("parts", []):
        metrics.DRIFT_PARTS.labels(part=part.get("part", "unknown")).inc()
        pdt = part.get("primary_damage_type")
        if pdt:
            metrics.DRIFT_DAMAGE_TYPE.labels(damage_type=pdt).inc()
        sev = (part.get("severity") or {}).get("grade")
        if sev:
            metrics.DRIFT_SEVERITY.labels(grade=sev).inc()
        if part.get("flagged_for_review", False):
            metrics.REVIEW_FLAGS.inc()


# --------------------------------------------------------------------------- #
# Auth config (public — tells the frontend whether MS SSO is enabled)
# --------------------------------------------------------------------------- #
@router.get("/auth/config", tags=["meta"])
async def auth_config() -> dict:
    import os

    client_id = os.environ.get("CDP_MS_CLIENT_ID", "")
    return {"client_id": client_id, "enabled": bool(client_id)}


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    assessor = getattr(request.app.state, "assessor", None)
    return HealthResponse(
        status="ok" if assessor is not None else "starting",
        version=request.app.version,
        device=str(getattr(assessor, "device", "unknown")),
        models_loaded=assessor is not None,
        pretrained_baseline=bool(getattr(assessor, "pretrained_baseline", True)),
    )


# --------------------------------------------------------------------------- #
# Prometheus
# --------------------------------------------------------------------------- #
@router.get("/metrics", tags=["meta"])
async def metrics_endpoint() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------- #
# Assess (sync, single image)
# --------------------------------------------------------------------------- #
@router.post("/assess", response_model=ClaimReport, tags=["assessment"])
@limiter.limit("30/minute")
async def assess(request: Request, file: UploadFile = File(...)) -> ClaimReport:
    assessor = getattr(request.app.state, "assessor", None)
    if assessor is None:
        metrics.ASSESSMENT_ERRORS.labels(reason="models_not_ready").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading, please retry in a moment.",
        )
    if not file.content_type or not file.content_type.startswith("image/"):
        metrics.ASSESSMENT_ERRORS.labels(reason="bad_content_type").inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected image content-type, got {file.content_type!r}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    t0 = time.time()
    try:
        report = assessor.assess(data, image_id=file.filename or "upload")
    except Exception as exc:
        logger.exception("Inference failed")
        metrics.ASSESSMENT_ERRORS.labels(reason="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    metrics.REQUESTS_TOTAL.labels(endpoint="/assess", status="200").inc()
    metrics.INFERENCE_LATENCY.labels(endpoint="/assess").observe(time.time() - t0)
    metrics.PARTS_DETECTED.observe(report.get("parts_detected", 0))

    # --- Drift monitoring + Prometheus counters ---
    _record_drift(request, report)

    return ClaimReport.model_validate(report)


# --------------------------------------------------------------------------- #
# Assess batch (async)
# --------------------------------------------------------------------------- #
@router.post("/assess/batch", response_model=BatchJobResponse, status_code=202, tags=["assessment"])
@limiter.limit("10/minute")
async def assess_batch(request: Request, files: list[UploadFile] = File(...)) -> BatchJobResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Max 50 images per batch job")

    b64_images: list[str] = []
    ids: list[str] = []
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Non-image upload: {f.filename}")
        data = await f.read()
        b64_images.append(base64.b64encode(data).decode("ascii"))
        ids.append(f.filename or "upload")

    # Enqueue Celery task
    try:
        from api.tasks import assess_images_task

        async_result = assess_images_task.apply_async(args=[b64_images, ids])
    except Exception as exc:
        logger.exception("Failed to enqueue job")
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}") from exc

    metrics.REQUESTS_TOTAL.labels(endpoint="/assess/batch", status="202").inc()
    return BatchJobResponse(job_id=async_result.id, status="queued")


# --------------------------------------------------------------------------- #
# Job status
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Feedback capture — Stage 1 of the human-in-the-loop retraining system.
#
# Accepts either:
#   Content-Type: application/json   → FeedbackRequest body, no image.
#   Content-Type: multipart/form-data → 'feedback' field (JSON string) + optional 'image' file.
# --------------------------------------------------------------------------- #
@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["feedback"],
)
@limiter.limit("60/minute")
async def submit_feedback(
    request: Request,
    feedback: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
) -> FeedbackResponse:
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        metrics.FEEDBACK_TOTAL.labels(outcome="store_unavailable").inc()
        raise HTTPException(status_code=503, detail="Feedback store not initialized.")

    # Parse body — accept JSON or multipart form.
    if feedback is not None:
        try:
            payload = FeedbackRequest.model_validate_json(feedback)
        except Exception as exc:
            metrics.FEEDBACK_TOTAL.labels(outcome="bad_payload").inc()
            raise HTTPException(status_code=400, detail=f"Invalid feedback JSON: {exc}") from exc
    else:
        try:
            raw = await request.json()
            payload = FeedbackRequest.model_validate(raw)
        except Exception as exc:
            metrics.FEEDBACK_TOTAL.labels(outcome="bad_payload").inc()
            raise HTTPException(status_code=400, detail=f"Invalid feedback body: {exc}") from exc

    image_bytes: bytes | None = None
    image_ct: str | None = None
    if image is not None:
        max_bytes = getattr(request.app.state, "feedback_max_bytes", 16 * 1024 * 1024)
        image_bytes = await image.read()
        if len(image_bytes) > max_bytes:
            metrics.FEEDBACK_TOTAL.labels(outcome="image_too_large").inc()
            raise HTTPException(
                status_code=413, detail=f"Image exceeds limit ({len(image_bytes)} > {max_bytes} bytes)."
            )
        if not image.content_type or not image.content_type.startswith("image/"):
            metrics.FEEDBACK_TOTAL.labels(outcome="bad_image_type").inc()
            raise HTTPException(status_code=400, detail=f"Unexpected image type {image.content_type!r}.")
        image_ct = image.content_type

    feedback_id = uuid.uuid4().hex
    manifest = {
        "feedback_id": feedback_id,
        "claim_id": payload.claim_id,
        "adjuster_id": payload.adjuster_id,
        "captured_at": utc_now_iso(),
        "has_image": image_bytes is not None,
        "notes": payload.notes,
        "schema_version": "1.0",
        "corrected_overall_assessment": payload.corrected_overall_assessment,
        "parts_delta": len(payload.corrected_parts),
    }

    corrected = {
        "corrected_parts": [p.model_dump() for p in payload.corrected_parts],
        "corrected_overall_assessment": payload.corrected_overall_assessment,
        "notes": payload.notes,
    }
    predicted = payload.original_report.model_dump()

    try:
        uri = store.put_bundle(
            claim_id=payload.claim_id,
            feedback_id=feedback_id,
            manifest=manifest,
            predicted=predicted,
            corrected=corrected,
            image_bytes=image_bytes,
            image_content_type=image_ct,
        )
    except Exception as exc:
        logger.exception("Failed to persist feedback bundle")
        metrics.FEEDBACK_TOTAL.labels(outcome="store_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {exc}") from exc

    metrics.FEEDBACK_TOTAL.labels(outcome="stored").inc()
    return FeedbackResponse(
        feedback_id=feedback_id, claim_id=payload.claim_id, stored_at=uri, status="stored"
    )


# --------------------------------------------------------------------------- #
# Admin: drift monitoring
# --------------------------------------------------------------------------- #
@router.get("/admin/drift", tags=["admin"])
async def drift_stats(request: Request) -> dict:
    monitor = getattr(request.app.state, "drift_monitor", None)
    if monitor is None:
        raise HTTPException(status_code=503, detail="Drift monitor not initialized.")
    return monitor.get_stats()


# --------------------------------------------------------------------------- #
# Admin: feedback browser
# --------------------------------------------------------------------------- #
@router.get("/admin/feedback", tags=["admin"])
async def list_feedback(
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Feedback store not initialized.")
    try:
        bundles = store.list_bundles()
    except Exception as exc:
        logger.exception("Failed to list feedback bundles")
        raise HTTPException(status_code=500, detail=f"Failed to list feedback: {exc}") from exc
    # Sort most-recent first by captured_at.
    bundles.sort(key=lambda b: b.get("captured_at", ""), reverse=True)
    return bundles[offset : offset + limit]


@router.get("/admin/feedback/{feedback_id}", tags=["admin"])
async def get_feedback(request: Request, feedback_id: str) -> dict:
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Feedback store not initialized.")
    try:
        bundle = store.get_bundle(feedback_id)
    except Exception as exc:
        logger.exception("Failed to read feedback bundle")
        raise HTTPException(status_code=500, detail=f"Failed to read feedback: {exc}") from exc
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"Feedback bundle {feedback_id!r} not found.")
    return bundle


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["assessment"])
async def job_status(job_id: str) -> JobStatusResponse:
    from celery.result import AsyncResult

    from api.celery_app import celery_app

    res = AsyncResult(job_id, app=celery_app)
    state = res.state
    status_map: dict[str, Any] = {
        "PENDING": "queued",
        "STARTED": "running",
        "RETRY": "running",
        "SUCCESS": "succeeded",
        "FAILURE": "failed",
    }
    mapped = status_map.get(state, "unknown")
    result = None
    error = None
    if mapped == "succeeded":
        try:
            result = [ClaimReport.model_validate(r) for r in res.get(timeout=1)]
        except Exception as exc:
            error = str(exc)
    elif mapped == "failed":
        error = str(res.info) if res.info else "unknown error"
    return JobStatusResponse(job_id=job_id, status=mapped, result=result, error=error)
