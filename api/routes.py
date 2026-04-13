"""HTTP routes for the FastAPI service."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import metrics
from api.middleware import limiter
from api.schemas import (
    BatchJobResponse,
    ClaimReport,
    HealthResponse,
    JobStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


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
