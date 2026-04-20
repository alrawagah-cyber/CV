"""FastAPI application factory + lifespan."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from api.middleware import APIKeyMiddleware, RequestLoggingMiddleware, configure_structlog, limiter
from api.routes import router

API_VERSION = "0.1.0"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; gracefully release on shutdown."""
    configure_structlog()
    cfg_path = os.environ.get("CDP_INFERENCE_CONFIG", "configs/inference.yaml")
    load_models = os.environ.get("CDP_LOAD_MODELS", "1") != "0"

    # In containerized deployments, pull weights from GCS before loading models.
    # Controlled via CDP_WEIGHTS_BUCKET env var — no-op when unset.
    try:
        from api.weight_fetcher import fetch_weights_if_configured

        refreshed = fetch_weights_if_configured()
        if refreshed:
            logger.info("Fetched %d weight file(s) from GCS: %s", len(refreshed), refreshed)
    except Exception as exc:
        logger.exception("Weight fetch failed: %s", exc)
        # Don't hard-fail startup; the assessor will log specific errors if
        # a weight file is missing.

    if load_models:
        try:
            from inference.claim_assessor import ClaimAssessor

            logger.info("Loading ClaimAssessor from %s ...", cfg_path)
            app.state.assessor = ClaimAssessor.from_config(cfg_path)
            logger.info("ClaimAssessor loaded (baseline=%s).", app.state.assessor.pretrained_baseline)
        except Exception as exc:
            logger.exception("Failed to load ClaimAssessor: %s", exc)
            app.state.assessor = None
    else:
        logger.warning("CDP_LOAD_MODELS=0 set — running API without models (test mode).")
        app.state.assessor = None

    # Feedback store is always loaded (local backend by default; no external deps).
    try:
        from api.feedback_store import build_store_from_config, load_feedback_config

        fb_cfg_path = os.environ.get("CDP_FEEDBACK_CONFIG", "configs/feedback.yaml")
        fb_cfg = load_feedback_config(fb_cfg_path)
        app.state.feedback_store = build_store_from_config(fb_cfg)
        app.state.feedback_max_bytes = int(fb_cfg.get("max_image_bytes", 16 * 1024 * 1024))
        logger.info("Feedback store ready (backend=%s).", fb_cfg.get("backend", "local"))
    except Exception as exc:
        logger.exception("Failed to initialize feedback store: %s", exc)
        app.state.feedback_store = None
        app.state.feedback_max_bytes = 0

    yield

    app.state.assessor = None
    app.state.feedback_store = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Car Damage Assessment API",
        version=API_VERSION,
        description=(
            "Three-layer enterprise pipeline for insurance-grade car damage "
            "assessment: (1) part detection, (2) damage-type classification, "
            "(3) severity + repair/replace recommendation."
        ),
        lifespan=lifespan,
    )

    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.limiter = limiter
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(APIKeyMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    app.include_router(router)

    # Serve the web frontend at /ui (single-file, no build step).
    from fastapi.responses import FileResponse

    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

    @app.get("/ui", include_in_schema=False)
    async def ui():
        return FileResponse(frontend_path, media_type="text/html")

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
