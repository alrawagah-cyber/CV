"""FastAPI application factory + lifespan."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from api.middleware import RequestLoggingMiddleware, configure_structlog, limiter
from api.routes import router


API_VERSION = "0.1.0"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; gracefully release on shutdown."""
    configure_structlog()
    cfg_path = os.environ.get("CDP_INFERENCE_CONFIG", "configs/inference.yaml")
    load_models = os.environ.get("CDP_LOAD_MODELS", "1") != "0"

    if load_models:
        try:
            from inference.claim_assessor import ClaimAssessor
            logger.info("Loading ClaimAssessor from %s ...", cfg_path)
            app.state.assessor = ClaimAssessor.from_config(cfg_path)
            logger.info("ClaimAssessor loaded (baseline=%s).",
                        app.state.assessor.pretrained_baseline)
        except Exception as exc:
            logger.exception("Failed to load ClaimAssessor: %s", exc)
            app.state.assessor = None
    else:
        logger.warning("CDP_LOAD_MODELS=0 set — running API without models (test mode).")
        app.state.assessor = None

    yield

    app.state.assessor = None


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

    app.state.limiter = limiter
    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
