"""Structlog request logging + rate-limit setup."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware


def configure_structlog() -> None:
    """Configure stdlib + structlog for JSON structured logs."""
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach a request_id and log request + response."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        log = structlog.get_logger("api")
        structlog.contextvars.bind_contextvars(
            request_id=request_id, path=request.url.path, method=request.method
        )
        t0 = time.time()
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            log.exception("request_failed", error=str(exc))
            structlog.contextvars.clear_contextvars()
            raise
        duration_ms = int((time.time() - t0) * 1000)
        response.headers["x-request-id"] = request_id
        log.info("request", status_code=response.status_code, duration_ms=duration_ms)
        structlog.contextvars.clear_contextvars()
        return response


limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])


API_KEY_HEADER = "x-api-key"
_OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid API key.

    Set the env var ``CDP_API_KEYS`` to a comma-separated list of allowed keys.
    If the var is unset or empty, auth is **disabled** (open access) so local
    development doesn't require a key.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        import os

        raw = os.environ.get("CDP_API_KEYS", "")
        if not raw:
            return await call_next(request)

        if request.url.path in _OPEN_PATHS or request.url.path.startswith("/docs"):
            return await call_next(request)

        valid_keys = {k.strip() for k in raw.split(",") if k.strip()}
        provided = request.headers.get(API_KEY_HEADER)
        if not provided or provided not in valid_keys:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
        return await call_next(request)
