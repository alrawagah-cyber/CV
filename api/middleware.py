"""Structlog request logging + rate-limit setup."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Awaitable, Callable

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
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path,
                                               method=request.method)
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
