"""Prometheus metrics for the API service."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "cdp_api_requests_total",
    "Total API requests.",
    labelnames=("endpoint", "status"),
)

INFERENCE_LATENCY = Histogram(
    "cdp_inference_latency_seconds",
    "End-to-end inference latency.",
    labelnames=("endpoint",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

PARTS_DETECTED = Histogram(
    "cdp_parts_detected",
    "Number of parts detected per image.",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
)

ASSESSMENT_ERRORS = Counter(
    "cdp_assessment_errors_total",
    "Assessment errors by reason.",
    labelnames=("reason",),
)

FEEDBACK_TOTAL = Counter(
    "cdp_feedback_total",
    "Feedback submissions captured.",
    labelnames=("outcome",),
)
