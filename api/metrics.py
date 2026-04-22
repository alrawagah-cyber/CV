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

# --- Drift monitoring ---
DRIFT_DAMAGE_TYPE = Counter(
    "cdp_drift_damage_type_total",
    "Damage type predictions for drift tracking.",
    labelnames=("damage_type",),
)

DRIFT_SEVERITY = Counter(
    "cdp_drift_severity_total",
    "Severity grade predictions for drift tracking.",
    labelnames=("grade",),
)

DRIFT_OVERALL = Counter(
    "cdp_drift_overall_total",
    "Overall assessment predictions for drift tracking.",
    labelnames=("assessment",),
)

DRIFT_PARTS = Counter(
    "cdp_drift_parts_total",
    "Detected part types for drift tracking.",
    labelnames=("part",),
)

REVIEW_FLAGS = Counter(
    "cdp_review_flags_total",
    "Parts flagged for human review by active learning.",
)
