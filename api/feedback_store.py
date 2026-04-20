"""Feedback storage backends.

Writes a *bundle* per submission:

    feedback/<claim_id>/<feedback_id>/
        manifest.json   — metadata
        predicted.json  — model's original ClaimReport
        corrected.json  — adjuster's corrected ClaimReport
        image.jpg       — original image (optional)

Two backends are supported:

* ``local``  — writes under a local directory (dev, tests, on-prem).
* ``gcs``    — writes to a Google Cloud Storage bucket (production).

GCS is imported lazily so tests and local runs don't need
``google-cloud-storage`` installed.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackStore(ABC):
    """Abstract interface for persisting adjuster feedback bundles."""

    @abstractmethod
    def put_bundle(
        self,
        claim_id: str,
        feedback_id: str,
        manifest: dict[str, Any],
        predicted: dict[str, Any],
        corrected: dict[str, Any],
        image_bytes: bytes | None = None,
        image_content_type: str | None = None,
    ) -> str:
        """Persist a feedback bundle; return the URI of the bundle directory."""


class LocalFeedbackStore(FeedbackStore):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bundle(
        self,
        claim_id: str,
        feedback_id: str,
        manifest: dict[str, Any],
        predicted: dict[str, Any],
        corrected: dict[str, Any],
        image_bytes: bytes | None = None,
        image_content_type: str | None = None,
    ) -> str:
        safe_claim = _sanitize_segment(claim_id)
        safe_feedback = _sanitize_segment(feedback_id)
        bundle = self.root / safe_claim / safe_feedback
        bundle.mkdir(parents=True, exist_ok=True)

        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        (bundle / "predicted.json").write_text(json.dumps(predicted, indent=2, default=str))
        (bundle / "corrected.json").write_text(json.dumps(corrected, indent=2, default=str))
        if image_bytes is not None:
            ext = _ext_for_content_type(image_content_type)
            (bundle / f"image{ext}").write_bytes(image_bytes)

        uri = bundle.resolve().as_uri()
        logger.info("Stored feedback bundle at %s", uri)
        return uri


class GcsFeedbackStore(FeedbackStore):
    """Google Cloud Storage backend. Lazy-initialized on first write."""

    def __init__(self, bucket: str, prefix: str = "feedback"):
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self._client = None
        self._bucket = None
        self._lock = threading.Lock()

    def _ensure_client(self) -> None:
        if self._bucket is not None:
            return
        with self._lock:
            if self._bucket is not None:
                return
            from google.cloud import storage  # lazy import

            self._client = storage.Client()
            self._bucket = self._client.bucket(self.bucket_name)

    def put_bundle(
        self,
        claim_id: str,
        feedback_id: str,
        manifest: dict[str, Any],
        predicted: dict[str, Any],
        corrected: dict[str, Any],
        image_bytes: bytes | None = None,
        image_content_type: str | None = None,
    ) -> str:
        self._ensure_client()
        safe_claim = _sanitize_segment(claim_id)
        safe_feedback = _sanitize_segment(feedback_id)
        base = f"{self.prefix}/{safe_claim}/{safe_feedback}"

        self._upload_json(f"{base}/manifest.json", manifest)
        self._upload_json(f"{base}/predicted.json", predicted)
        self._upload_json(f"{base}/corrected.json", corrected)
        if image_bytes is not None:
            ext = _ext_for_content_type(image_content_type)
            blob = self._bucket.blob(f"{base}/image{ext}")
            blob.upload_from_string(image_bytes, content_type=image_content_type or "image/jpeg")

        uri = f"gs://{self.bucket_name}/{base}"
        logger.info("Stored feedback bundle at %s", uri)
        return uri

    def _upload_json(self, key: str, body: dict[str, Any]) -> None:
        blob = self._bucket.blob(key)
        blob.upload_from_string(json.dumps(body, indent=2, default=str), content_type="application/json")


def build_store_from_config(cfg: dict[str, Any]) -> FeedbackStore:
    backend = (cfg.get("backend") or "local").lower()
    if backend == "local":
        root = cfg.get("local_root") or "data/feedback"
        return LocalFeedbackStore(root)
    if backend == "gcs":
        bucket = cfg.get("bucket")
        if not bucket:
            raise ValueError("feedback.yaml backend=gcs requires 'bucket'")
        prefix = cfg.get("prefix", "feedback")
        return GcsFeedbackStore(bucket=bucket, prefix=prefix)
    raise ValueError(f"Unknown feedback backend {backend!r}")


def load_feedback_config(path: str | Path) -> dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.exists():
        return {"backend": "local", "local_root": "data/feedback"}
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def _sanitize_segment(s: str) -> str:
    """Allow alphanumerics, ``-`` and ``_``; replace anything else with ``_``."""
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s).strip("_") or "unknown"


def _ext_for_content_type(ct: str | None) -> str:
    if not ct:
        return ".jpg"
    ct = ct.lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    return ".jpg"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
