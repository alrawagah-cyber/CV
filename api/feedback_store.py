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

    @abstractmethod
    def list_bundles(self) -> list[dict[str, Any]]:
        """Return summary dicts for every stored feedback bundle."""

    @abstractmethod
    def get_bundle(self, feedback_id: str) -> dict[str, Any] | None:
        """Return the full bundle (manifest + predicted + corrected) or ``None``."""


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

    def list_bundles(self) -> list[dict[str, Any]]:
        bundles: list[dict[str, Any]] = []
        if not self.root.exists():
            return bundles
        for manifest_path in self.root.rglob("manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                logger.warning("Skipping unreadable manifest at %s", manifest_path)
                continue
            bundle_dir = manifest_path.parent
            has_image = any(bundle_dir.glob("image.*"))
            bundles.append(
                {
                    "feedback_id": manifest.get("feedback_id", bundle_dir.name),
                    "claim_id": manifest.get("claim_id", bundle_dir.parent.name),
                    "adjuster_id": manifest.get("adjuster_id", ""),
                    "captured_at": manifest.get("captured_at", ""),
                    "has_image": has_image,
                    "parts_delta": manifest.get("parts_delta", 0),
                    "notes": manifest.get("notes"),
                }
            )
        return bundles

    def get_bundle(self, feedback_id: str) -> dict[str, Any] | None:
        safe = _sanitize_segment(feedback_id)
        if not self.root.exists():
            return None
        # Search for the bundle directory across all claim_id folders.
        for candidate in self.root.iterdir():
            bundle_dir = candidate / safe
            if bundle_dir.is_dir() and (bundle_dir / "manifest.json").exists():
                try:
                    manifest = json.loads((bundle_dir / "manifest.json").read_text())
                    predicted = json.loads((bundle_dir / "predicted.json").read_text())
                    corrected = json.loads((bundle_dir / "corrected.json").read_text())
                except Exception:
                    logger.warning("Failed to read bundle at %s", bundle_dir)
                    return None
                return {"manifest": manifest, "predicted": predicted, "corrected": corrected}
        return None


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

    def list_bundles(self) -> list[dict[str, Any]]:
        self._ensure_client()
        bundles: list[dict[str, Any]] = []
        prefix = f"{self.prefix}/"
        manifest_suffix = "/manifest.json"
        blobs = self._client.list_blobs(self._bucket, prefix=prefix)
        for blob in blobs:
            if not blob.name.endswith(manifest_suffix):
                continue
            try:
                data = blob.download_as_text()
                manifest = json.loads(data)
            except Exception:
                logger.warning("Skipping unreadable manifest at %s", blob.name)
                continue
            bundle_prefix = blob.name[: -len("manifest.json")]
            has_image = any(
                True
                for b in self._client.list_blobs(self._bucket, prefix=bundle_prefix)
                if b.name.startswith(bundle_prefix + "image.")
            )
            bundles.append(
                {
                    "feedback_id": manifest.get("feedback_id", ""),
                    "claim_id": manifest.get("claim_id", ""),
                    "adjuster_id": manifest.get("adjuster_id", ""),
                    "captured_at": manifest.get("captured_at", ""),
                    "has_image": has_image,
                    "parts_delta": manifest.get("parts_delta", 0),
                    "notes": manifest.get("notes"),
                }
            )
        return bundles

    def get_bundle(self, feedback_id: str) -> dict[str, Any] | None:
        self._ensure_client()
        safe = _sanitize_segment(feedback_id)
        # Scan for a matching bundle across all claim_id sub-prefixes.
        prefix = f"{self.prefix}/"
        target_suffix = f"/{safe}/manifest.json"
        blobs = self._client.list_blobs(self._bucket, prefix=prefix)
        manifest_blob = None
        for blob in blobs:
            if blob.name.endswith(target_suffix):
                manifest_blob = blob
                break
        if manifest_blob is None:
            return None
        bundle_prefix = manifest_blob.name[: -len("manifest.json")]
        try:
            manifest = json.loads(manifest_blob.download_as_text())
            predicted_blob = self._bucket.blob(bundle_prefix + "predicted.json")
            predicted = json.loads(predicted_blob.download_as_text())
            corrected_blob = self._bucket.blob(bundle_prefix + "corrected.json")
            corrected = json.loads(corrected_blob.download_as_text())
        except Exception:
            logger.warning("Failed to read bundle at %s", bundle_prefix)
            return None
        return {"manifest": manifest, "predicted": predicted, "corrected": corrected}

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
