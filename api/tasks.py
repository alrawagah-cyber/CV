"""Celery tasks for batched assessment."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from celery.exceptions import MaxRetriesExceededError

from api.celery_app import celery_app

logger = logging.getLogger(__name__)

# Module-level assessor to avoid reloading large weights per task.
_ASSESSOR: Any | None = None


def _get_assessor() -> Any:
    global _ASSESSOR
    if _ASSESSOR is None:
        from inference.claim_assessor import ClaimAssessor

        cfg_path = os.environ.get("CDP_INFERENCE_CONFIG", "configs/inference.yaml")
        _ASSESSOR = ClaimAssessor.from_config(cfg_path)
        logger.info("Celery worker loaded assessor from %s", cfg_path)
    return _ASSESSOR


@celery_app.task(
    bind=True,
    name="assess_images",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
)
def assess_images_task(
    self, b64_images: list[str], image_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Assess a batch of base64-encoded images. Returns a list of report dicts."""
    try:
        assessor = _get_assessor()
        raw = [base64.b64decode(b) for b in b64_images]
        reports = assessor.assess_batch(raw, image_ids=image_ids)
        return reports
    except MaxRetriesExceededError:
        logger.exception("Max retries exceeded for assess_images_task")
        raise
    except Exception as exc:
        logger.exception("assess_images_task failed: %s", exc)
        raise self.retry(exc=exc) from exc
