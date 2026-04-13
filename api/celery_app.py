"""Celery application for async batch assessment.

Broker/backend default to Redis at redis://redis:6379/0 (docker-compose) and
can be overridden via env vars CDP_CELERY_BROKER and CDP_CELERY_BACKEND.
"""

from __future__ import annotations

import os

from celery import Celery

BROKER = os.environ.get("CDP_CELERY_BROKER", "redis://redis:6379/0")
BACKEND = os.environ.get("CDP_CELERY_BACKEND", "redis://redis:6379/1")

celery_app = Celery(
    "car_damage_pipeline",
    broker=BROKER,
    backend=BACKEND,
    include=["api.tasks"],
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=600,
    task_soft_time_limit=540,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
)
