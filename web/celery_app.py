from __future__ import annotations

import logging

from celery import Celery

from .config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "lunar_pipeline",
    broker=settings.celery_broker_url or "redis://localhost:6379/0",
    backend=settings.celery_result_backend or "redis://localhost:6379/0",
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_ignore_result=True,
)

logger.info(
    "Celery app '%s' configured (broker=%s)",
    celery_app.main,
    celery_app.conf.broker_url,
)
