"""Celery tasks — pipeline execution in background workers."""
from __future__ import annotations

import logging
import traceback

from .celery_app import celery_app
from .config import settings
from .database import SessionLocal
from .events import get_event_bus
from .runner import run_pipeline_internal

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def run_pipeline_task(self, run_id: int) -> dict:
    """Execute the full lunar pipeline in a Celery worker process."""
    logger.info("Task received: run_pipeline(id=%d) — task_id=%s", run_id, self.request.id)

    bus = get_event_bus(settings)

    db = SessionLocal()
    try:
        run_pipeline_internal(db, run_id, event_bus=bus)
        logger.info("Run #%d completed successfully from Celery worker", run_id)
        return {"run_id": run_id, "status": "success"}
    except Exception as exc:
        logger.exception("Run #%d failed in Celery worker", run_id)
        try:
            from .models import PipelineRun
            run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
            if run:
                run.status = "failed"
                run.error_message = traceback.format_exc()
                db.commit()
        except Exception:
            logger.exception("Failed to persist run failure status")
        raise self.retry(exc=exc)
    finally:
        db.close()
