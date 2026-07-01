#!/usr/bin/env bash
set -e

echo "[start-worker] Celery worker starting — CONCURRENCY=${CELERY_CONCURRENCY:-2}"

exec celery -A web.celery_app worker \
    --loglevel=info \
    --concurrency="${CELERY_CONCURRENCY:-2}" \
    --max-tasks-per-child=10
