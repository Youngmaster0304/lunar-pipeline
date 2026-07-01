#!/usr/bin/env bash
set -e

echo "[start] LUNAR_PIPELINE starting — PORT=${PORT:-8000} WORKERS=${UVICORN_WORKERS:-2}"

# DB tables created automatically by app startup (on_startup calls init_db).
# Run Alembic migrations if they exist and are preferred:
# alembic upgrade head

exec uvicorn web.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers "${UVICORN_WORKERS:-2}" \
    --proxy-headers \
    --forwarded-allow-ips '*'
