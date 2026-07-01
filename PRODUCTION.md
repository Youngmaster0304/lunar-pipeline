# Production-Grade Deployment Proposal

## Current Architecture

```
Browser → FastAPI (sync) → SQLite → pipeline modules (inline)
               ↓
          static/figures per run
```

**Weaknesses:**
- Pipeline blocks the HTTP worker — long request, no concurrency
- SQLite single-writer — crashes under concurrent runs
- No auth, no rate limiting, no CORS policy
- No background task queue — if process restarts mid-run, work is lost
- Figures stored on local disk — no backup, no CDN
- Single-process, no horizontal scaling

---

## Proposed Architecture

```
Browser → Nginx (HTTPS + static) → FastAPI (async) → Celery → Redis → Pipeline workers
               ↓                                            ↓
          auth gateway (oauth2-proxy)                  PostgreSQL
```

### Tier 1 — Minimal Safe (1–2 days)

| Area | Change | Why |
|------|--------|-----|
| **Async pipeline** | Move `run_pipeline_internal` to `FastAPI.BackgroundTasks`; POST returns `{"id": ..}` immediately | Avoids HTTP timeout for long-running pipelines |
| **PostgreSQL** | Replace SQLite with PostgreSQL via SQLAlchemy | Concurrent writes, point-in-time recovery |
| **Migrations** | Add Alembic for schema versioning | Safe schema evolution |
| **Config** | `.env` file + `pydantic-settings` for DB URL, secrets, paths | No hardcoded paths |
| **Logging** | `structlog` to JSON file + PostgreSQL `logs` table | Structured debugging |
| **Docker** | `Dockerfile` for the FastAPI app | Reproducible deploys |
| **CORS** | `fastapi.middleware.cors` restrict to known origins | Security |

### Tier 2 — Concurrent Pipeline (1–2 weeks)

| Area | Change | Why |
|------|--------|-----|
| **Celery + Redis** | Pipeline runs asynchronously via Celery tasks; status stored in Redis during execution | Multiple runs in parallel, no HTTP blocking |
| **WebSocket** | `/ws/runs/{id}` pushes stage-complete events | Real-time frontend progress without polling |
| **Task queue** | Celery `task_acks_late` + result backend (Redis) | Survives worker crash mid-run |
| **Worker pool** | 2–4 Celery workers, auto-scaled | Throughput |
| **Rate limit** | `slowapi` on POST `/api/runs` (e.g. 5/min per IP) | Prevents runaway submissions |

### Tier 3 — Production Hardening (3–4 weeks)

| Area | Change | Why |
|------|--------|-----|
| **Auth** | oauth2-proxy or Auth0 in front of FastAPI | Role-based access (dev vs operator) |
| **Reverse proxy** | Nginx: HTTPS termination, static file serve, gzip, buffer limits | Performance + security |
| **Object storage** | Figures → S3/MinIO; served via signed URLs or CDN | Durable, scalable, decoupled from app server |
| **Backup** | Daily `pg_dump` to S3 + hourly WAL archiving | Disaster recovery |
| **Metrics** | Prometheus endpoint (`/metrics`) + Grafana dashboard | Uptime, run duration, failure rate, queue depth |
| **Sentry** | `sentry-sdk` for error tracking | Catch production issues before users report them |
| **CI/CD** | GitHub Actions: pytest → build → docker push → deploy | Zero-touch releases |

---

## Specific Code Changes for Each Tier

### Tier 1 — Minimum Viable

```python
# web/main.py — BackgroundTasks pattern
from fastapi import BackgroundTasks

@app.post("/api/runs")
def create_run(background_tasks: BackgroundTasks, db: Session = Depends(get_session)):
    run = PipelineRun(status="running")
    db.add(run); db.commit(); db.refresh(run)
    background_tasks.add_task(run_pipeline_internal, db, run.id)
    return {"id": run.id, "status": "running"}
```

```dockerfile
# Dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: .
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql+psycopg2://user:pass@db/lunar
  db:
    image: postgres:16
    volumes: ["pgdata:/var/lib/postgresql/data"]
volumes: { pgdata: {} }
```

### Tier 2 — Concurrent

```python
# celery_app.py
from celery import Celery
celery_app = Celery("lunar", broker="redis://redis:6379/0")

@celery_app.task(acks_late=True, max_retries=3)
def run_pipeline_task(run_id: int):
    from web.database import SessionLocal
    db = SessionLocal()
    try:
        run_pipeline_internal(db, run_id)
    finally:
        db.close()
```

### Tier 3 — Hardened

```
# nginx.conf snippet
location /runs/ {
    alias /data/runs/;
    expires 30d;
    add_header Cache-Control "public, immutable";
}
location /api/ {
    proxy_pass http://fastapi:8000;
    limit_req zone=api burst=10;
}
```

---

## What to Ship Now (Zero-Cost)

Already done in this refactor:
- [x] `BackgroundTasks`-compatible `runner.py` (non-blocking if caller uses background tasks)
- [x] Health endpoint `GET /api/health`
- [x] DELETE endpoint with cascade cleanup
- [x] `.env`-ready config (use `os.environ.get` in `config.py`)
- [x] `pagination`, `search`, `sort` on the frontend
- [x] `lightbox` figure viewer
- [x] `keyboard shortcuts` (r=run, esc=close)
- [x] `toast notifications` replacing raw alerts
- [x] `skeleton loading` for summary cards

Run the current version:
```powershell
cd lunar_pipeline
python -m uvicorn web.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

First thing to address before production: **background task execution** (Tier 1 item 1).
