"""FastAPI web app — production-grade lunar pipeline dashboard."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import shutil
import time
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .config import settings
from .database import get_session, init_db
from .events import get_event_bus
from .models import FigureRecord, PipelineRun, StageResult
from .runner import run_pipeline_internal
from .schemas import PipelineRunOut, PipelineRunSummary

# ── Structured logging setup ──
_log = logging.getLogger("web.main")
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ── App ──
app = FastAPI(title=settings.app_name, version=settings.app_version)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))

runs_dir = Path(settings.runs_dir)
runs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(runs_dir)), name="runs")


@app.on_event("startup")
def on_startup() -> None:
    _log.info("Starting %s v%s", settings.app_name, settings.app_version)
    bus = get_event_bus(settings)
    bus.set_loop(asyncio.get_event_loop())
    init_db()
    _log.info(
        "Database initialised (url=%s) — event_bus=%s",
        settings.database_url,
        type(bus).__name__,
    )


# ── Frontend ──

@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_session)):
    runs = (
        db.query(PipelineRun)
        .order_by(PipelineRun.created_at.desc())
        .limit(50)
        .all()
    )
    runs_data = [
        {
            "id": r.id,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
            "status": r.status,
            "dsc_name": r.dsc_name,
            "ice_volume_m3": r.ice_volume_m3,
            "dash_feasible": r.dash_feasible,
            "duration_s": r.duration_s,
        }
        for r in runs
    ]
    return templates.TemplateResponse(request, "index.html", {"runs": runs_data})


# ── API — Pipeline runs ──

@app.post("/api/runs")
def create_run(db: Session = Depends(get_session)):
    run = PipelineRun(status="running")
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id

    # Dispatch to Celery worker if broker configured, else local thread
    if settings.celery_broker_url:
        from .tasks import run_pipeline_task
        run_pipeline_task.delay(run_id)
        _log.info("Run #%d dispatched to Celery worker", run_id)
    else:
        import threading
        t = threading.Thread(target=_run_pipeline_wrapper, args=(run_id,), daemon=True)
        t.start()
        _log.info("Run #%d dispatched via local daemon thread", run_id)

    return {"id": run_id, "status": "running"}


def _run_pipeline_wrapper(run_id: int) -> None:
    """Create a fresh session and run the pipeline locally."""
    from .database import SessionLocal as _SessionLocal
    bus = get_event_bus(settings)

    db = _SessionLocal()
    try:
        run_pipeline_internal(db, run_id, event_bus=bus)
        _log.info("Run #%d completed", run_id)
    except Exception:
        _log.exception("Run #%d failed unexpectedly", run_id)
    finally:
        db.close()


@app.get("/api/runs", response_model=List[PipelineRunSummary])
def list_runs(db: Session = Depends(get_session)):
    runs = (
        db.query(PipelineRun)
        .order_by(PipelineRun.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        PipelineRunSummary(
            id=r.id,
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
            status=r.status,
            dsc_name=r.dsc_name,
            ice_volume_m3=r.ice_volume_m3,
            dash_feasible=r.dash_feasible,
            duration_s=r.duration_s,
        )
        for r in runs
    ]


@app.get("/api/runs/{run_id}", response_model=PipelineRunOut)
def get_run(run_id: int, db: Session = Depends(get_session)):
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_out(run, db)


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: int, db: Session = Depends(get_session)):
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    db.query(StageResult).filter(StageResult.run_id == run_id).delete()
    db.query(FigureRecord).filter(FigureRecord.run_id == run_id).delete()
    db.delete(run)
    db.commit()
    run_dir = runs_dir / str(run_id)
    if run_dir.exists():
        shutil.rmtree(str(run_dir))
    _log.info("Run #%d deleted", run_id)
    return {"status": "deleted", "id": run_id}


# ── Downloads ──

@app.get("/api/runs/{run_id}/download/json")
def download_json(run_id: int, db: Session = Depends(get_session)):
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    out = _run_to_out(run, db)
    return Response(
        content=out.model_dump_json(indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename=run_{run_id}.json'},
    )


@app.get("/api/runs/{run_id}/download/csv")
def download_csv(run_id: int, db: Session = Depends(get_session)):
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    out = _run_to_out(run, db)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["field", "value"])
    for field, val in out.model_dump().items():
        w.writerow([field, str(val)])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=run_{run_id}.csv'},
    )


@app.get("/api/runs/{run_id}/figures/{filename}")
def serve_figure(run_id: int, filename: str):
    fp = runs_dir / str(run_id) / filename
    if not fp.exists() or fp.suffix != ".png":
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(str(fp), media_type="image/png")


# ── WebSocket live updates ──

@app.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: int):
    await websocket.accept()
    bus = get_event_bus(settings)
    if asyncio.iscoroutinefunction(bus.subscribe):
        q = await bus.subscribe(run_id)
    else:
        q = bus.subscribe(run_id)
    try:
        while True:
            msg = await asyncio.wait_for(q.get(), timeout=30)
            await websocket.send_json(msg)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        pass
    finally:
        if asyncio.iscoroutinefunction(bus.unsubscribe):
            await bus.unsubscribe(run_id, q)
        else:
            bus.unsubscribe(run_id, q)


# ── Health ──

@app.get("/api/health")
def health():
    return {"status": "ok", "version": settings.app_version, "uptime": time.time() - _startup_time}

_startup_time = time.time()


# ── Helpers ──

def _run_to_out(run: PipelineRun, db: Session) -> PipelineRunOut:
    stages = db.query(StageResult).filter(StageResult.run_id == run.id).order_by(StageResult.stage_index).all()
    figures = db.query(FigureRecord).filter(FigureRecord.run_id == run.id).all()

    gis_files = []
    run_dir = runs_dir / str(run.id)
    if run_dir.exists():
        for f in sorted(run_dir.iterdir()):
            if f.suffix == ".geojson":
                gis_files.append({"filename": f.name, "url": f"/runs/{run.id}/{f.name}", "format": "GeoJSON"})
            elif f.suffix == ".tif":
                gis_files.append({"filename": f.name, "url": f"/runs/{run.id}/{f.name}", "format": "GeoTIFF"})

    return PipelineRunOut(
        id=run.id,
        created_at=run.created_at.strftime("%Y-%m-%d %H:%M:%S") if run.created_at else "",
        status=run.status,
        dsc_name=run.dsc_name,
        ice_volume_m3=run.ice_volume_m3,
        n_candidates=run.n_candidates,
        n_dsc_craters=run.n_dsc_craters,
        dash_feasible=run.dash_feasible,
        slam_mae=run.slam_mae,
        efpi_ice_pct=run.efpi_ice_pct,
        error_message=run.error_message,
        duration_s=run.duration_s,
        stages=[
            {
                "stage_index": s.stage_index,
                "stage_name": s.stage_name,
                "status": s.status,
                "log_output": s.log_output,
            }
            for s in stages
        ],
        figures=[
            {
                "filename": f.filename,
                "title": f.title or f.filename,
                "url": f"/runs/{run.id}/{f.filename}",
            }
            for f in figures
        ],
        gis_files=gis_files,
    )
