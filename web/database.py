"""Database setup — supports both SQLite and PostgreSQL via env."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


def _build_engine():
    url = settings.database_url
    if url.startswith("sqlite"):
        return create_engine(url, echo=settings.database_echo, connect_args={"check_same_thread": False})
    return create_engine(url, echo=settings.database_echo, pool_pre_ping=True)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from .models import PipelineRun, StageResult, FigureRecord  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
