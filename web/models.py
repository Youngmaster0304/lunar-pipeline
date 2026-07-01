"""SQLAlchemy models for pipeline run persistence."""
from __future__ import annotations

import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(20), default="running")  # running / success / failed
    dsc_name = Column(String(100), default="Crater F2")
    ice_volume_m3 = Column(Float, default=0.0)
    n_candidates = Column(Integer, default=0)
    n_dsc_craters = Column(Integer, default=0)
    dash_feasible = Column(Boolean, default=False)
    slam_mae = Column(Float, default=0.0)
    efpi_ice_pct = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    duration_s = Column(Float, default=0.0)

    stages = relationship("StageResult", back_populates="run", cascade="all, delete-orphan")
    figures = relationship("FigureRecord", back_populates="run", cascade="all, delete-orphan")


class StageResult(Base):
    __tablename__ = "stage_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False)
    stage_index = Column(Integer, nullable=False)
    stage_name = Column(String(100), nullable=False)
    status = Column(String(20), default="pending")  # pending / running / success / failed
    log_output = Column(Text, nullable=True)

    run = relationship("PipelineRun", back_populates="stages")


class FigureRecord(Base):
    __tablename__ = "figures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False)
    filename = Column(String(100), nullable=False)
    title = Column(String(200), nullable=True)

    run = relationship("PipelineRun", back_populates="figures")
