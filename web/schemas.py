"""Pydantic schemas for API responses."""
from __future__ import annotations

from pydantic import BaseModel
from typing import List, Optional


class StageResultOut(BaseModel):
    stage_index: int
    stage_name: str
    status: str
    log_output: Optional[str] = None


class FigureOut(BaseModel):
    filename: str
    title: Optional[str] = None
    url: str


class GisFileOut(BaseModel):
    filename: str
    url: str
    format: str  # "GeoTIFF" or "GeoJSON"


class PipelineRunOut(BaseModel):
    id: int
    created_at: str
    status: str
    dsc_name: str
    ice_volume_m3: float
    n_candidates: int
    n_dsc_craters: int
    dash_feasible: bool
    slam_mae: float
    efpi_ice_pct: float
    error_message: Optional[str] = None
    duration_s: float
    stages: List[StageResultOut] = []
    figures: List[FigureOut] = []
    gis_files: List[GisFileOut] = []


class PipelineRunSummary(BaseModel):
    id: int
    created_at: str
    status: str
    dsc_name: str
    ice_volume_m3: float
    dash_feasible: bool
    duration_s: float
