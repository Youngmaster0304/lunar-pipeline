"""Environment-based configuration using pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Lunar Pipeline Dashboard"
    app_version: str = "3.0.0"
    debug: bool = False

    # Database
    database_url: str = "sqlite:///./web/pipeline.db"
    database_echo: bool = False

    # CORS
    cors_origins: List[str] = ["*"]

    # Pipeline
    figures_dir: str = str(Path(__file__).resolve().parent.parent / "figures")
    runs_dir: str = str(Path(__file__).resolve().parent / "runs")

    # Real GeoTIFF data paths (empty = use synthetic fixtures)
    dfsar_tile_path: str = ""
    dem_path: str = ""
    dte_path: str = ""
    illumination_path: str = ""
    ohrc_path: str = ""

    # Celery / async workers
    celery_broker_url: str = ""
    celery_result_backend: str = ""

    # Redis (for cross-process event bus; empty = in-memory only)
    redis_url: str = ""

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    model_config = {"env_prefix": "LUNAR_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
