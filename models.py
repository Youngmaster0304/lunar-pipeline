import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from database import Base

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(50), default="running")
    dsc_name = Column(String(100), default="Crater F2")
    ice_volume_m3 = Column(Float, nullable=True)
    n_candidates = Column(Integer, nullable=True)
    dash_feasible = Column(Boolean, nullable=True)
    bug2_steps = Column(Integer, nullable=True)
    slam_mae = Column(Float, nullable=True)
    drill_heat_j = Column(Float, nullable=True)
    post_drill_temp_k = Column(Float, nullable=True)
    sublimation_kg_s = Column(Float, nullable=True)
    efpi_fringe_shift = Column(Float, nullable=True)
    ice_volume_pct = Column(Float, nullable=True)
    duration_s = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    image_ice_mask = Column(Text, nullable=True)
    image_slope_map = Column(Text, nullable=True)
    image_path_map = Column(Text, nullable=True)
