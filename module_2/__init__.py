"""Module 2: Landing Site Seeding."""
from .terrain_analysis import (
    compute_slope,
    load_auxiliary_rasters,
    make_synthetic_auxiliary,
    compute_surface_roughness,
    detect_boulders,
    make_synthetic_ohrc,
)
from .mcda import build_candidate_mask, extract_candidate_sites, CandidateSite, compute_exposure_score

__all__ = [
    "compute_slope",
    "load_auxiliary_rasters",
    "make_synthetic_auxiliary",
    "compute_surface_roughness",
    "detect_boulders",
    "make_synthetic_ohrc",
    "build_candidate_mask",
    "extract_candidate_sites",
    "CandidateSite",
    "compute_exposure_score",
]
