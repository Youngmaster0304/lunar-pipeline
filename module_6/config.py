"""Configuration dataclass for Module 6: PSR and Doubly-Shadowed Crater Detection.

All horizon, crater-detection, and illumination parameters are centralised here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Module6Config:
    """Configuration for the PSR detection and doubly-shadowed crater identification pipeline.

    Attributes
    ----------
    solar_max_elevation_deg : float
        Maximum solar elevation angle at the lunar south pole during a precession
        cycle.  Faustini crater region: ~1.5°.  Units: degrees.
    psr_illumination_threshold : float
        Maximum illumination persistence fraction below which a pixel is classified
        as permanently shadowed.  Dimensionless, range [0, 1].
    crater_min_depth_m : float
        Minimum topographic depth (relative to local mean) for a depression to be
        classified as a crater.  Units: metres.
    crater_min_area_pixels : int
        Minimum number of connected pixels for a valid crater.
    horizon_search_radius : int
        Maximum pixel distance for horizon line-of-sight computation.
    """
    solar_max_elevation_deg: float = 1.5
    psr_illumination_threshold: float = 0.05
    crater_min_depth_m: float = 50.0
    crater_min_area_pixels: int = 9
    horizon_search_radius: int = 50

    def __post_init__(self) -> None:
        if not 0.0 <= self.solar_max_elevation_deg <= 90.0:
            raise ValueError(
                f"solar_max_elevation_deg must be in [0, 90]; got {self.solar_max_elevation_deg}"
            )
        if not 0.0 <= self.psr_illumination_threshold <= 1.0:
            raise ValueError(
                f"psr_illumination_threshold must be in [0, 1]; got {self.psr_illumination_threshold}"
            )
        if self.crater_min_depth_m <= 0:
            raise ValueError(f"crater_min_depth_m must be > 0; got {self.crater_min_depth_m}")
        if self.crater_min_area_pixels <= 0:
            raise ValueError(f"crater_min_area_pixels must be > 0; got {self.crater_min_area_pixels}")
