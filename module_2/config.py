"""Configuration dataclass for Module 2: Landing Site Seeding.

All MCDA weights, terrain thresholds, and peak-detection parameters are
centralised here so that no magic numbers appear in the algorithmic code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Module2Config:
    """Top-level configuration for the landing-site seeding pipeline.

    Attributes
    ----------
    max_slope_deg : float
        Maximum allowable terrain slope for a candidate landing site.
        Units: degrees.  Range: (0, 90).  Default 10° is consistent with
        Chandrayaan-3 and ISRO lander engineering constraints.
    illumination_threshold : float
        Minimum required solar illumination persistence as a fraction of the
        total observation / exploration window.
        Dimensionless, range [0, 1].  Default 0.70 means ≥ 70 % of the time
        the pixel must be sunlit.
    exposure_weight_illumination : float
        Weight applied to the normalised illumination layer in the weighted
        exposure score.  Must satisfy:
        ``exposure_weight_illumination + exposure_weight_dte == 1.0``.
        Dimensionless.
    exposure_weight_dte : float
        Weight applied to the DTE (Direct-To-Earth) boolean layer.
        Must satisfy:
        ``exposure_weight_illumination + exposure_weight_dte == 1.0``.
        Dimensionless.
    peak_min_distance : int
        Minimum separation between candidate peaks in pixels.  Used to
        enforce spatial diversity among landing site proposals.
        Units: pixels (grid cells).
    peak_neighborhood_size : int
        Window side-length for the ``scipy.ndimage.maximum_filter`` used to
        detect local maxima in the exposure score.  Must be an odd positive
        integer.  Units: pixels.
    max_candidates : int
        Maximum number of candidate sites returned.  Sites are ranked by
        exposure score in descending order before truncation.

    OHRC / Surface Analysis Attributes
    -----------------------------------
    roughness_window : int
        Window size for local surface roughness computation.  Must be odd.
    max_roughness_m : float
        Maximum allowable surface roughness for a candidate landing site.
        Units: metres (local elevation standard deviation).
    boulder_prominence_m : float
        Minimum elevation above local background to classify as a boulder.
        Units: metres.
    boulder_buffer_px : int
        Exclusion buffer radius around each detected boulder (pixels).
    max_boulder_count : int
        Maximum number of boulders allowed within a candidate site
        neighbourhood for it to be considered safe.
    roughness_weight : float
        Weight for surface roughness in the exposure score (higher =
        smoother terrain preferred).
    ohrc_roughness_window : int
        Window size for OHRC-derived high-resolution roughness.
    min_consecutive_sunlit_days : int
        Minimum required duration of uninterrupted direct sunlight for a
        landing site.  A lunar day is ~28 Earth days, so 10 consecutive
        Earth days of sunlight are required for the lander to operate as
        a relay and charging base before the rover enters the DSC.
        Default: 10 days.
    """

    max_slope_deg: float = 10.0
    illumination_threshold: float = 0.70
    exposure_weight_illumination: float = 0.60
    exposure_weight_dte: float = 0.40
    peak_min_distance: int = 5
    peak_neighborhood_size: int = 11
    max_candidates: int = 10

    # OHRC / surface analysis
    roughness_window: int = 5
    max_roughness_m: float = 3.0
    boulder_prominence_m: float = 2.0
    boulder_buffer_px: int = 2
    max_boulder_count: int = 3
    roughness_weight: float = 0.15
    ohrc_roughness_window: int = 7

    # Safe landing: 3 strict criteria
    min_consecutive_sunlit_days: int = 10

    def __post_init__(self) -> None:
        """Validate configuration values after initialisation."""
        if not 0.0 < self.max_slope_deg < 90.0:
            raise ValueError(
                f"max_slope_deg must be in (0, 90); got {self.max_slope_deg}"
            )
        if not 0.0 <= self.illumination_threshold <= 1.0:
            raise ValueError(
                f"illumination_threshold must be in [0, 1]; got {self.illumination_threshold}"
            )
        weight_sum = self.exposure_weight_illumination + self.exposure_weight_dte
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Exposure weights must sum to 1.0; got {weight_sum:.6f}"
            )
        if self.peak_neighborhood_size % 2 == 0:
            raise ValueError(
                f"peak_neighborhood_size must be odd; got {self.peak_neighborhood_size}"
            )
        if self.max_candidates <= 0:
            raise ValueError(
                f"max_candidates must be positive; got {self.max_candidates}"
            )
        if self.roughness_window < 3 or self.roughness_window % 2 == 0:
            raise ValueError(
                f"roughness_window must be odd and ≥ 3; got {self.roughness_window}"
            )
        if self.max_roughness_m <= 0:
            raise ValueError(
                f"max_roughness_m must be > 0; got {self.max_roughness_m}"
            )
        if not 0.0 <= self.roughness_weight <= 1.0:
            raise ValueError(
                f"roughness_weight must be in [0, 1]; got {self.roughness_weight}"
            )
        if self.min_consecutive_sunlit_days <= 0:
            raise ValueError(
                f"min_consecutive_sunlit_days must be > 0; got {self.min_consecutive_sunlit_days}"
            )
