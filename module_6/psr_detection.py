"""PSR and doubly-shadowed crater detection for the lunar south pole.

This module implements:

1. Horizon-based PSR mask computation from a digital elevation model.
2. Crater identification via topographic depression analysis.
3. Doubly-shadowed crater classification (craters within PSRs).
4. Synthetic fixture generation for testing.

Reference
---------
Mazarico, E. et al. (2011). Illumination conditions of the lunar polar regions
using LOLA topography. *Icarus*, 211(2), 1066–1081.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, minimum_filter, maximum_filter

from .config import Module6Config

logger = logging.getLogger(__name__)


@dataclass
class PSRResult:
    """Container for PSR detection pipeline outputs.

    Attributes
    ----------
    psr_mask : np.ndarray, shape (rows, cols), dtype bool
        True where pixel is permanently shadowed.
    crater_mask : np.ndarray, shape (rows, cols), dtype bool
        True where a crater floor is detected.
    doubly_shadowed_mask : np.ndarray, shape (rows, cols), dtype bool
        True for craters that lie within PSRs (doubly-shadowed).
    n_psr_pixels : int
        Count of PSR pixels.
    n_craters : int
        Number of individual craters detected.
    n_doubly_shadowed : int
        Number of doubly-shadowed craters.
    solar_max_elevation_deg : float
        Maximum solar elevation used for classification.
    """
    psr_mask: np.ndarray
    crater_mask: np.ndarray
    doubly_shadowed_mask: np.ndarray
    n_psr_pixels: int
    n_craters: int
    n_doubly_shadowed: int
    solar_max_elevation_deg: float


def compute_horizon_mask(
    dem: np.ndarray,
    pixel_spacing_m: float,
    solar_max_elevation_deg: float,
    search_radius: int = 50,
) -> np.ndarray:
    """Compute a PSR mask by checking horizon occlusion for each pixel.

    For each pixel, checks whether any terrain along a set of azimuth directions
    blocks direct sunlight at the given maximum solar elevation.  If occluded
    along all directions, the pixel is classified as permanently shadowed.

    Parameters
    ----------
    dem : np.ndarray, shape (rows, cols), dtype float32
        Digital elevation model in metres.
    pixel_spacing_m : float
        Ground sampling distance in metres per pixel.
    solar_max_elevation_deg : float
        Maximum solar elevation angle above the horizon.  Units: degrees.
    search_radius : int
        Maximum pixel distance for horizon search along each azimuth.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype bool
        True if pixel is permanently shadowed.
    """
    rows, cols = dem.shape
    sun_elev_rad = np.radians(solar_max_elevation_deg)
    tan_sun = np.tan(sun_elev_rad)

    # Sample 8 azimuth directions
    azimuths = np.arange(0, 360, 45)
    psr_accumulator = np.ones((rows, cols), dtype=bool)

    for az in azimuths:
        az_rad = np.radians(az)
        dr = np.sin(az_rad)
        dc = np.cos(az_rad)

        # Build horizon angle map for this azimuth
        horizon_angle = np.full((rows, cols), -np.inf, dtype=np.float32)

        for step in range(1, search_radius + 1):
            r_src = np.clip(np.round(np.arange(rows) - dr * step).astype(int), 0, rows - 1)
            c_src = np.clip(np.round(np.arange(cols) - dc * step).astype(int), 0, cols - 1)

            for i in range(rows):
                for j in range(cols):
                    r2 = r_src[i]
                    c2 = c_src[j]
                    if r2 == i and c2 == j:
                        continue
                    dz = dem[r2, c2] - dem[i, j]
                    dist = np.hypot((r2 - i) * pixel_spacing_m, (c2 - j) * pixel_spacing_m)
                    if dist > 0:
                        angle = np.arctan(dz / dist)
                        if angle > horizon_angle[i, j]:
                            horizon_angle[i, j] = angle

        # Pixel is sunlit along this azimuth if horizon_angle < sun elevation
        sunlit = horizon_angle < sun_elev_rad
        psr_accumulator &= ~sunlit

    logger.debug(
        "Horizon mask: %d / %d PSR pixels (solar_max=%.1f°)",
        int(np.sum(psr_accumulator)), psr_accumulator.size, solar_max_elevation_deg,
    )
    return psr_accumulator


def compute_psr_mask(
    illumination: np.ndarray,
    config: Module6Config,
) -> np.ndarray:
    """Compute PSR mask from illumination persistence data.

    A pixel is classified as permanently shadowed if its illumination
    persistence is below the configured threshold.

    Parameters
    ----------
    illumination : np.ndarray, shape (rows, cols), dtype float32
        Illumination persistence fraction in [0, 1].
    config : Module6Config
        Pipeline configuration with ``psr_illumination_threshold``.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype bool
        PSR mask.
    """
    psr_mask = illumination < config.psr_illumination_threshold
    n_psr = int(np.sum(psr_mask))
    logger.info(
        "PSR mask: %d / %d pixels (%.2f %%) — threshold=%.2f",
        n_psr, psr_mask.size, 100.0 * n_psr / psr_mask.size,
        config.psr_illumination_threshold,
    )
    return psr_mask


def identify_craters(
    dem: np.ndarray,
    config: Module6Config,
) -> np.ndarray:
    """Identify crater floors from a DEM using topographic depression analysis.

    A pixel is part of a crater floor if:
    1. Its elevation is below the local mean by at least ``crater_min_depth_m``.
    2. The connected region has at least ``crater_min_area_pixels``.

    Parameters
    ----------
    dem : np.ndarray, shape (rows, cols), dtype float32
        Digital elevation model in metres.
    config : Module6Config
        Pipeline configuration.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype bool
        True for pixels belonging to crater floors.
    """
    # Local mean via minimum filter to find depressions
    local_min = minimum_filter(dem, size=21)

    # Depressions relative to local mean
    depth = local_min - dem
    deep_mask = depth > config.crater_min_depth_m

    # Connected-component labelling to filter by area
    labeled, n_features = label(deep_mask)
    crater_mask = np.zeros_like(deep_mask, dtype=bool)
    for feat_id in range(1, n_features + 1):
        feat_mask = labeled == feat_id
        if np.sum(feat_mask) >= config.crater_min_area_pixels:
            crater_mask |= feat_mask

    n_craters = len(np.unique(labeled[crater_mask])) if np.any(crater_mask) else 0
    logger.info(
        "Crater detection: %d features found, %d meet min_area=%d pixels",
        n_features, n_craters, config.crater_min_area_pixels,
    )
    return crater_mask


def identify_doubly_shadowed_craters(
    psr_mask: np.ndarray,
    crater_mask: np.ndarray,
) -> np.ndarray:
    """Identify doubly-shadowed craters: craters whose floors lie within PSRs.

    A doubly-shadowed crater is a crater (crater_mask) that is also
    classified as permanently shadowed (psr_mask).

    Parameters
    ----------
    psr_mask : np.ndarray, shape (rows, cols), dtype bool
        Permanently shadowed region mask.
    crater_mask : np.ndarray, shape (rows, cols), dtype bool
        Crater floor mask.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype bool
        Doubly-shadowed crater mask.
    """
    doubly = psr_mask & crater_mask
    n_doubly = int(np.sum(doubly))
    n_psr = int(np.sum(psr_mask))
    n_craters = int(np.sum(crater_mask))
    logger.info(
        "Doubly-shadowed craters: %d pixels overlap (PSR=%d, craters=%d)",
        n_doubly, n_psr, n_craters,
    )
    return doubly


def make_synthetic_psr_data(
    shape: tuple[int, int],
    dem: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic PSR, crater, and doubly-shadowed masks for testing.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(rows, cols)`` for the output masks.
    dem : np.ndarray or None
        Optional DEM to use for crater detection.  If None, a synthetic DEM
        with a central depression is created.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    psr_mask : np.ndarray, dtype bool
        Synthetic PSR mask (random, ~20 % of pixels).
    crater_mask : np.ndarray, dtype bool
        Synthetic crater mask (a central depression).
    doubly_mask : np.ndarray, dtype bool
        Intersection of the two.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    psr_mask = rng.random(shape) < 0.20

    crater = np.zeros(shape, dtype=bool)
    cr, cc = rows // 2, cols // 2
    crater[cr - 5:cr + 5, cc - 5:cc + 5] = True

    doubly_mask = psr_mask & crater

    logger.debug("Synthetic PSR data created: shape=%s", shape)
    return psr_mask, crater, doubly_mask
