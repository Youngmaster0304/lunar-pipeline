"""Multi-Criteria Decision Analysis (MCDA) for lunar landing site seeding.

This module provides the decision layer of the landing-site seeding pipeline:

1. ``build_candidate_mask``   — Hard constraints (slope, DTE, illumination).
2. ``compute_exposure_score`` — Weighted continuous score for site ranking.
3. ``extract_candidate_sites``— Local-maxima extraction and site ranking.
4. ``CandidateSite``          — Dataclass representing one candidate site.

Scoring equation (exposure score)
-----------------------------------
  S = w_ill · f_ill + w_dte · dte_bool
where:
  w_ill = config.exposure_weight_illumination
  w_dte = config.exposure_weight_dte
  f_ill = illumination persistence fraction ∈ [0, 1]
  dte_bool = DTE binary ∈ {0, 1}
  w_ill + w_dte = 1.0  (enforced by Module2Config.__post_init__)

Local-maxima detection uses a morphological maximum filter of window size
``config.peak_neighborhood_size``.  A pixel is a local maximum if its
exposure score equals the maximum in its neighbourhood.  Candidate pixels
that are not in the ``candidate_mask`` are excluded before peak detection
by setting their score to −∞.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from .config import Module2Config

logger = logging.getLogger(__name__)


@dataclass
class CandidateSite:
    """Represents a single candidate lunar landing site.

    Attributes
    ----------
    row : int
        Row index (0-based) in the input raster grid.
    col : int
        Column index (0-based) in the input raster grid.
    slope_deg : float
        Terrain slope at this site in degrees.  Range: [0°, 90°).
    dte_ok : bool
        Whether this site has Direct-To-Earth line-of-sight.
    illumination_fraction : float
        Solar illumination persistence fraction.  Range: [0, 1].
    exposure_score : float
        Weighted exposure score = w_ill · illumination + w_dte · dte.
        Range: [0, 1].
    rank : int
        Rank of this site among all candidates, starting from 1 (best).
    """

    row: int
    col: int
    slope_deg: float
    dte_ok: bool
    illumination_fraction: float
    exposure_score: float
    rank: int


def build_candidate_mask(
    slope: np.ndarray,
    dte: np.ndarray,
    illumination: np.ndarray,
    config: Module2Config,
    roughness: np.ndarray | None = None,
    boulder_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply hard constraints to produce a boolean candidate-site mask.

    A pixel is eligible as a candidate landing site if and only if ALL
    three of the following conditions are satisfied simultaneously:

    1. slope < config.max_slope_deg
       (terrain gentle enough for safe landing)
    2. dte == True
       (direct Earth communications link available)
    3. illumination >= config.illumination_threshold
       (sufficient solar power for the exploration window)

    Parameters
    ----------
    slope : np.ndarray, shape (rows, cols), dtype float32
        Terrain slope in degrees.  Output of ``compute_slope``.
    dte : np.ndarray, shape (rows, cols), dtype bool
        Direct-To-Earth boolean raster.
    illumination : np.ndarray, shape (rows, cols), dtype float32
        Illumination persistence fraction ∈ [0, 1].
    config : Module2Config
        Pipeline configuration with threshold parameters.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype bool
        True where all three conditions are met.

    Raises
    ------
    ValueError
        If not all input arrays have the same shape.
    """
    arrays = {"slope": slope, "dte": dte, "illumination": illumination}
    if roughness is not None:
        arrays["roughness"] = roughness
    if boulder_mask is not None:
        arrays["boulder_mask"] = boulder_mask

    shapes = {arr.shape for arr in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(
            f"All input arrays must have the same shape; got {shapes}."
        )

    condition_slope  = slope < config.max_slope_deg
    condition_dte    = dte.astype(bool)
    condition_illum  = illumination >= config.illumination_threshold

    candidate_mask: np.ndarray = condition_slope & condition_dte & condition_illum

    # OHRC-derived constraints (optional)
    if roughness is not None:
        condition_roughness = roughness < config.max_roughness_m
        candidate_mask &= condition_roughness
        n_rejected_rough = int(np.sum(~condition_roughness & (condition_slope & condition_dte & condition_illum)))
        logger.info("Roughness constraint: %d pixels rejected (max=%.2f m)", n_rejected_rough, config.max_roughness_m)

    if boulder_mask is not None:
        # Expand boulder mask by buffer radius
        from scipy.ndimage import binary_dilation
        struct = np.ones((config.boulder_buffer_px * 2 + 1,) * 2, dtype=bool)
        boulder_exclusion = binary_dilation(boulder_mask, structure=struct)
        candidate_mask &= ~boulder_exclusion
        n_rejected_boulders = int(np.sum(boulder_exclusion & (condition_slope & condition_dte & condition_illum)))
        logger.info("Boulder exclusion: %d pixels rejected (buffer=%d px)", n_rejected_boulders, config.boulder_buffer_px)

    n_candidates = int(np.sum(candidate_mask))
    logger.info(
        "Candidate mask built: %d / %d pixels pass constraints (%.2f %%)",
        n_candidates,
        candidate_mask.size,
        100.0 * n_candidates / candidate_mask.size,
    )
    return candidate_mask


def compute_exposure_score(
    illumination: np.ndarray,
    dte: np.ndarray,
    config: Module2Config,
    roughness: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the weighted exposure score for each pixel.

    Equation
    --------
    S(r, c) = w_ill · illumination(r, c) + w_dte · dte(r, c)

    where:
      w_ill = config.exposure_weight_illumination   (default 0.60)
      w_dte = config.exposure_weight_dte            (default 0.40)
      w_ill + w_dte = 1.0  (enforced by Module2Config)

    Parameters
    ----------
    illumination : np.ndarray, shape (rows, cols), dtype float32
        Illumination persistence fraction ∈ [0, 1].
    dte : np.ndarray, shape (rows, cols), dtype bool or float32
        Direct-To-Earth boolean raster.  Converted to float internally.
    config : Module2Config
        Pipeline configuration with exposure weights.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype float32
        Exposure score ∈ [0, 1].
    """
    score: np.ndarray = (
        (1.0 - config.roughness_weight) * (
            config.exposure_weight_illumination * illumination.astype(np.float32)
            + config.exposure_weight_dte * dte.astype(np.float32)
        )
    )

    # Roughness penalty: smoother terrain scores higher
    if roughness is not None:
        roughness_norm = roughness / (config.max_roughness_m + 1e-10)
        roughness_penalty = config.roughness_weight * (1.0 - np.clip(roughness_norm, 0.0, 1.0))
        score += roughness_penalty

    logger.debug(
        "Exposure score: min=%.4f max=%.4f mean=%.4f (roughness_weight=%.2f)",
        float(np.min(score)),
        float(np.max(score)),
        float(np.mean(score)),
        config.roughness_weight,
    )
    return score.astype(np.float32)


def extract_candidate_sites(
    candidate_mask: np.ndarray,
    exposure_score: np.ndarray,
    slope: np.ndarray,
    dte: np.ndarray,
    illumination: np.ndarray,
    config: Module2Config,
    roughness: np.ndarray | None = None,
) -> list[CandidateSite]:
    """Extract ranked candidate landing sites from the exposure score.

    Algorithm
    ---------
    1. Mask exposure_score:
         masked_score = exposure_score.copy()
         masked_score[~candidate_mask] = −∞
       This ensures non-candidate pixels cannot be local maxima.

    2. Local-maxima detection via morphological maximum filter:
         filtered = maximum_filter(masked_score,
                                   size=config.peak_neighborhood_size)
         is_peak = (masked_score == filtered) & candidate_mask

    3. Collect peak positions, sort by exposure_score descending,
       take at most config.max_candidates.

    4. Build and return a list of ``CandidateSite`` objects sorted by rank
       (rank 1 = highest exposure score).

    Parameters
    ----------
    candidate_mask : np.ndarray, shape (rows, cols), dtype bool
        Output of ``build_candidate_mask``.
    exposure_score : np.ndarray, shape (rows, cols), dtype float32
        Output of ``compute_exposure_score``.
    slope : np.ndarray, shape (rows, cols), dtype float32
        Terrain slope in degrees.
    dte : np.ndarray, shape (rows, cols), dtype bool
        Direct-To-Earth boolean raster.
    illumination : np.ndarray, shape (rows, cols), dtype float32
        Illumination persistence fraction.
    config : Module2Config
        Pipeline configuration.

    Returns
    -------
    list[CandidateSite]
        Sorted list of candidate sites (rank 1 = best), length ≤
        config.max_candidates.  Returns an empty list if no candidates pass.

    Raises
    ------
    ValueError
        If input arrays have inconsistent shapes.
    """
    arrays = {
        "candidate_mask": candidate_mask,
        "exposure_score": exposure_score,
        "slope": slope,
        "dte": dte,
        "illumination": illumination,
    }
    if roughness is not None:
        arrays["roughness"] = roughness
    shapes = {name: arr.shape for name, arr in arrays.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(
            f"All input arrays must have the same shape; got: {shapes}"
        )

    # Step 1: mask out non-candidates
    masked_score = exposure_score.astype(np.float64).copy()
    masked_score[~candidate_mask] = -np.inf

    # Early exit if no candidates at all
    if not np.any(candidate_mask):
        logger.info("No candidate pixels found; returning empty site list.")
        return []

    # Step 2: morphological local-maxima detection
    filtered = maximum_filter(
        masked_score,
        size=config.peak_neighborhood_size,
        mode="reflect",
    )
    is_peak: np.ndarray = (masked_score == filtered) & candidate_mask

    peak_rows, peak_cols = np.where(is_peak)
    if peak_rows.size == 0:
        logger.info("No local-maxima peaks found; returning empty site list.")
        return []

    # Step 3: sort peaks by exposure_score descending
    peak_scores = exposure_score[peak_rows, peak_cols].astype(np.float64)
    order = np.argsort(-peak_scores)  # descending
    top_k = min(config.max_candidates, len(order))
    order = order[:top_k]

    # Step 4: build CandidateSite objects
    sites: list[CandidateSite] = []
    for rank, idx in enumerate(order, start=1):
        r = int(peak_rows[idx])
        c = int(peak_cols[idx])
        sites.append(
            CandidateSite(
                row=r,
                col=c,
                slope_deg=float(slope[r, c]),
                dte_ok=bool(dte[r, c]),
                illumination_fraction=float(illumination[r, c]),
                exposure_score=float(exposure_score[r, c]),
                rank=rank,
            )
        )

    logger.info(
        "Extracted %d candidate sites (max_candidates=%d).",
        len(sites),
        config.max_candidates,
    )
    for site in sites:
        logger.debug(
            "  Rank %d: (row=%d, col=%d) slope=%.2f° illum=%.3f score=%.4f",
            site.rank,
            site.row,
            site.col,
            site.slope_deg,
            site.illumination_fraction,
            site.exposure_score,
        )

    return sites
