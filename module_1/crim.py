"""Complex Refractive Index Model (CRIM) for volumetric water-ice retrieval.

The CRIM mixing rule expresses the bulk complex permittivity of a multi-
component mixture as a linear combination of the square roots of the
component permittivities, weighted by volumetric fractions:

  √ε_mix = Σᵢ fᵢ · √εᵢ                               … (1)

where fᵢ is the volumetric fraction of component i and Σᵢ fᵢ = 1.

Three-component model
---------------------
Components:
  1. Water ice       ε_ice = 3.15 + 0.001j  (Cumming 1952)
  2. Lunar regolith  ε_reg = 2.7  + 0.002j  (Olhoeft & Strangway 1975)
  3. Vacuum / pores  ε_vac = 1.0  + 0.0j    (free space)

Porosity constraint:
  f_vac = 0.40  (40 % vacuum; Carrier et al. 1991, lunar regolith porosity)
  f_ice + f_reg = 0.60   → f_reg = 0.60 − f_ice

Substituting into (1):
  √ε_mix(f_ice) = f_ice·√ε_ice + (0.60−f_ice)·√ε_reg + 0.40·√ε_vac   … (2)

Inversion
---------
Given a measured bulk permittivity ε_mix (proxy from backscatter; see
docstring of ``invert_ice_fraction`` for the empirical calibration):
  Solve for f_ice ∈ [0, 0.60] such that |√ε_mix(f_ice)|² = ε_mix_mag
  using scipy.optimize.brentq.

References
----------
Cumming, W. A. (1952). *The dielectric properties of ice and snow at 3.2
  centimeters.* Journal of Applied Physics, 23(7), 768–773.
Olhoeft, G. R., & Strangway, D. W. (1975). *Dielectric properties of the
  first 100 meters of the Moon.* Earth and Planetary Science Letters, 24(3),
  394–404.
Carrier, W. D., Olhoeft, G. R., & Mendell, W. (1991). Physical properties of
  the lunar surface. In *Lunar Sourcebook*, Cambridge University Press.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .config import Module1Config

logger = logging.getLogger(__name__)

# Small epsilon for numerical safety
_EPS: float = float(np.finfo(np.float64).eps)


def _crim_sqrt_eps(
    f_ice: float,
    eps_ice: complex,
    eps_reg: complex,
    eps_vac: complex,
    solid_fraction: float,
    vac_fraction: float,
) -> complex:
    """Evaluate √ε_mix for a given ice volumetric fraction.

    Implements equation (2) of the module docstring.

    Parameters
    ----------
    f_ice : float
        Volumetric fraction of water ice.  Range: [0, solid_fraction].
    eps_ice : complex
        Complex relative permittivity of water ice.
    eps_reg : complex
        Complex relative permittivity of dry lunar regolith.
    eps_vac : complex
        Complex relative permittivity of vacuum pore space.
    solid_fraction : float
        Total solid filling fraction = 1 − porosity.  (i.e. f_ice + f_reg).
    vac_fraction : float
        Volumetric fraction of vacuum pore space (= porosity).

    Returns
    -------
    complex
        √ε_mix  (complex refractive index of the mixture).
    """
    f_reg = solid_fraction - f_ice
    return (
        f_ice  * np.sqrt(eps_ice)
        + f_reg * np.sqrt(eps_reg)
        + vac_fraction * np.sqrt(eps_vac)
    )


def _make_residual(
    eps_mix_magnitude: float,
    eps_ice: complex,
    eps_reg: complex,
    eps_vac: complex,
    solid_fraction: float,
    vac_fraction: float,
) -> Callable[[float], float]:
    """Return a scalar residual function for brentq root-finding.

    Residual(f_ice) = |√ε_mix(f_ice)|² − ε_mix_magnitude

    Parameters
    ----------
    eps_mix_magnitude : float
        Target effective permittivity magnitude (real-valued proxy for ε_mix).
    eps_ice, eps_reg, eps_vac : complex
        End-member permittivities (see module docstring).
    solid_fraction : float
        = 1 − porosity.
    vac_fraction : float
        = porosity.

    Returns
    -------
    Callable[[float], float]
        Residual function suitable for ``scipy.optimize.brentq``.
    """
    def _residual(f_ice: float) -> float:
        sqrt_eps_mix = _crim_sqrt_eps(
            f_ice, eps_ice, eps_reg, eps_vac, solid_fraction, vac_fraction
        )
        predicted_magnitude = abs(sqrt_eps_mix) ** 2
        return predicted_magnitude - eps_mix_magnitude

    return _residual


def _invert_single_pixel(
    eps_mix_magnitude: float,
    eps_ice: complex,
    eps_reg: complex,
    eps_vac: complex,
    solid_fraction: float,
    vac_fraction: float,
    tolerance: float,
) -> float:
    """Invert the CRIM equation for a single pixel.

    Uses ``scipy.optimize.brentq`` because the residual is monotone in
    f_ice (increasing ice → increasing permittivity → increasing |√ε|²).

    Parameters
    ----------
    eps_mix_magnitude : float
        Effective permittivity magnitude derived from measured backscatter.
    eps_ice, eps_reg, eps_vac : complex
        End-member permittivities.
    solid_fraction : float
        Total solid fraction = 1 − porosity.
    vac_fraction : float
        Vacuum / pore fraction = porosity.
    tolerance : float
        Absolute convergence tolerance for brentq.

    Returns
    -------
    float
        Ice volumetric fraction f_ice ∈ [0, solid_fraction].
        Clamped to [0, solid_fraction] if the measured value is outside
        the physically reachable range.
    """
    residual = _make_residual(
        eps_mix_magnitude, eps_ice, eps_reg, eps_vac, solid_fraction, vac_fraction
    )

    # Bounds of the search: f_ice = 0 (pure regolith+vacuum) → f_ice = solid_fraction (no reg)
    lo_val = residual(0.0)
    hi_val = residual(solid_fraction)

    if lo_val > 0.0:
        # Measured permittivity lower than all-vacuum case → clamp to 0
        return 0.0
    if hi_val < 0.0:
        # Measured permittivity exceeds all-ice case → clamp to solid_fraction
        return float(solid_fraction)

    try:
        f_ice = brentq(residual, 0.0, solid_fraction, xtol=tolerance, maxiter=200)
    except ValueError:
        # Fallback: return midpoint; log a warning
        logger.warning(
            "brentq failed for eps_mix_magnitude=%.4g; returning 0.0", eps_mix_magnitude
        )
        f_ice = 0.0

    return float(f_ice)


def invert_ice_fraction(
    backscatter_db: np.ndarray,
    config: Module1Config,
) -> np.ndarray:
    """Invert radar backscatter to volumetric water-ice fraction via CRIM.

    Pipeline
    --------
    1. Convert dB backscatter to linear power:
          σ₀_lin = 10^(backscatter_dB / 10)

    2. Map linear σ₀ to effective bulk permittivity (simplified empirical proxy):
          ε_mix = config.crim_sigma_to_eps_slope · σ₀_lin
                  + config.crim_sigma_to_eps_intercept
       NOTE: This linear calibration is an intentional simplification.  A
       rigorous approach requires a full EM forward model (e.g. IEM or AIEM)
       relating ε_mix to σ₀ as a function of surface roughness and incidence
       angle.  The slope and intercept should be calibrated against drill-core
       or ground-truth measurements.

    3. For each pixel, invert CRIM equation (2) for f_ice using
       scipy.optimize.brentq on the domain [0, solid_fraction]:
          √ε_mix(f_ice) = f_ice·√ε_ice + (solid−f_ice)·√ε_reg + vac·√ε_vac
       Solve until |residual| < config.crim_tolerance.

    The pixel-wise scipy.optimize loop is unavoidable because brentq requires
    a scalar function; parallelism is left to the caller (e.g. joblib or Dask).

    Parameters
    ----------
    backscatter_db : np.ndarray, shape (rows, cols), dtype float32 or float64
        Calibrated radar backscatter coefficient in decibels (dB).
        Typical range for lunar surfaces: −30 dB to −5 dB.
    config : Module1Config
        Pipeline configuration.  The following fields are consumed:
          ``crim_eps_ice``, ``crim_eps_regolith``, ``crim_eps_vacuum``,
          ``crim_porosity``, ``crim_tolerance``,
          ``crim_sigma_to_eps_slope``, ``crim_sigma_to_eps_intercept``.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype float32
        Volumetric water-ice fraction f_ice ∈ [0, solid_fraction].
        A value of 0.30 means 30 % of the top ~5 m of regolith is water ice
        (the DFSAR L-band penetration depth for this permittivity range).

    Raises
    ------
    ValueError
        If ``backscatter_db`` is not a 2-D array or contains non-finite values.
    """
    if backscatter_db.ndim != 2:
        raise ValueError(
            f"backscatter_db must be 2-D; got shape {backscatter_db.shape}"
        )
    if not np.all(np.isfinite(backscatter_db)):
        raise ValueError("backscatter_db contains non-finite values (NaN or Inf).")

    logger.info(
        "CRIM inversion: shape=%s dB range=[%.2f, %.2f]",
        backscatter_db.shape,
        float(np.min(backscatter_db)),
        float(np.max(backscatter_db)),
    )

    # Step 1: dB → linear
    sigma_lin: np.ndarray = np.power(10.0, backscatter_db / 10.0)

    # Step 2: linear σ₀ → effective permittivity (empirical proxy)
    eps_mix_arr: np.ndarray = (
        config.crim_sigma_to_eps_slope * sigma_lin
        + config.crim_sigma_to_eps_intercept
    )
    # Physical plausibility clamp: ε_mix must be ≥ ε_vac = 1
    eps_mix_arr = np.clip(eps_mix_arr, 1.0, None)

    eps_ice     = config.crim_eps_ice
    eps_reg     = config.crim_eps_regolith
    eps_vac     = config.crim_eps_vacuum
    vac_frac    = config.crim_porosity
    solid_frac  = 1.0 - vac_frac
    tolerance   = config.crim_tolerance

    rows, cols = backscatter_db.shape
    f_ice_flat = np.empty(rows * cols, dtype=np.float32)

    # Pixel-wise inversion – scipy.optimize.brentq requires scalar inputs.
    # This loop is the computational bottleneck; vectorisation is not possible
    # without re-implementing the root-finder in NumPy (not done here to keep
    # numerical correctness guarantees).
    eps_flat = eps_mix_arr.ravel()
    for idx in range(rows * cols):
        f_ice_flat[idx] = _invert_single_pixel(
            float(eps_flat[idx]),
            eps_ice,
            eps_reg,
            eps_vac,
            solid_frac,
            vac_frac,
            tolerance,
        )

    f_ice_2d: np.ndarray = f_ice_flat.reshape(rows, cols)

    logger.info(
        "CRIM inversion complete: f_ice range=[%.4f, %.4f] mean=%.4f",
        float(np.min(f_ice_2d)),
        float(np.max(f_ice_2d)),
        float(np.mean(f_ice_2d)),
    )
    return f_ice_2d


def compute_ice_volume(
    f_ice: np.ndarray,
    pixel_area_m2: float,
    depth_m: float = 5.0,
) -> float:
    """Compute total water-ice volume from CRIM inversion.

    Volume = Σ (f_ice[pixel] × pixel_area × depth_m)

    where:
    - pixel_area_m2 is the ground area of one DFSAR pixel (typically ~75 m × 75 m
      from 4-look L-band, giving ~5625 m²).
    - depth_m is the radar penetration depth (≈5 m for L-band in lunar regolith).

    Parameters
    ----------
    f_ice : np.ndarray, shape (rows, cols)
        Volumetric water-ice fraction from ``invert_ice_fraction``.
    pixel_area_m2 : float
        Ground area of one radar pixel (m²).
    depth_m : float
        Effective penetration depth (m). Default: 5.0.

    Returns
    -------
    float
        Total water-ice volume in cubic metres (m³).
    """
    total_volume = float(np.sum(f_ice) * pixel_area_m2 * depth_m)
    logger.info(
        "Ice volume = %.2e m³ (%.2f tonnes @ 917 kg/m³)",
        total_volume,
        total_volume * 917.0,
    )
    return total_volume
