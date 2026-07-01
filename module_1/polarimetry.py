"""Radar polarimetric decomposition functions for DFSAR compact-pol data.

This module implements:
- Circular Polarisation Ratio (CPR) computation from compact-pol channels.
- Degree of Polarisation (DOP) via the 2×2 coherency matrix.
- Boolean ice-anomaly mask combining CPR and DOP thresholds.

Compact-Pol Conventions (Chandrayaan-2 DFSAR)
----------------------------------------------
L-band:  Transmit Left-Hand Circular (LHC), receive linear H (LH) and V (LV).
S-band:  Transmit Right-Hand Circular (RHC), receive linear H (RH) and V (RV).

Same-sense and opposite-sense circular reconstruction:
  LL  = (LH − j·LV) / √2       [L-band same-sense]
  LR  = (LH + j·LV) / √2       [L-band opposite-sense]
  RR  = (RH + j·RV) / √2       [S-band same-sense]
  RL  = (RH − j·RV) / √2       [S-band opposite-sense]

Reference
---------
Rao, M. N. et al. (2022). Chandrayaan-2 DFSAR compact-pol convention.
*Planetary and Space Science*, 220, 105508.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import Module1Config
from .dfsar_reader import DFSARProduct

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Small epsilon to guard against division by zero throughout this module.
_EPS: float = np.finfo(np.float32).eps


def compute_cpr(product: DFSARProduct, config: Module1Config) -> np.ndarray:
    """Compute the Circular Polarisation Ratio (CPR) from compact-pol data.

    The CPR is the ratio of same-sense circular to opposite-sense circular
    backscattered power.  For rough/volume scatterers (e.g. water-ice deposits)
    CPR > 1; for smooth Fresnel reflectors CPR < 1.

    Compact-pol channel reconstruction
    ------------------------------------
    L-band (LHC transmit, H/V receive):
      LL  = (LH − j·LV) / √2          same-sense
      LR  = (LH + j·LV) / √2          opposite-sense
      P_SS_L = |LL|² = ½(|LH|² + |LV|² + 2·Im(LH·LV*))
      P_OS_L = |LR|² = ½(|LH|² + |LV|² − 2·Im(LH·LV*))

    S-band (RHC transmit, H/V receive):
      RR  = (RH + j·RV) / √2          same-sense
      RL  = (RH − j·RV) / √2          opposite-sense
      P_SS_S = |RR|² = ½(|RH|² + |RV|² + 2·Im(RH·RV*))   [sign flipped for RHC]
      P_OS_S = |RL|² = ½(|RH|² + |RV|² − 2·Im(RH·RV*))

    CPR selection (config.cpr_band):
      'L'    → CPR = P_SS_L / (P_OS_L + ε)
      'S'    → CPR = P_SS_S / (P_OS_S + ε)
      'both' → CPR = (P_SS_L + P_SS_S) / (P_OS_L + P_OS_S + ε)

    Reference: DFSAR compact-pol convention per Rao et al. 2022,
               Planetary and Space Science.

    Parameters
    ----------
    product : DFSARProduct
        Loaded compact-pol product.  Channels ``lh``, ``lv``, ``rh``, ``rv``
        must be complex64 arrays of identical shape.
    config : Module1Config
        Pipeline configuration.  ``config.cpr_band`` controls which bands
        contribute.

    Returns
    -------
    np.ndarray, shape product.shape, dtype float32
        CPR values.  Dimensionless.  Typical range [0, ∞); ice anomalies > 1.
    """
    lh: np.ndarray = product.lh
    lv: np.ndarray = product.lv
    rh: np.ndarray = product.rh
    rv: np.ndarray = product.rv

    # ── L-band circular reconstruction ──────────────────────────────────────
    _inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
    ll = (lh - 1j * lv) * _inv_sqrt2   # same-sense, LHC transmit
    lr = (lh + 1j * lv) * _inv_sqrt2   # opposite-sense

    p_ss_l = np.abs(ll) ** 2
    p_os_l = np.abs(lr) ** 2

    # ── S-band circular reconstruction ──────────────────────────────────────
    rr = (rh + 1j * rv) * _inv_sqrt2   # same-sense, RHC transmit
    rl = (rh - 1j * rv) * _inv_sqrt2   # opposite-sense

    p_ss_s = np.abs(rr) ** 2
    p_os_s = np.abs(rl) ** 2

    # ── Band selection ───────────────────────────────────────────────────────
    band = config.cpr_band
    if band == "L":
        numerator = p_ss_l
        denominator = p_os_l + _EPS
    elif band == "S":
        numerator = p_ss_s
        denominator = p_os_s + _EPS
    else:  # 'both'
        numerator = p_ss_l + p_ss_s
        denominator = p_os_l + p_os_s + _EPS

    cpr = (numerator / denominator).astype(np.float32)

    logger.debug(
        "CPR computed (band='%s'): min=%.4g max=%.4g mean=%.4g",
        band,
        float(np.nanmin(cpr)),
        float(np.nanmax(cpr)),
        float(np.nanmean(cpr)),
    )
    return cpr


def compute_dop(
    product: DFSARProduct,
    config: Module1Config | None = None,
    window_size: int = 5,
) -> np.ndarray:
    """Compute the Degree of Polarisation (DOP) from compact-pol data.

    The DOP quantifies how polarised the backscattered wave is.  It is derived
    from the spatially-averaged 2×2 coherency matrix **⟨T⟩** of each compact-
    pol pair.  Spatial averaging (multi-looking) is essential: for a single
    look (deterministic Jones vector) det(T) = 0 algebraically, giving DOP = 1
    regardless of the true polarisation state.

    Coherency matrix (L-band, E = [LH; LV])
    -----------------------------------------
    ⟨T_L⟩[0,0] = ⟨|LH|²⟩
    ⟨T_L⟩[1,1] = ⟨|LV|²⟩
    ⟨T_L⟩[0,1] = ⟨LH · LV*⟩
    ⟨T_L⟩[1,0] = conj(⟨T_L⟩[0,1])

    DOP formula (for a 2×2 coherency matrix):
      DOP = √( 1 − 4·det(⟨T⟩) / trace(⟨T⟩)² )

    where:
      det(⟨T⟩)   = ⟨T⟩[0,0]·⟨T⟩[1,1] − |⟨T⟩[0,1]|²
      trace(⟨T⟩) = ⟨T⟩[0,0] + ⟨T⟩[1,1]

    Identical structure applied to S-band (RH, RV channels).
    Final DOP is the pixel-wise mean of DOP_L and DOP_S.

    Spatial averaging is implemented as a uniform boxcar filter of size
    ``window_size × window_size`` applied to each coherency-matrix element
    before forming the determinant trace ratio.

    Ranges
    ------
    DOP ∈ [0, 1]:
      0 → fully unpolarised (equal power in all polarisations)
      1 → fully polarised (single dominant scattering mechanism)

    Ice-bearing pixels exhibit partial depolarisation (DOP < 0.13, Rao 2022).

    Parameters
    ----------
    product : DFSARProduct
        Compact-pol product with complex channels.
    config : Module1Config or None
        Not currently used; kept for API symmetry.
    window_size : int
        Boxcar averaging window size (odd integer).  Default 5.
        Set to 1 to disable averaging (returns DOP = 1 for all pixels).

    Returns
    -------
    np.ndarray, shape product.shape, dtype float32
        DOP values in [0, 1].  Dimensionless.
    """
    lh = product.lh
    lv = product.lv
    rh = product.rh
    rv = product.rv

    def _boxcar(arr: np.ndarray, w: int) -> np.ndarray:
        """Uniform boxcar filter with reflected padding."""
        if w <= 1:
            return arr
        kernel = np.ones((w, w), dtype=arr.dtype) / (w * w)
        padded = np.pad(arr, pad_width=w // 2, mode="reflect")
        from scipy.signal import convolve2d
        return convolve2d(padded, kernel, mode="valid")

    def _dop_from_pair(ch_h: np.ndarray, ch_v: np.ndarray) -> np.ndarray:
        """Compute DOP from a single compact-pol pair with multi-looking.

        Parameters
        ----------
        ch_h : np.ndarray, complex
            H-polarisation receive channel (complex amplitudes).
        ch_v : np.ndarray, complex
            V-polarisation receive channel (complex amplitudes).

        Returns
        -------
        np.ndarray, float32
            DOP for this pair.
        """
        t00 = np.abs(ch_h) ** 2                     # |H|²
        t11 = np.abs(ch_v) ** 2                     # |V|²
        t01 = ch_h * np.conj(ch_v)                  # H · V*

        # Spatial averaging (multi-looking) of coherency matrix elements
        if window_size > 1:
            t00 = _boxcar(t00, window_size)
            t11 = _boxcar(t11, window_size)
            t01 = _boxcar(t01, window_size)

        trace = t00 + t11                            # trace(⟨T⟩)
        det   = t00 * t11 - np.abs(t01) ** 2        # det(⟨T⟩)

        trace_sq = trace ** 2
        ratio = np.where(
            trace_sq > _EPS,
            4.0 * np.real(det) / (trace_sq + _EPS),
            0.0,
        )
        ratio_clamped = np.clip(1.0 - ratio, 0.0, 1.0)
        return np.sqrt(ratio_clamped).astype(np.float32)

    dop_l = _dop_from_pair(lh, lv)
    dop_s = _dop_from_pair(rh, rv)
    dop   = ((dop_l + dop_s) * 0.5).astype(np.float32)

    logger.debug(
        "DOP computed: min=%.4g max=%.4g mean=%.4g",
        float(np.nanmin(dop)),
        float(np.nanmax(dop)),
        float(np.nanmean(dop)),
    )
    return dop


def build_ice_mask(
    cpr: np.ndarray,
    dop: np.ndarray,
    config: Module1Config,
) -> np.ndarray:
    """Construct a boolean ice-anomaly mask using ML Isolation Forest.

    Replaces rigid CPR/DOP thresholds with an unsupervised Isolation Forest
    that identifies statistically anomalous pixels in the (CPR, DOP, log-CPR)
    feature space. This is robust to absolute signal level shifts in real
    ISRO DFSAR data caused by orbital geometry or calibration drift.

    The model flags the top ``contamination`` fraction (default 5%) of pixels
    as anomalous ice candidates. Falls back to a top-percentile CPR heuristic
    if scikit-learn is unavailable.

    Parameters
    ----------
    cpr : np.ndarray, shape (rows, cols), dtype float32
        Circular Polarisation Ratio map.
    dop : np.ndarray, shape (rows, cols), dtype float32
        Degree of Polarisation map, range [0, 1].
    config : Module1Config
        Pipeline configuration.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype bool
        True where anomalous ice-bearing pixels are predicted.
    """
    if cpr.shape != dop.shape:
        raise ValueError(
            f"CPR shape {cpr.shape} and DOP shape {dop.shape} must match."
        )

    rows, cols = cpr.shape
    n_pixels = rows * cols

    # Build feature matrix: CPR, DOP, log-CPR (more Gaussian distributed)
    cpr_flat = cpr.ravel().astype(np.float64)
    dop_flat = dop.ravel().astype(np.float64)
    log_cpr_flat = np.log1p(np.clip(cpr_flat, 0, None))

    X = np.column_stack([cpr_flat, dop_flat, log_cpr_flat])

    # Replace NaN/Inf with column medians
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        finite_mask = np.isfinite(col)
        if not np.all(finite_mask):
            median_val = np.median(col[finite_mask]) if np.any(finite_mask) else 0.0
            col[~finite_mask] = median_val
            X[:, col_idx] = col

    if _SKLEARN_AVAILABLE:
        # Normalise features so CPR magnitude doesn't dominate
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        contamination = getattr(config, 'ml_contamination', 0.05)
        clf = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_scaled)

        # -1 = anomaly (ice candidate), +1 = normal background
        predictions = clf.predict(X_scaled)
        ice_flat = (predictions == -1)

        logger.info(
            "IsolationForest ice mask: %d / %d pixels (%.2f%%) [contamination=%.2f]",
            int(np.sum(ice_flat)), n_pixels,
            100.0 * np.sum(ice_flat) / n_pixels,
            contamination,
        )
    else:
        # Fallback: top-5% CPR pixels
        logger.warning("scikit-learn unavailable — using top-5%% CPR percentile fallback.")
        threshold = np.nanpercentile(cpr_flat, 95)
        ice_flat = cpr_flat >= threshold
        logger.info(
            "Fallback CPR threshold=%.4g: %d / %d pixels flagged",
            threshold, int(np.sum(ice_flat)), n_pixels,
        )

    return ice_flat.reshape(rows, cols).astype(bool)
