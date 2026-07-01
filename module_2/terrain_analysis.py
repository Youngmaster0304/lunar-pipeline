"""Terrain analysis functions for lunar DEM processing (Module 2).

Provides:
- ``compute_slope``  — Sobel-based slope computation from a DEM.
- ``load_auxiliary_rasters`` — Load DTE and illumination persistence rasters.
- ``make_synthetic_auxiliary`` — Generate synthetic test data.

Slope computation uses the standard 3×3 Sobel gradient operator, which
approximates the partial derivatives of elevation:

  ∂z/∂x ≈ Sobel_x(z) / (8 · pixel_spacing_m)
  ∂z/∂y ≈ Sobel_y(z) / (8 · pixel_spacing_m)

The gradient magnitude is:
  |∇z| = √((∂z/∂x)² + (∂z/∂y)²)    [m/m, dimensionless]

Slope in degrees:
  slope = arctan(|∇z|) × 180 / π    [°]

The Sobel kernel (x-direction, normalised form):
  [ −1  0  +1 ]
  [ −2  0  +2 ] / 8
  [ −1  0  +1 ]

Dividing by 8 (rather than the raw filter sum of 4) gives the central-
difference approximation to the derivative with the standard Sobel
smoothing factor included.  ``scipy.ndimage.sobel`` applies the
unnormalised kernel (sum = 4); we therefore divide by 8·pixel_spacing_m
to obtain the proper physical gradient.

Reference
---------
Horn, B. K. P. (1981). *Hill shading and the reflectance map.*
Proceedings of the IEEE, 69(1), 14–47.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, sobel

logger = logging.getLogger(__name__)


def compute_slope(dem: np.ndarray, pixel_spacing_m: float) -> np.ndarray:
    """Compute terrain slope from a digital elevation model using Sobel gradients.

    Algorithm
    ---------
    1. Apply ``scipy.ndimage.sobel`` along axis=1 (x-direction) and axis=0
       (y-direction) to obtain raw gradient images.
    2. Normalise by ``8 × pixel_spacing_m`` to convert Sobel filter output
       to physical gradient (m/m):
         gx = sobel_x / (8 · pixel_spacing_m)
         gy = sobel_y / (8 · pixel_spacing_m)
    3. Gradient magnitude:
         |∇z| = √(gx² + gy²)        [m/m]
    4. Slope in degrees:
         slope = arctan(|∇z|) × 180/π    [°]

    The Sobel kernel normalisation factor of 8 arises from the classical
    Sobel definition where the kernel is:
      [−1, 0, +1; −2, 0, +2; −1, 0, +1]
    which accumulates to a scale factor of 4 in the central-difference sense,
    and the additional factor of 2 from the pixel_spacing yields the full 8.

    Parameters
    ----------
    dem : np.ndarray, shape (rows, cols), dtype float32 or float64
        Digital Elevation Model.  Elevation values in metres.
        Must be 2-D and finite (no NaN or Inf).
    pixel_spacing_m : float
        Ground sampling distance in metres per pixel.  Must be positive.
        Typical LOLA DEM: 20 m/pixel; Mini-RF DEM: 75 m/pixel.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype float32
        Terrain slope in degrees.  Range: [0°, 90°).

    Raises
    ------
    ValueError
        If ``dem`` is not 2-D, contains NaN/Inf, or ``pixel_spacing_m ≤ 0``.
    """
    if dem.ndim != 2:
        raise ValueError(
            f"dem must be 2-D; got shape {dem.shape} (ndim={dem.ndim})."
        )
    if not np.all(np.isfinite(dem)):
        raise ValueError("dem contains NaN or Inf values; clean the input first.")
    if pixel_spacing_m <= 0.0:
        raise ValueError(
            f"pixel_spacing_m must be positive; got {pixel_spacing_m}."
        )

    dem_f = dem.astype(np.float64)

    # Sobel gradient images (scipy uses mode='reflect' by default)
    sobel_x = sobel(dem_f, axis=1)   # ∂z/∂x direction (columns)
    sobel_y = sobel(dem_f, axis=0)   # ∂z/∂y direction (rows)

    # Normalise to physical gradient [m/m]
    # scipy.ndimage.sobel output corresponds to 4·Δz over 2·pixel_spacing
    # → divide by 8·pixel_spacing_m to get central-difference derivative
    scale = 8.0 * pixel_spacing_m
    gx = sobel_x / scale
    gy = sobel_y / scale

    gradient_magnitude = np.hypot(gx, gy)  # |∇z| in m/m
    slope_deg = np.degrees(np.arctan(gradient_magnitude)).astype(np.float32)

    logger.debug(
        "Slope computed: pixel_spacing_m=%.2f min=%.2f° max=%.2f° mean=%.2f°",
        pixel_spacing_m,
        float(np.min(slope_deg)),
        float(np.max(slope_deg)),
        float(np.mean(slope_deg)),
    )
    return slope_deg


def load_auxiliary_rasters(
    dte_path: str,
    illumination_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load DTE and illumination persistence rasters from disk.

    DTE (Direct-To-Earth) is a boolean raster indicating whether each pixel
    has a direct line-of-sight to Earth.  Illumination persistence is a
    float raster in [0, 1] representing the fraction of a given time window
    during which the pixel is sunlit.

    Parameters
    ----------
    dte_path : str
        Path to the DTE raster.  Expected dtype: uint8 or bool (0=no LOS,
        1=LOS).  Any non-zero value is interpreted as True.
    illumination_path : str
        Path to the illumination persistence raster.  Expected dtype: float32.
        Values must lie in [0, 1].

    Returns
    -------
    dte : np.ndarray, shape (rows, cols), dtype bool
        Direct-To-Earth boolean array.
    illumination : np.ndarray, shape (rows, cols), dtype float32
        Illumination persistence fraction array.

    Raises
    ------
    ImportError
        If ``rasterio`` is not installed.
    ValueError
        If the two rasters have different shapes, or if illumination values
        fall outside [0, 1].
    """
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to load auxiliary rasters. "
            "Install with: pip install rasterio"
        ) from exc

    logger.info("Loading DTE raster from: %s", dte_path)
    with rasterio.open(dte_path) as src:
        dte_raw = src.read(1)
    dte = (dte_raw != 0).astype(bool)
    logger.debug("DTE shape=%s, True fraction=%.3f", dte.shape, float(dte.mean()))

    logger.info("Loading illumination raster from: %s", illumination_path)
    with rasterio.open(illumination_path) as src:
        illum_raw = src.read(1).astype(np.float32)

    if dte.shape != illum_raw.shape:
        raise ValueError(
            f"DTE shape {dte.shape} != illumination shape {illum_raw.shape}."
        )

    illum_min = float(np.nanmin(illum_raw))
    illum_max = float(np.nanmax(illum_raw))
    if illum_min < 0.0 or illum_max > 1.0:
        raise ValueError(
            f"Illumination values must be in [0, 1]; found range [{illum_min:.4f}, {illum_max:.4f}]."
        )

    logger.debug(
        "Illumination shape=%s range=[%.3f, %.3f]",
        illum_raw.shape,
        illum_min,
        illum_max,
    )
    return dte, illum_raw


def load_ohrc_ortho(ohrc_path: str) -> np.ndarray:
    """Load OHRC high-resolution ortho raster from a GeoTIFF on disk.

    Parameters
    ----------
    ohrc_path : str
        Path to the OHRC GeoTIFF.  Expected content: 1-band float32 or
        uint16 elevation / backscatter raster.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype float32
        OHRC ortho image array.

    Raises
    ------
    ImportError
        If ``rasterio`` is not installed.
    FileNotFoundError
        If the path does not exist.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to load OHRC ortho rasters. "
            "Install with: pip install rasterio"
        ) from exc

    logger.info("Loading OHRC ortho from: %s", ohrc_path)
    with rasterio.open(ohrc_path) as src:
        img = src.read(1).astype(np.float32)
    logger.debug("OHRC ortho loaded: shape=%s range=[%.2f, %.2f]", img.shape, float(np.nanmin(img)), float(np.nanmax(img)))
    return img


def make_synthetic_auxiliary(
    shape: tuple[int, int],
    dte_true_fraction: float = 0.7,
    illumination_mean: float = 0.65,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic DTE and illumination rasters for testing.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(rows, cols)`` of the output arrays.
    dte_true_fraction : float
        Fraction of pixels that have direct-to-Earth line-of-sight.
        Range: [0, 1].  Default 0.7.
    illumination_mean : float
        Mean illumination persistence value.  The distribution is Beta(α, β)
        parameterised to achieve this mean with moderate spread.
        Range: (0, 1).  Default 0.65.
    seed : int
        NumPy random seed for reproducibility.  Default 42.

    Returns
    -------
    dte : np.ndarray, shape ``shape``, dtype bool
        Synthetic Direct-To-Earth boolean array.
    illumination : np.ndarray, shape ``shape``, dtype float32
        Synthetic illumination persistence array in [0, 1].
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    n = rows * cols

    # DTE: Bernoulli with given true fraction
    dte_flat = rng.random(n) < dte_true_fraction
    dte = dte_flat.reshape(shape)

    # Illumination: Beta distribution clipped to [0, 1]
    mean = illumination_mean
    alpha = mean * 5.0
    beta_param = (1.0 - mean) * 5.0
    illum_flat = rng.beta(alpha, beta_param, size=n).astype(np.float32)
    illumination = illum_flat.reshape(shape)

    logger.debug(
        "Synthetic auxiliary created: shape=%s dte_frac=%.2f illum_mean=%.3f",
        shape,
        float(dte.mean()),
        float(illumination.mean()),
    )
    return dte, illumination


def compute_surface_roughness(
    dem: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute surface roughness as local standard deviation of elevation.

    Uses a sliding window to compute the standard deviation of elevation
    within each neighbourhood.  Higher values indicate rougher terrain
    (boulder fields, crater rim ejecta).

    Parameters
    ----------
    dem : np.ndarray, shape (rows, cols), dtype float32
        Digital elevation model in metres.
    window_size : int
        Side length of the square moving window.  Must be odd and ≥ 3.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype float32
        Local elevation standard deviation in metres.
    """
    from scipy.ndimage import uniform_filter

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be odd and ≥ 3; got {window_size}")

    dem_f = dem.astype(np.float64)
    sq = uniform_filter(dem_f ** 2, size=window_size, mode='reflect')
    mu = uniform_filter(dem_f, size=window_size, mode='reflect')
    variance = np.clip(sq - mu ** 2, 0.0, None)
    roughness = np.sqrt(variance).astype(np.float32)

    logger.debug(
        "Surface roughness: window=%d range=[%.2f, %.2f] mean=%.2f m",
        window_size,
        float(np.min(roughness)),
        float(np.max(roughness)),
        float(np.mean(roughness)),
    )
    return roughness


def detect_boulders(
    dem: np.ndarray,
    prominence_threshold: float = 2.0,
    min_distance: int = 3,
) -> np.ndarray:
    """Detect boulders as local elevation maxima above a prominence threshold.

    A boulder is defined as a pixel whose elevation exceeds the local
    background (median-filtered DEM) by at least ``prominence_threshold``
    metres and is a local maximum in its neighbourhood.

    Parameters
    ----------
    dem : np.ndarray, shape (rows, cols), dtype float32
        Digital elevation model in metres.
    prominence_threshold : float
        Minimum elevation above local background to classify as a boulder.
        Units: metres.
    min_distance : int
        Minimum separation between boulder detections in pixels.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype bool
        True at boulder locations.
    """
    from scipy.ndimage import maximum_filter, median_filter

    dem_f = dem.astype(np.float64)

    # Local background: median filter removes boulder-scale features
    background = median_filter(dem_f, size=min_distance * 2 + 1, mode='reflect')

    # Residual elevation above background
    residual = dem_f - background

    # Local maxima detection
    local_max = dem_f == maximum_filter(dem_f, size=min_distance, mode='reflect')

    boulder_mask = local_max & (residual > prominence_threshold)

    n_boulders = int(np.sum(boulder_mask))
    logger.info(
        "Boulder detection: %d boulders found (prominence=%.1f m, min_dist=%d px)",
        n_boulders, prominence_threshold, min_distance,
    )
    return boulder_mask


def make_synthetic_ohrc(
    shape: tuple[int, int],
    dem: np.ndarray | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic high-resolution OHRC-like imagery for testing.

    Produces a simulated DEM with crater-like depressions, ridge features,
    and random boulder-sized noise to approximate OHRC-resolution topography.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(rows, cols)``.
    dem : np.ndarray or None
        Optional base DEM to add high-res detail to.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape ``shape``, dtype float32
        Synthetic high-resolution DEM with fine-scale terrain features.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    if dem is None:
        dem = np.ones(shape, dtype=np.float32) * 1000.0

    # Add fractal-like noise for fine-scale roughness
    fine_noise = rng.normal(0, 1.5, size=shape).astype(np.float32)
    smooth_noise = minimum_filter(maximum_filter(fine_noise, size=3), size=3)

    # Add a few simulated boulders (sharp local maxima)
    boulder_z = np.zeros(shape, dtype=np.float32)
    for _ in range(rng.poisson(20)):
        br = rng.integers(5, rows - 5)
        bc = rng.integers(5, cols - 5)
        bh = rng.uniform(3.0, 8.0)
        boulder_z[br - 1:br + 2, bc - 1:bc + 2] = bh

    ohrc_dem = dem + smooth_noise + boulder_z
    logger.debug("Synthetic OHRC DEM created: shape=%s", shape)
    return ohrc_dem.astype(np.float32)
