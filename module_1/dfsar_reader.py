"""DFSAR product reader for Chandrayaan-2 DFSAR Level-2 compact-polarimetry data.

This module provides:
- ``DFSARProduct`` – an immutable dataclass holding all four compact-pol
  channels plus georeferencing metadata.
- ``load_dfsar_product`` – reads a GeoTIFF (or any rasterio-supported format)
  with 4 bands into a ``DFSARProduct``.
- ``load_dfsar_product_synthetic`` – generates a synthetic product for unit
  testing without real satellite data.

Data convention (DFSAR compact-pol)
-------------------------------------
Band 1 → LH  (L-band, H-receive linear amplitude/power)
Band 2 → LV  (L-band, V-receive linear amplitude/power)
Band 3 → RH  (S-band, H-receive linear amplitude/power)
Band 4 → RV  (S-band, V-receive linear amplitude/power)

Reference
---------
Rao, M. N. et al. (2022). *Chandrayaan-2 DFSAR: first results from the
Lunar South Pole.* Planetary and Space Science, 220, 105508.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Module1Config

logger = logging.getLogger(__name__)


@dataclass
class DFSARProduct:
    """Immutable container for a DFSAR compact-polarimetry tile.

    All amplitude arrays are complex64 if the source file contains
    complex data, or float32 power values if only real power was stored.
    Shape consistency across all four arrays is guaranteed by
    ``load_dfsar_product``.

    Attributes
    ----------
    lh : np.ndarray, shape (rows, cols), dtype complex64 or float32
        L-band, H-receive complex backscatter amplitude (or power if
        the source file does not carry phase information).
        Units: complex voltage ratio (dimensionless) or m²/m² (power).
    lv : np.ndarray, shape (rows, cols), dtype complex64 or float32
        L-band, V-receive complex backscatter amplitude.
    rh : np.ndarray, shape (rows, cols), dtype complex64 or float32
        S-band (referred to as 'R' for the second radar), H-receive
        complex backscatter amplitude.
    rv : np.ndarray, shape (rows, cols), dtype complex64 or float32
        S-band, V-receive complex backscatter amplitude.
    geotransform : tuple[float, float, float, float, float, float]
        GDAL-style affine parameters:
        ``(x_origin, pixel_width, 0, y_origin, 0, pixel_height)``.
        Units: map units of the CRS (typically degrees or metres).
    crs : str
        Coordinate Reference System as a WKT string or EPSG code string
        (e.g. ``'EPSG:32643'``).
    shape : tuple[int, int]
        ``(rows, cols)`` of every band array.
    """

    lh: np.ndarray
    lv: np.ndarray
    rh: np.ndarray
    rv: np.ndarray
    geotransform: tuple
    crs: str
    shape: tuple


def load_dfsar_product(path: str, config: Module1Config) -> DFSARProduct:
    """Load a DFSAR compact-pol tile from a rasterio-readable raster file.

    The function expects a 4-band raster whose bands correspond to the
    DFSAR compact-pol convention (LH, LV, RH, RV).  If the bands are
    stored as complex dtype (e.g. ``complex64``) they are returned as-is.
    If the bands are stored as real floating-point (power) values they are
    cast to ``complex64`` with zero imaginary part so that downstream
    polarimetric operators can apply a uniform code path.

    Parameters
    ----------
    path : str
        Absolute or relative path to the input raster file.
    config : Module1Config
        Pipeline configuration object (currently used for future
        band-selection extensions; validated on entry).

    Returns
    -------
    DFSARProduct
        Populated product container.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or cannot be opened by rasterio.
    ValueError
        If the raster does not have exactly 4 bands, if band shapes are
        inconsistent, or if any band contains only NaN values.
    ImportError
        If ``rasterio`` is not installed in the current environment.
    """
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to load real DFSAR products. "
            "Install it with: pip install rasterio"
        ) from exc

    logger.info("Opening DFSAR product: %s", path)

    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(
                f"Expected ≥ 4 bands in {path}; found {src.count}."
            )

        crs_str = src.crs.to_wkt() if src.crs else "UNKNOWN"
        gt = (
            src.transform.c,
            src.transform.a,
            0.0,
            src.transform.f,
            0.0,
            src.transform.e,
        )

        bands_raw: list[np.ndarray] = []
        for band_idx in range(1, 5):
            arr = src.read(band_idx)
            logger.debug(
                "Band %d: shape=%s dtype=%s min=%.4g max=%.4g mean=%.4g",
                band_idx,
                arr.shape,
                arr.dtype,
                float(np.nanmin(arr)),
                float(np.nanmax(arr)),
                float(np.nanmean(arr)),
            )
            if np.all(np.isnan(arr)):
                raise ValueError(
                    f"Band {band_idx} in {path} contains only NaN values."
                )
            bands_raw.append(arr)

    # Validate shape consistency
    ref_shape = bands_raw[0].shape
    for i, arr in enumerate(bands_raw[1:], start=2):
        if arr.shape != ref_shape:
            raise ValueError(
                f"Band {i} shape {arr.shape} differs from band 1 shape {ref_shape}."
            )

    # Promote real arrays to complex so downstream code is uniform
    def _to_complex(arr: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(arr):
            return arr.astype(np.complex64)
        return arr.astype(np.float32).astype(np.complex64)

    lh, lv, rh, rv = [_to_complex(b) for b in bands_raw]

    logger.info(
        "Loaded DFSARProduct: shape=%s crs='%s'", ref_shape, crs_str[:60]
    )
    return DFSARProduct(
        lh=lh,
        lv=lv,
        rh=rh,
        rv=rv,
        geotransform=gt,
        crs=crs_str,
        shape=ref_shape,
    )


def load_dfsar_product_synthetic(
    shape: tuple[int, int] = (100, 100),
    seed: int = 42,
) -> DFSARProduct:
    """Generate a synthetic DFSAR product for testing and validation.

    The four compact-pol channels are populated with complex Gaussian noise
    that has unit variance per real/imaginary component, simulating a
    fully-developed speckle scene.  A small region (top-left quadrant) is
    given elevated same-sense power to simulate an ice-bearing anomaly.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(rows, cols)`` of the synthetic product.  Default ``(100, 100)``.
    seed : int
        NumPy random seed for reproducibility.  Default ``42``.

    Returns
    -------
    DFSARProduct
        Synthetic product with:
        - ``geotransform`` representing a 5 m/pixel grid at a nominal
          lunar south-pole location (lon=0°, lat=-89.5°).
        - ``crs = 'EPSG:4326'`` (geographic, WGS-84 as placeholder).
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    def _noise(scale: float = 1.0) -> np.ndarray:
        re = rng.normal(0.0, scale / np.sqrt(2), size=(rows, cols))
        im = rng.normal(0.0, scale / np.sqrt(2), size=(rows, cols))
        return (re + 1j * im).astype(np.complex64)

    lh = _noise(1.0)
    lv = _noise(1.0)
    rh = _noise(1.0)
    rv = _noise(1.0)

    # Inject ice-like anomaly in top-left quadrant: boost same-sense power
    # by reducing cross-pol component so that CPR ≫ 1 in that region.
    r_half, c_half = rows // 2, cols // 2
    lv[:r_half, :c_half] *= 0.1   # suppress cross-pol → high CPR
    rv[:r_half, :c_half] *= 0.1

    logger.debug(
        "Synthetic DFSARProduct created: shape=%s seed=%d", shape, seed
    )
    return DFSARProduct(
        lh=lh,
        lv=lv,
        rh=rh,
        rv=rv,
        geotransform=(0.0, 5e-5, 0.0, -89.5, 0.0, -5e-5),
        crs="EPSG:4326",
        shape=shape,
    )
