"""Pytest test suite for Module 2: Landing Site Seeding.

All tests use purely synthetic NumPy arrays; no real raster data files
are required.  Helper factories build controlled DEM, DTE, and illumination
arrays that yield predictable slope/score/mask outcomes.

Run with::

    pytest module_2/tests/test_module2.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from module_2.config import Module2Config
from module_2.mcda import (
    CandidateSite,
    build_candidate_mask,
    compute_exposure_score,
    extract_candidate_sites,
)
from module_2.terrain_analysis import compute_slope, make_synthetic_auxiliary


# ─────────────────────────────────────────────────────────────────────────────
# Tests for compute_slope
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSlope:
    """Tests for ``terrain_analysis.compute_slope``."""

    def test_slope_flat_dem(self) -> None:
        """A flat (all-zero) DEM must produce zero slope everywhere.

        For a constant surface z(r, c) = 0:
          Sobel_x = 0, Sobel_y = 0 → |∇z| = 0 → slope = arctan(0) = 0°.
        """
        dem = np.zeros((20, 20), dtype=np.float32)
        slope = compute_slope(dem, pixel_spacing_m=20.0)
        assert slope.dtype == np.float32
        assert slope.shape == dem.shape
        np.testing.assert_allclose(slope, 0.0, atol=1e-5)

    def test_slope_ramp(self) -> None:
        """A linear ramp with known spacing must produce the correct slope angle.

        Construct a DEM that rises by ``rise`` metres per pixel in the x-direction:
          z(r, c) = rise * c

        The gradient in x is:
          ∂z/∂x = rise / pixel_spacing_m   [m/m]

        Expected slope:
          slope = arctan(rise / pixel_spacing_m) × 180 / π   [°]

        We check the interior pixels (away from the 1-pixel border where
        Sobel may introduce boundary effects).
        """
        pixel_spacing_m = 20.0
        rise_per_pixel  = 2.0      # 2 m elevation gain per pixel
        rows, cols = 30, 30

        # z increases by rise_per_pixel per column → physical grad = rise_per_pixel / spacing
        col_indices = np.arange(cols, dtype=np.float32)
        dem = np.outer(np.ones(rows, dtype=np.float32), col_indices * rise_per_pixel)

        slope = compute_slope(dem, pixel_spacing_m=pixel_spacing_m)

        expected_grad_mm = rise_per_pixel / pixel_spacing_m  # m/m
        expected_slope_deg = float(np.degrees(np.arctan(expected_grad_mm)))

        # Interior pixels (avoid 1-pixel Sobel border artefacts)
        interior = slope[1:-1, 1:-1]
        np.testing.assert_allclose(
            interior,
            expected_slope_deg,
            atol=0.5,  # ±0.5° tolerance for Sobel boundary effects
            err_msg=(
                f"Expected interior slope ≈ {expected_slope_deg:.3f}°; "
                f"got mean={interior.mean():.3f}°"
            ),
        )

    def test_slope_single_spike(self) -> None:
        """A single elevated pixel must not crash and must return finite values.

        This tests robustness of the Sobel filter to impulse-like inputs.
        """
        dem = np.zeros((15, 15), dtype=np.float32)
        dem[7, 7] = 100.0  # spike
        slope = compute_slope(dem, pixel_spacing_m=10.0)
        assert np.all(np.isfinite(slope)), "Slope contains NaN or Inf for spike DEM"
        assert float(np.max(slope)) > 0.0

    def test_slope_invalid_1d_raises(self) -> None:
        """1-D input must raise ValueError."""
        dem_1d = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            compute_slope(dem_1d, pixel_spacing_m=20.0)

    def test_slope_nan_raises(self) -> None:
        """DEM with NaN must raise ValueError."""
        dem = np.zeros((10, 10), dtype=np.float32)
        dem[5, 5] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_slope(dem, pixel_spacing_m=20.0)

    def test_slope_negative_spacing_raises(self) -> None:
        """Non-positive pixel spacing must raise ValueError."""
        dem = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="positive"):
            compute_slope(dem, pixel_spacing_m=-5.0)

    def test_slope_dtype_float32(self) -> None:
        """Output must be float32 regardless of input dtype."""
        dem = np.ones((10, 10), dtype=np.float64) * 500.0
        slope = compute_slope(dem, pixel_spacing_m=30.0)
        assert slope.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Tests for build_candidate_mask
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCandidateMask:
    """Tests for ``mcda.build_candidate_mask``."""

    def _default_config(self) -> Module2Config:
        return Module2Config(
            max_slope_deg=10.0,
            illumination_threshold=0.7,
            exposure_weight_illumination=0.6,
            exposure_weight_dte=0.4,
        )

    def test_candidate_mask_all_good(self) -> None:
        """All pixels pass all constraints → mask is all True."""
        config = self._default_config()
        shape = (10, 10)
        slope      = np.full(shape, 5.0,  dtype=np.float32)   # < 10°
        dte        = np.ones(shape,  dtype=bool)               # all True
        illumination = np.full(shape, 0.9, dtype=np.float32)  # ≥ 0.7
        mask = build_candidate_mask(slope, dte, illumination, config)
        assert mask.dtype == bool
        assert np.all(mask), "Expected all True but got some False"

    def test_candidate_mask_none_pass_high_slope(self) -> None:
        """High slope everywhere → no pixel passes → mask is all False."""
        config = self._default_config()
        shape = (10, 10)
        slope        = np.full(shape, 45.0, dtype=np.float32)  # > 10°
        dte          = np.ones(shape,  dtype=bool)
        illumination = np.full(shape, 0.9,  dtype=np.float32)
        mask = build_candidate_mask(slope, dte, illumination, config)
        assert not np.any(mask), "Expected all False but got some True"

    def test_candidate_mask_none_pass_no_dte(self) -> None:
        """DTE=False everywhere → mask all False."""
        config = self._default_config()
        shape = (8, 8)
        slope        = np.full(shape, 5.0,  dtype=np.float32)
        dte          = np.zeros(shape, dtype=bool)              # all False
        illumination = np.full(shape, 0.9,  dtype=np.float32)
        mask = build_candidate_mask(slope, dte, illumination, config)
        assert not np.any(mask)

    def test_candidate_mask_none_pass_low_illumination(self) -> None:
        """Illumination below threshold everywhere → mask all False."""
        config = self._default_config()
        shape = (8, 8)
        slope        = np.full(shape, 5.0, dtype=np.float32)
        dte          = np.ones(shape,  dtype=bool)
        illumination = np.full(shape, 0.1, dtype=np.float32)   # < 0.7
        mask = build_candidate_mask(slope, dte, illumination, config)
        assert not np.any(mask)

    def test_candidate_mask_partial(self) -> None:
        """Only the top row passes → only top row is True."""
        config = self._default_config()
        shape = (5, 5)
        slope        = np.full(shape, 45.0, dtype=np.float32)
        slope[0, :]  = 5.0   # only row 0 has good slope
        dte          = np.ones(shape, dtype=bool)
        illumination = np.full(shape, 0.9, dtype=np.float32)
        mask = build_candidate_mask(slope, dte, illumination, config)
        assert np.all(mask[0, :]), "Expected top row to be True"
        assert not np.any(mask[1:, :]), "Expected remaining rows to be False"

    def test_candidate_mask_shape_mismatch_raises(self) -> None:
        """Inconsistent shapes must raise ValueError."""
        config = self._default_config()
        slope        = np.zeros((5, 5), dtype=np.float32)
        dte          = np.zeros((5, 5), dtype=bool)
        illumination = np.zeros((6, 5), dtype=np.float32)  # wrong shape
        with pytest.raises(ValueError, match="shape"):
            build_candidate_mask(slope, dte, illumination, config)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for extract_candidate_sites
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractCandidateSites:
    """Tests for ``mcda.extract_candidate_sites``."""

    def _config(self, **kwargs: float | int) -> Module2Config:
        defaults = dict(
            max_slope_deg=10.0,
            illumination_threshold=0.7,
            exposure_weight_illumination=0.6,
            exposure_weight_dte=0.4,
            peak_neighborhood_size=3,
            max_candidates=10,
        )
        defaults.update(kwargs)
        return Module2Config(**defaults)  # type: ignore[arg-type]

    def test_extract_candidates_empty(self) -> None:
        """Empty candidate mask must return an empty list without error."""
        config = self._config()
        shape = (10, 10)
        candidate_mask = np.zeros(shape, dtype=bool)
        exposure_score = np.zeros(shape, dtype=np.float32)
        slope          = np.zeros(shape, dtype=np.float32)
        dte            = np.zeros(shape, dtype=bool)
        illumination   = np.zeros(shape, dtype=np.float32)
        sites = extract_candidate_sites(
            candidate_mask, exposure_score, slope, dte, illumination, config
        )
        assert sites == []

    def test_extract_candidates_ranked(self) -> None:
        """Three clear peaks with distinct scores must be returned in correct rank order.

        Grid: 21×21 pixels, peak_neighborhood_size=3.
        Three peaks planted at (5,5), (10,10), (15,15) with scores 0.9, 0.7, 0.5.
        All other candidate pixels have score 0.2 (below peaks).
        Expected: sites returned ranked 1→2→3 by descending score.
        """
        config = self._config(peak_neighborhood_size=3, max_candidates=5)
        shape = (21, 21)

        # All pixels start as candidates
        candidate_mask = np.ones(shape, dtype=bool)
        slope          = np.full(shape, 3.0,  dtype=np.float32)
        dte            = np.ones(shape,  dtype=bool)
        illumination   = np.full(shape, 0.9,  dtype=np.float32)

        # Exposure score: background low, three peaks
        exposure_score = np.full(shape, 0.2, dtype=np.float32)
        peaks = [(5,  5,  0.9),
                 (10, 10, 0.7),
                 (15, 15, 0.5)]
        for (pr, pc, score) in peaks:
            exposure_score[pr, pc] = score

        sites = extract_candidate_sites(
            candidate_mask, exposure_score, slope, dte, illumination, config
        )

        assert len(sites) >= 3, f"Expected ≥ 3 sites; got {len(sites)}"

        # Verify top-3 ranks and descending score order
        top3 = sites[:3]
        assert top3[0].rank == 1
        assert top3[1].rank == 2
        assert top3[2].rank == 3
        assert top3[0].exposure_score >= top3[1].exposure_score
        assert top3[1].exposure_score >= top3[2].exposure_score

        # Verify the highest-score site is at (5, 5)
        assert top3[0].row == 5 and top3[0].col == 5, (
            f"Expected peak at (5,5), got ({top3[0].row}, {top3[0].col})"
        )

    def test_extract_candidates_respects_max_candidates(self) -> None:
        """Returned list length must be ≤ config.max_candidates."""
        config = self._config(peak_neighborhood_size=3, max_candidates=2)
        shape = (21, 21)
        candidate_mask = np.ones(shape, dtype=bool)
        exposure_score = np.random.default_rng(0).random(shape).astype(np.float32)
        slope          = np.full(shape, 3.0, dtype=np.float32)
        dte            = np.ones(shape, dtype=bool)
        illumination   = np.full(shape, 0.9, dtype=np.float32)
        sites = extract_candidate_sites(
            candidate_mask, exposure_score, slope, dte, illumination, config
        )
        assert len(sites) <= 2

    def test_extract_candidates_site_attributes(self) -> None:
        """Each returned CandidateSite must have consistent attribute values."""
        config = self._config(peak_neighborhood_size=3, max_candidates=5)
        shape = (15, 15)
        dte_arr   = np.ones(shape, dtype=bool)
        illum_arr = np.full(shape, 0.85, dtype=np.float32)
        slope_arr = np.full(shape, 4.0,  dtype=np.float32)
        candidate_mask = np.ones(shape, dtype=bool)
        exposure_score = compute_exposure_score(illum_arr, dte_arr, config)

        sites = extract_candidate_sites(
            candidate_mask, exposure_score, slope_arr, dte_arr, illum_arr, config
        )
        for site in sites:
            assert isinstance(site, CandidateSite)
            assert 0 <= site.row < shape[0]
            assert 0 <= site.col < shape[1]
            assert site.slope_deg >= 0.0
            assert site.rank >= 1
            assert 0.0 <= site.illumination_fraction <= 1.0
            assert 0.0 <= site.exposure_score <= 1.0

    def test_extract_candidates_non_candidate_pixels_excluded(self) -> None:
        """Non-candidate pixels must never appear in the output."""
        config = self._config(peak_neighborhood_size=3, max_candidates=10)
        shape = (15, 15)
        # Only top-left 5×5 are candidates
        candidate_mask = np.zeros(shape, dtype=bool)
        candidate_mask[:5, :5] = True

        exposure_score = np.random.default_rng(5).random(shape).astype(np.float32)
        # Give rest of image very high scores (should still be excluded)
        exposure_score[5:, 5:] = 99.0

        slope = np.full(shape, 3.0, dtype=np.float32)
        dte   = np.ones(shape, dtype=bool)
        illum = np.full(shape, 0.9, dtype=np.float32)

        sites = extract_candidate_sites(
            candidate_mask, exposure_score, slope, dte, illum, config
        )
        for site in sites:
            assert candidate_mask[site.row, site.col], (
                f"Site at ({site.row}, {site.col}) is outside the candidate mask."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests for make_synthetic_auxiliary
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeSyntheticAuxiliary:
    """Tests for ``terrain_analysis.make_synthetic_auxiliary``."""

    def test_synthetic_auxiliary_shapes(self) -> None:
        """Returned arrays must have the requested shape."""
        shape = (30, 40)
        dte, illum = make_synthetic_auxiliary(shape)
        assert dte.shape == shape
        assert illum.shape == shape

    def test_synthetic_auxiliary_dtypes(self) -> None:
        """DTE must be bool; illumination must be float32."""
        shape = (20, 20)
        dte, illum = make_synthetic_auxiliary(shape)
        assert dte.dtype == bool
        assert illum.dtype == np.float32

    def test_synthetic_auxiliary_illumination_range(self) -> None:
        """Illumination values must lie in [0, 1]."""
        shape = (50, 50)
        _, illum = make_synthetic_auxiliary(shape, seed=123)
        assert float(illum.min()) >= 0.0
        assert float(illum.max()) <= 1.0

    def test_synthetic_auxiliary_dte_fraction(self) -> None:
        """DTE true fraction should be close to the requested value."""
        shape = (200, 200)
        dte_frac = 0.65
        dte, _ = make_synthetic_auxiliary(shape, dte_true_fraction=dte_frac, seed=7)
        actual_frac = float(dte.mean())
        # Allow ±5% tolerance for random variation
        assert abs(actual_frac - dte_frac) < 0.05, (
            f"Expected DTE fraction ≈ {dte_frac:.2f}; got {actual_frac:.4f}"
        )
