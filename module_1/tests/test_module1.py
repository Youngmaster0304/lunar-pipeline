"""Pytest test suite for Module 1: Radar Polarimetric Decomposition (DFSAR).

All tests use purely synthetic NumPy arrays; no real satellite data files
are required.  The synthetic DFSARProduct factory is used for integration-
style tests, while polarimetry and CRIM tests build minimal arrays inline.

Run with::

    pytest module_1/tests/test_module1.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from module_1.config import Module1Config
from module_1.crim import _crim_sqrt_eps, invert_ice_fraction
from module_1.dfsar_reader import DFSARProduct, load_dfsar_product_synthetic
from module_1.polarimetry import build_ice_mask, compute_cpr, compute_dop


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_product(
    lh: np.ndarray,
    lv: np.ndarray,
    rh: np.ndarray,
    rv: np.ndarray,
) -> DFSARProduct:
    """Construct a minimal DFSARProduct from four arrays of the same shape."""
    shape = lh.shape
    return DFSARProduct(
        lh=lh.astype(np.complex64),
        lv=lv.astype(np.complex64),
        rh=rh.astype(np.complex64),
        rv=rv.astype(np.complex64),
        geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        crs="EPSG:4326",
        shape=shape,
    )


def _ones(shape: tuple[int, int] = (10, 10)) -> np.ndarray:
    return np.ones(shape, dtype=np.complex64)


def _zeros(shape: tuple[int, int] = (10, 10)) -> np.ndarray:
    return np.zeros(shape, dtype=np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — CPR > 1.0 for high same-sense power
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCPR:
    """Tests for ``compute_cpr``."""

    def test_cpr_synthetic_high_ice(self) -> None:
        """Ice anomaly with Im(LH·LV*) < 0 yields CPR > 1.0 (L-band only).

        Derivation (L-band, LHC transmit):
          LL = (LH − j·LV)/√2   (same-sense)
          LR = (LH + j·LV)/√2   (opposite-sense)
          |LL|² = ½(|LH|²+|LV|² − 2·Im(LH·LV*))
          |LR|² = ½(|LH|²+|LV|² + 2·Im(LH·LV*))
          CPR = |LL|² / |LR|²

        With LV = −j·LH (real LH, negative-imag LV):
          Im(LH·LV*) = Im(LH·(−j·LH)*) = Im(LH·j·LH*) = Im(j·|LH|²) = 1 > 0
          → |LL|² < |LR|² → CPR < 1

        With LV = +j·LH (real LH, positive-imag LV):
          Im(LH·LV*) = Im(LH·(j·LH)*) = Im(LH·(−j)·LH*) = Im(−j·|LH|²) = −1 < 0
          → |LL|² > |LR|² → CPR > 1 ✓
        """
        shape = (8, 8)
        config = Module1Config(cpr_band="L")

        lh = _ones(shape)
        lv = np.full(shape, 1j, dtype=np.complex64)   # +j → Im(LH·LV*) < 0

        product = _make_product(lh, lv, lh, lv)
        cpr = compute_cpr(product, config)

        assert cpr.dtype == np.float32
        assert cpr.shape == shape
        assert np.all(cpr > 1.0), f"Expected CPR > 1.0, got mean={cpr.mean():.4f}"
        _ = config  # silence linter

    def test_cpr_band_l_only(self) -> None:
        """CPR with cpr_band='L' should use only L-band channels."""
        shape = (5, 5)
        config = Module1Config(cpr_band="L")
        lh = _ones(shape)
        lv = _zeros(shape)
        # S-band channels set to garbage — should be ignored
        rh = np.full(shape, 999.0 + 0j, dtype=np.complex64)
        rv = np.full(shape, 999.0 + 0j, dtype=np.complex64)
        product = _make_product(lh, lv, rh, rv)
        cpr = compute_cpr(product, config)
        assert cpr.shape == shape
        assert cpr.dtype == np.float32

    def test_cpr_band_s_only(self) -> None:
        """CPR with cpr_band='S' should use only S-band channels."""
        shape = (5, 5)
        config = Module1Config(cpr_band="S")
        lh = np.full(shape, 999.0 + 0j, dtype=np.complex64)
        lv = np.full(shape, 999.0 + 0j, dtype=np.complex64)
        rh = _ones(shape)
        rv = _zeros(shape)
        product = _make_product(lh, lv, rh, rv)
        cpr = compute_cpr(product, config)
        assert cpr.shape == shape
        assert cpr.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 & 3 — DOP limits (fully polarised and unpolarised)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeDOP:
    """Tests for ``compute_dop``."""

    def test_dop_fully_polarized(self) -> None:
        """A fully polarised signal concentrates power in one channel → DOP ≈ 1.

        If all energy is in LH and RH (zero LV and RV):
          T_L[0,0] = 1, T_L[1,1] = 0, T_L[0,1] = 0
          det(T_L) = 0, trace(T_L) = 1
          DOP_L = √(1 − 4·0/1²) = 1.0
        Same for S-band. Final DOP = 1.0.
        """
        shape = (6, 6)
        lh = _ones(shape)
        lv = _zeros(shape)
        rh = _ones(shape)
        rv = _zeros(shape)
        product = _make_product(lh, lv, rh, rv)
        dop = compute_dop(product)
        assert dop.dtype == np.float32
        assert dop.shape == shape
        np.testing.assert_allclose(dop, 1.0, atol=1e-4)

    def test_dop_unpolarized(self) -> None:
        """Uncorrelated HH/VV channels → DOP ≈ 0 (with spatial averaging).

        For a single-look 2×2 coherency matrix:
          det(⟨T⟩) = |H|²·|V|² − |H·V*|² = 0   (algebraic identity)
          → DOP = 1 for all purely deterministic (single-look) measurements.

        Spatial averaging (multi-looking, window_size > 1) breaks the
        degeneracy: for independent H and V, the averaged off-diagonal
        term ⟨H·V*⟩ → 0, while ⟨|H|²⟩·⟨|V|²⟩ > 0, giving DOP → 0.
        """
        rng = np.random.default_rng(0)
        shape = (200, 200)
        lh = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        lv = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        rh = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        rv = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        product = _make_product(lh, lv, rh, rv)

        # With multi-looking (default window_size=5), DOP should approach 0
        dop = compute_dop(product, window_size=5)
        assert float(np.mean(dop)) < 0.3, (
            f"Expected mean DOP close to 0 with multi-looking; got {np.mean(dop):.4f}"
        )

        # Without multi-looking (window_size=1), DOP = 1 for all pixels
        dop_single = compute_dop(product, window_size=1)
        np.testing.assert_allclose(dop_single, 1.0, atol=1e-4)

    def test_dop_range(self) -> None:
        """DOP must always be in [0, 1] for any input."""
        rng = np.random.default_rng(7)
        shape = (50, 50)
        lh = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        lv = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        rh = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        rv = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        product = _make_product(lh, lv, rh, rv)
        dop = compute_dop(product)
        assert np.all(dop >= 0.0), "DOP contains values < 0"
        assert np.all(dop <= 1.0 + 1e-5), "DOP contains values > 1"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Ice mask thresholding
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildIceMask:
    """Tests for ``build_ice_mask``."""

    def test_ice_mask_threshold(self) -> None:
        """ice_mask True only where CPR > threshold AND DOP < threshold."""
        config = Module1Config(cpr_threshold=1.0, dop_threshold=0.13)
        shape = (4, 4)
        # Quadrant setup:
        #   top-left:  CPR=2.0, DOP=0.05 → ice (True)
        #   top-right: CPR=2.0, DOP=0.50 → no ice (DOP too high)
        #   bot-left:  CPR=0.5, DOP=0.05 → no ice (CPR too low)
        #   bot-right: CPR=0.5, DOP=0.50 → no ice
        cpr = np.array(
            [[2.0, 2.0, 0.5, 0.5],
             [2.0, 2.0, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5]],
            dtype=np.float32,
        )
        dop = np.array(
            [[0.05, 0.50, 0.05, 0.50],
             [0.05, 0.50, 0.05, 0.50],
             [0.05, 0.50, 0.05, 0.50],
             [0.05, 0.50, 0.05, 0.50]],
            dtype=np.float32,
        )
        mask = build_ice_mask(cpr, dop, config)
        assert mask.dtype == bool
        # Columns 0 and 2 have DOP=0.05; columns 1 and 3 have DOP=0.50
        # Rows 0-1: CPR=2.0; rows 2-3: CPR=0.5
        expected = np.array(
            [[True,  False, False, False],
             [True,  False, False, False],
             [False, False, False, False],
             [False, False, False, False]],
        )
        np.testing.assert_array_equal(mask, expected)

    def test_ice_mask_shape_mismatch_raises(self) -> None:
        """Mismatched CPR/DOP shapes must raise ValueError."""
        config = Module1Config()
        cpr = np.ones((5, 5), dtype=np.float32)
        dop = np.ones((4, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            build_ice_mask(cpr, dop, config)

    def test_ice_mask_all_ice(self) -> None:
        """When all pixels satisfy both conditions, mask is all True."""
        config = Module1Config(cpr_threshold=0.5, dop_threshold=0.8)
        cpr = np.full((6, 6), 1.0, dtype=np.float32)
        dop = np.full((6, 6), 0.1, dtype=np.float32)
        mask = build_ice_mask(cpr, dop, config)
        assert np.all(mask)

    def test_ice_mask_no_ice(self) -> None:
        """When no pixel satisfies both conditions, mask is all False."""
        config = Module1Config(cpr_threshold=2.0, dop_threshold=0.05)
        cpr = np.full((6, 6), 0.5, dtype=np.float32)
        dop = np.full((6, 6), 0.5, dtype=np.float32)
        mask = build_ice_mask(cpr, dop, config)
        assert not np.any(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — CRIM round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestCRIM:
    """Tests for CRIM inversion functions."""

    def test_crim_roundtrip(self) -> None:
        """Forward CRIM at f_ice=0.3, then invert → recovered f_ice within 0.05.

        Forward model:
          √ε_mix = 0.3·√ε_ice + (0.6−0.3)·√ε_reg + 0.4·√ε_vac
          ε_mix_mag = |√ε_mix|²

        Inversion is done via ``invert_ice_fraction`` on a 1×1 synthetic
        backscatter map constructed to produce exactly ε_mix_mag.
        """
        config = Module1Config()
        f_ice_true = 0.3
        solid = 1.0 - config.crim_porosity   # 0.6
        vac   = config.crim_porosity          # 0.4

        # Forward CRIM
        sqrt_eps_mix = _crim_sqrt_eps(
            f_ice_true,
            config.crim_eps_ice,
            config.crim_eps_regolith,
            config.crim_eps_vacuum,
            solid,
            vac,
        )
        eps_mix_target = abs(sqrt_eps_mix) ** 2  # scalar float

        # Back-calculate the backscatter_db that produces eps_mix_target:
        # eps_mix = slope * sigma_lin + intercept  →  sigma_lin = (eps − intercept) / slope
        sigma_lin = (eps_mix_target - config.crim_sigma_to_eps_intercept) / config.crim_sigma_to_eps_slope
        sigma_lin = max(sigma_lin, 1e-9)
        backscatter_db = np.full((1, 1), 10.0 * np.log10(sigma_lin), dtype=np.float32)

        f_ice_recovered = invert_ice_fraction(backscatter_db, config)

        assert f_ice_recovered.shape == (1, 1)
        recovered = float(f_ice_recovered[0, 0])
        assert abs(recovered - f_ice_true) < 0.05, (
            f"CRIM round-trip failed: true={f_ice_true:.4f} recovered={recovered:.4f}"
        )

    def test_crim_zero_ice(self) -> None:
        """Forward at f_ice=0.0 → inversion recovers ≈ 0."""
        config = Module1Config()
        solid = 1.0 - config.crim_porosity
        vac   = config.crim_porosity
        f_ice_true = 0.0
        sqrt_eps_mix = _crim_sqrt_eps(
            f_ice_true,
            config.crim_eps_ice,
            config.crim_eps_regolith,
            config.crim_eps_vacuum,
            solid,
            vac,
        )
        eps_mix_target = abs(sqrt_eps_mix) ** 2
        sigma_lin = max(
            (eps_mix_target - config.crim_sigma_to_eps_intercept) / config.crim_sigma_to_eps_slope,
            1e-9,
        )
        backscatter_db = np.full((1, 1), 10.0 * np.log10(sigma_lin), dtype=np.float32)
        f_ice_recovered = invert_ice_fraction(backscatter_db, config)
        assert abs(float(f_ice_recovered[0, 0]) - f_ice_true) < 0.05

    def test_crim_max_ice(self) -> None:
        """Forward at f_ice=solid_fraction → inversion recovers ≈ solid_fraction."""
        config = Module1Config()
        solid = 1.0 - config.crim_porosity
        vac   = config.crim_porosity
        f_ice_true = solid
        sqrt_eps_mix = _crim_sqrt_eps(
            f_ice_true,
            config.crim_eps_ice,
            config.crim_eps_regolith,
            config.crim_eps_vacuum,
            solid,
            vac,
        )
        eps_mix_target = abs(sqrt_eps_mix) ** 2
        sigma_lin = max(
            (eps_mix_target - config.crim_sigma_to_eps_intercept) / config.crim_sigma_to_eps_slope,
            1e-9,
        )
        backscatter_db = np.full((1, 1), 10.0 * np.log10(sigma_lin), dtype=np.float32)
        f_ice_recovered = invert_ice_fraction(backscatter_db, config)
        assert abs(float(f_ice_recovered[0, 0]) - f_ice_true) < 0.05

    def test_crim_invalid_1d_raises(self) -> None:
        """invert_ice_fraction must raise ValueError for 1-D input."""
        config = Module1Config()
        with pytest.raises(ValueError, match="2-D"):
            invert_ice_fraction(np.array([-15.0], dtype=np.float32), config)

    def test_crim_nonfinite_raises(self) -> None:
        """invert_ice_fraction must raise ValueError for NaN/Inf input."""
        config = Module1Config()
        arr = np.full((3, 3), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            invert_ice_fraction(arr, config)

    def test_ice_volume_basic(self) -> None:
        """compute_ice_volume returns expected volume for known f_ice."""
        from module_1.crim import compute_ice_volume
        f_ice = np.full((10, 10), 0.3, dtype=np.float32)
        vol = compute_ice_volume(f_ice, pixel_area_m2=100.0, depth_m=5.0)
        expected = 10 * 10 * 0.3 * 100.0 * 5.0  # = 15000.0
        assert abs(vol - expected) < 1e-2, f"Expected {expected}, got {vol}"

    def test_ice_volume_zero(self) -> None:
        """Zero ice fraction gives zero volume."""
        from module_1.crim import compute_ice_volume
        f_ice = np.zeros((5, 5), dtype=np.float32)
        vol = compute_ice_volume(f_ice, pixel_area_m2=100.0)
        assert vol == 0.0

    def test_crim_output_range(self) -> None:
        """All returned f_ice values must lie in [0, solid_fraction]."""
        config = Module1Config()
        solid = 1.0 - config.crim_porosity
        rng = np.random.default_rng(1)
        backscatter_db = rng.uniform(-30.0, -5.0, size=(10, 10)).astype(np.float32)
        f_ice = invert_ice_fraction(backscatter_db, config)
        assert np.all(f_ice >= 0.0), "f_ice contains negative values"
        assert np.all(f_ice <= solid + 1e-4), "f_ice exceeds solid fraction"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Synthetic loader
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSynthetic:
    """Tests for ``load_dfsar_product_synthetic``."""

    def test_load_synthetic(self) -> None:
        """load_dfsar_product_synthetic() should return a valid DFSARProduct."""
        shape = (50, 60)
        product = load_dfsar_product_synthetic(shape=shape, seed=0)
        assert isinstance(product, DFSARProduct)
        assert product.shape == shape
        for name, arr in [("lh", product.lh), ("lv", product.lv),
                          ("rh", product.rh), ("rv", product.rv)]:
            assert arr.shape == shape, f"{name} has wrong shape: {arr.shape}"
            assert np.iscomplexobj(arr), f"{name} is not complex"
            assert not np.all(arr == 0), f"{name} is all zeros"

    def test_load_synthetic_default_shape(self) -> None:
        """Default shape is (100, 100)."""
        product = load_dfsar_product_synthetic()
        assert product.shape == (100, 100)

    def test_load_synthetic_geotransform(self) -> None:
        """geotransform must be a 6-tuple of floats."""
        product = load_dfsar_product_synthetic()
        assert len(product.geotransform) == 6

    def test_load_synthetic_ice_anomaly_region(self) -> None:
        """Top-left quadrant should have higher CPR than bottom-right (ice anomaly)."""
        product = load_dfsar_product_synthetic(shape=(100, 100), seed=42)
        config = Module1Config(cpr_band="both")
        cpr = compute_cpr(product, config)
        # Top-left quadrant (rows 0:50, cols 0:50) → ice anomaly
        cpr_ice = cpr[:50, :50]
        # Bottom-right quadrant (rows 50:, cols 50:) → background
        cpr_bg = cpr[50:, 50:]
        assert np.median(cpr_ice) > np.median(cpr_bg), (
            "Expected ice-anomaly quadrant to have higher median CPR than background."
        )
