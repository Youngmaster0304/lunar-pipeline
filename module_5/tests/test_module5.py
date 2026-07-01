"""
module_5/tests/test_module5.py
===============================
Complete pytest test suite for Module 5: EFPI In-Situ Ice Sensing.

Tests cover:
- EFPI fringe-order calculation against known analytical values.
- Monotonicity of fringe shift with increasing vapour density.
- Round-trip ice density inversion (fringe shift → ice %).
- Fourier heat-transfer formula verification against hand-calculated values.
- Validation error for physically impossible thermal configuration.
- Clausius-Clapeyron sublimation rate positivity at valid temperature.
- Input validation for sub-zero temperature.
- Linear scaling of vapour density with accumulation time.

All tests use default Module5Config unless a specific override is needed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from module_5.clausius_clapeyron import compute_sublimation_rate, compute_vapor_density
from module_5.config import Module5Config
from module_5.efpi_model import EFPIModel
from module_5.thermal_model import compute_drill_heat_transfer, compute_post_drill_temperature

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(name="cfg")
def default_config() -> Module5Config:
    """Return a default Module5Config with nominal PSR parameters."""
    return Module5Config()


@pytest.fixture(name="efpi")
def default_efpi(cfg: Module5Config) -> EFPIModel:
    """Return an EFPIModel built from the default config."""
    return EFPIModel(cfg)


# ---------------------------------------------------------------------------
# Test 1: Fringe order at nominal cavity length
# ---------------------------------------------------------------------------


def test_fringe_order_nominal(efpi: EFPIModel, cfg: Module5Config) -> None:
    """Fringe order at the nominal 50 µm cavity should equal 2*n*L/λ ≈ 64.516.

    Analytical check:
        m = 2 * 1.0 * 50e-6 / 1550e-9
          = 100e-6 / 1550e-9
          ≈ 64.516...
    """
    expected_m = 2.0 * 1.0 * 50e-6 / 1550e-9  # ≈ 64.516
    computed_m = efpi.fringe_order(50e-6)
    assert math.isclose(
        computed_m, expected_m, rel_tol=1e-9
    ), f"Expected m ≈ {expected_m:.6f}, got {computed_m:.6f}"
    # Verify the cached _m0 matches as well
    assert math.isclose(efpi._m0, expected_m, rel_tol=1e-9)


def test_fringe_order_scales_linearly_with_cavity(efpi: EFPIModel) -> None:
    """Doubling the cavity length should exactly double the fringe order."""
    m1 = efpi.fringe_order(50e-6)
    m2 = efpi.fringe_order(100e-6)
    assert math.isclose(m2, 2.0 * m1, rel_tol=1e-12)


def test_fringe_order_invalid_cavity(efpi: EFPIModel) -> None:
    """Non-positive cavity length must raise ValueError."""
    with pytest.raises(ValueError, match="cavity_length_m must be > 0"):
        efpi.fringe_order(0.0)
    with pytest.raises(ValueError, match="cavity_length_m must be > 0"):
        efpi.fringe_order(-1e-6)


# ---------------------------------------------------------------------------
# Test 2: Fringe shift monotone with increasing vapour density
# ---------------------------------------------------------------------------


def test_fringe_shift_monotone_humidity(efpi: EFPIModel) -> None:
    """Fringe shift must increase strictly monotonically as ρ_vapour increases.

    Physical basis: higher water-vapour density increases n, which increases
    the optical path length 2nL, which increases the fringe order and therefore
    the fringe shift relative to the vacuum reference.
    """
    densities = np.linspace(0.0, 0.1, 50)  # 0 to 0.1 kg/m³
    shifts = np.array([efpi.fringe_shift_from_humidity(float(rho)) for rho in densities])

    # Strictly monotone increasing
    diffs = np.diff(shifts)
    assert np.all(diffs > 0), (
        f"Fringe shift is not strictly monotone increasing; "
        f"min diff = {diffs.min():.4e}"
    )


def test_fringe_shift_zero_at_vacuum(efpi: EFPIModel) -> None:
    """At zero vapour density the fringe shift must be exactly zero."""
    delta_m = efpi.fringe_shift_from_humidity(0.0)
    assert delta_m == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Test 3: Ice density round-trip inversion
# ---------------------------------------------------------------------------


def test_ice_density_inversion(efpi: EFPIModel, cfg: Module5Config) -> None:
    """Compute fringe shift from a known vapour density, invert → close match.

    Round-trip:
        ρ_known → Δm (via fringe_shift_from_humidity)
               → ρ_recovered (via infer_ice_density → calibration inversion)

    Since infer_ice_density uses the linear calibration while
    fringe_shift_from_humidity uses physics, we test the *calibrated*
    round-trip:
        Δm_cal = slope * ρ + intercept  →  ρ_recovered = (Δm - intercept) / slope
    """
    # Use calibration directly for a clean round-trip test
    rho_known = 0.05  # kg/m³
    # Compute Δm using calibration forward model
    delta_m_cal = (
        cfg.fringe_to_density_slope * rho_known + cfg.fringe_to_density_intercept
    )
    # Invert via EFPIModel
    ice_pct = efpi.infer_ice_density(delta_m_cal)

    # rho_recovered = ice_pct * 100.0 / 100.0  (scaling in infer_ice_density)
    rho_recovered = ice_pct * 100.0 / 100.0  # → same as rho_known * (100 / _rho_max)
    # _rho_max = 100.0, so ice_pct = rho_known * 100 / 100 = rho_known (in %)
    # rho_recovered should equal rho_known
    assert ice_pct == pytest.approx(rho_known * 100.0 / 100.0, rel=1e-9), (
        f"Round-trip failed: expected {rho_known:.4f}, got {rho_recovered:.4f}"
    )


def test_ice_density_clamp_zero(efpi: EFPIModel) -> None:
    """Negative fringe shift must clamp ice percentage to 0, not go negative."""
    ice_pct = efpi.infer_ice_density(-999.0)
    assert ice_pct == pytest.approx(0.0, abs=1e-12)


def test_ice_density_clamp_hundred(efpi: EFPIModel, cfg: Module5Config) -> None:
    """Extremely large fringe shift must clamp ice percentage to 100 %."""
    # fringe shift large enough to exceed ρ_max = 100 kg/m³
    massive_shift = cfg.fringe_to_density_slope * 1e6 + cfg.fringe_to_density_intercept
    ice_pct = efpi.infer_ice_density(massive_shift)
    assert ice_pct == pytest.approx(100.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 4: Heat transfer Fourier formula — exact numerical check
# ---------------------------------------------------------------------------


def test_heat_transfer_fourier(cfg: Module5Config) -> None:
    """Verify Q = k * A * (T_drill - T_reg) / d * t against hand calculation.

    Using default config:
        k   = 0.5  W/(m·K)
        A   = 1e-4 m²
        ΔT  = 350.0 - 25.0 = 325.0 K
        d   = 0.1 m
        t   = 30.0 s

    Q = 0.5 * 1e-4 * 325.0 / 0.1 * 30.0
      = 0.5 * 1e-4 * 3250.0 * 30.0
      = 0.5 * 1e-4 * 97500.0
      = 0.5 * 9.75
      = 4.875  J
    """
    expected_Q = (
        cfg.drill_thermal_conductivity
        * cfg.drill_contact_area_m2
        * (cfg.drill_temp_k - cfg.regolith_temp_k)
        / cfg.drill_depth_m
        * cfg.drill_contact_duration_s
    )
    computed_Q = compute_drill_heat_transfer(cfg)
    assert math.isclose(
        computed_Q, expected_Q, rel_tol=1e-12
    ), f"Expected Q={expected_Q:.6f} J, got {computed_Q:.6f} J"
    # Confirm the hand-calculated value
    assert math.isclose(computed_Q, 4.875, rel_tol=1e-9)


def test_heat_transfer_scales_with_contact_time() -> None:
    """Doubling contact time must exactly double Q."""
    cfg1 = Module5Config(drill_contact_duration_s=30.0)
    cfg2 = Module5Config(drill_contact_duration_s=60.0)
    Q1 = compute_drill_heat_transfer(cfg1)
    Q2 = compute_drill_heat_transfer(cfg2)
    assert math.isclose(Q2, 2.0 * Q1, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Test 5: Heat transfer raises ValueError for T_drill <= T_reg
# ---------------------------------------------------------------------------


def test_heat_transfer_invalid() -> None:
    """T_drill <= T_reg must raise ValueError (heat cannot flow uphill)."""
    # Equal temperatures
    cfg_equal = Module5Config(drill_temp_k=25.0, regolith_temp_k=25.0)
    with pytest.raises(ValueError, match="must be strictly greater than"):
        compute_drill_heat_transfer(cfg_equal)

    # Drill colder than regolith
    cfg_cold = Module5Config(drill_temp_k=10.0, regolith_temp_k=25.0)
    with pytest.raises(ValueError, match="must be strictly greater than"):
        compute_drill_heat_transfer(cfg_cold)


# ---------------------------------------------------------------------------
# Test 6: Clausius-Clapeyron — positive sublimation rate at T > 0
# ---------------------------------------------------------------------------


def test_clausius_clapeyron_positive_rate(cfg: Module5Config) -> None:
    """At T = 100 K (well above absolute zero) the sublimation rate must be > 0.

    Even at 100 K (still far below triple point), the equilibrium vapour
    pressure is negligibly small but positive, and with P_sealed = 0 Pa the
    Hertz-Knudsen flux is strictly positive.
    """
    dm_dt = compute_sublimation_rate(100.0, 0.0, cfg.sealed_volume_m3, cfg)
    assert dm_dt > 0, f"Expected dm/dt > 0 at 100 K, got {dm_dt}"


def test_clausius_clapeyron_rate_increases_with_temperature(cfg: Module5Config) -> None:
    """Sublimation rate must increase with surface temperature.

    Higher T → higher P_eq → higher flux (for fixed P_sealed = 0).
    """
    rates = [compute_sublimation_rate(T, 0.0, cfg.sealed_volume_m3, cfg)
             for T in [50.0, 100.0, 150.0, 200.0]]
    assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1)), (
        f"Sublimation rates not monotonically increasing: {rates}"
    )


# ---------------------------------------------------------------------------
# Test 7: Clausius-Clapeyron raises ValueError for T <= 0
# ---------------------------------------------------------------------------


def test_clausius_clapeyron_invalid_temp(cfg: Module5Config) -> None:
    """T_surface_k = 0 K must raise ValueError."""
    with pytest.raises(ValueError, match="absolute zero"):
        compute_sublimation_rate(0.0, 0.0, cfg.sealed_volume_m3, cfg)

    with pytest.raises(ValueError, match="absolute zero"):
        compute_sublimation_rate(-50.0, 0.0, cfg.sealed_volume_m3, cfg)


# ---------------------------------------------------------------------------
# Test 8: Vapour density scales linearly with accumulation time
# ---------------------------------------------------------------------------


def test_vapor_density_scales_with_time() -> None:
    """Doubling the accumulation duration must exactly double vapour density."""
    dm_dt = 1.234e-9   # kg/s (arbitrary)
    V = 1e-5           # m³

    rho_1 = compute_vapor_density(dm_dt, V, 60.0)
    rho_2 = compute_vapor_density(dm_dt, V, 120.0)

    assert math.isclose(rho_2, 2.0 * rho_1, rel_tol=1e-12), (
        f"Expected rho_2 = 2 * rho_1, got rho_1={rho_1:.4e}, rho_2={rho_2:.4e}"
    )


def test_vapor_density_zero_rate() -> None:
    """Zero sublimation rate must yield zero vapour density regardless of time."""
    rho = compute_vapor_density(0.0, 1e-5, 3600.0)
    assert rho == pytest.approx(0.0, abs=1e-20)


def test_vapor_density_invalid_volume() -> None:
    """Non-positive sealed volume must raise ValueError."""
    with pytest.raises(ValueError, match="sealed_volume_m3 must be > 0"):
        compute_vapor_density(1e-9, 0.0, 60.0)


def test_vapor_density_invalid_duration() -> None:
    """Non-positive duration must raise ValueError."""
    with pytest.raises(ValueError, match="duration_s must be > 0"):
        compute_vapor_density(1e-9, 1e-5, 0.0)


# ---------------------------------------------------------------------------
# Test: Post-drill temperature calorimetry
# ---------------------------------------------------------------------------


def test_post_drill_temperature_basic() -> None:
    """Check post-drill temperature against hand calculation.

    Q = 4.875 J, m = 0.5 kg, Cp = 750 J/(kg·K), T0 = 25 K
    ΔT = 4.875 / (0.5 * 750) = 4.875 / 375 = 0.013 K
    T_final = 25.013 K
    """
    Q = 4.875
    T_final = compute_post_drill_temperature(Q, 0.5, 750.0, 25.0)
    expected = 25.0 + 4.875 / (0.5 * 750.0)
    assert math.isclose(T_final, expected, rel_tol=1e-12)


def test_post_drill_temperature_default_cp() -> None:
    """Verify default Cp = 750 J/(kg·K) is applied correctly."""
    Q = 750.0   # J
    m = 1.0     # kg
    # ΔT = 750 / (1.0 * 750) = 1 K; T_final = 26 K from T0 = 25 K
    T_final = compute_post_drill_temperature(Q, m)
    assert math.isclose(T_final, 26.0, rel_tol=1e-12)


def test_post_drill_temperature_invalid_mass() -> None:
    """Zero mass must raise ValueError."""
    with pytest.raises(ValueError, match="regolith_mass_kg must be > 0"):
        compute_post_drill_temperature(100.0, 0.0)


# ---------------------------------------------------------------------------
# Test: Module-level infer_ice_density wrapper
# ---------------------------------------------------------------------------


def test_module_level_infer_ice_density(cfg: Module5Config) -> None:
    """Module-level infer_ice_density convenience function must be callable."""
    from module_5 import infer_ice_density
    # Zero fringe shift → 0 % ice (intercept = 0)
    ice_pct = infer_ice_density(0.0, cfg)
    assert ice_pct == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Test: Config validation
# ---------------------------------------------------------------------------


def test_config_invalid_cavity_length() -> None:
    """Negative or zero cavity length must raise ValueError in config."""
    with pytest.raises(ValueError, match="efpi_cavity_length_m"):
        Module5Config(efpi_cavity_length_m=-1e-6)


def test_config_invalid_refractive_index() -> None:
    """Refractive index below 1.0 must raise ValueError in config."""
    with pytest.raises(ValueError, match="efpi_n_gap"):
        Module5Config(efpi_n_gap=0.5)


def test_config_zero_slope_raises() -> None:
    """Zero fringe_to_density_slope must raise ValueError (inversion undefined)."""
    with pytest.raises(ValueError, match="fringe_to_density_slope"):
        Module5Config(fringe_to_density_slope=0.0)


def test_config_defaults_are_valid() -> None:
    """The default Module5Config must construct without raising any exception."""
    cfg = Module5Config()
    assert cfg.efpi_cavity_length_m == pytest.approx(50e-6)
    assert cfg.efpi_wavelength_m == pytest.approx(1550e-9)
    assert cfg.regolith_temp_k == pytest.approx(25.0)
