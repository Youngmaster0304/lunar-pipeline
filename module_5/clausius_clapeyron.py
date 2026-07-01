"""
module_5/clausius_clapeyron.py
==============================
Sublimation kinetics model for water ice in lunar regolith using the
Clausius-Clapeyron equation and the Hertz-Knudsen evaporation flux formula.

When the drill bit heats the cryogenic regolith, any embedded water-ice
sublimates into the sealed sample chamber.  This module computes:
1. The equilibrium vapour pressure at the post-drill surface temperature via
   the integrated Clausius-Clapeyron relation.
2. The sublimation mass flux (kg/s) via the Hertz-Knudsen equation.
3. The accumulated vapour density inside the sealed chamber after a given
   exposure duration.

References
----------
- Clausius, R. (1850). Über die bewegende Kraft der Wärme. *Annalen der
  Physik*, 155(3), 368-397.
- Clapeyron, É. (1834). Mémoire sur la puissance motrice de la chaleur.
  *Journal de l'École Polytechnique*, 23, 153-190.
- Knudsen, M. (1909). Die maximale Verdampfungsgeschwindigkeit des Quecksilbers.
  *Annalen der Physik*, 334(1), 179-193.  (Hertz-Knudsen equation.)
- Murphy, D. M., & Koop, T. (2005). Review of the vapour pressures of ice and
  supercooled water for atmospheric applications. *Quarterly Journal of the
  Royal Meteorological Society*, 131(608), 1539-1565.
"""
from __future__ import annotations

import logging
import math

from .config import Module5Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants (SI, fixed — do not place in config)
# ---------------------------------------------------------------------------

_L_SUB: float = 2.83e6       # Latent heat of sublimation of water ice [J/kg]
_R_W: float = 461.5          # Specific gas constant for water vapour [J/(kg·K)]
_P_TRIPLE: float = 611.73    # Triple-point pressure of water [Pa]
_T_TRIPLE: float = 273.16    # Triple-point temperature of water [K]
_M_MOL: float = 0.018015     # Molar mass of water [kg/mol]
_R_UNIVERSAL: float = 8.314  # Universal gas constant [J/(mol·K)]


def compute_sublimation_rate(
    T_surface_k: float,
    P_sealed_pa: float,
    sealed_volume_m3: float,
    config: Module5Config,
) -> float:
    """Compute the water-ice sublimation mass flux at a given surface temperature.

    Step 1 — Equilibrium Vapour Pressure (Clausius-Clapeyron)
    ----------------------------------------------------------
    Integrating the Clausius-Clapeyron equation:

        dP/dT = L_sub · P / (R_w · T²)

    from the triple point (T_triple, P_triple) to temperature T gives the
    August-Roche-Magnus approximation (exact for constant L_sub):

        P_eq(T) = P_triple · exp[ (L_sub / R_w) · (1/T_triple − 1/T) ]

    where:
        L_sub    = 2.83×10⁶ J/kg  (latent heat of sublimation)
        R_w      = 461.5 J/(kg·K) (specific gas constant for water vapour)
        P_triple = 611.73 Pa      (water triple-point pressure)
        T_triple = 273.16 K       (water triple-point temperature)
        T        = T_surface_k [K]

    Step 2 — Sublimation Mass Flux (Hertz-Knudsen Equation)
    --------------------------------------------------------
    The net evaporative mass flux from a surface at temperature T with
    ambient vapour pressure P_sealed is:

        dm/dt = α · (P_eq − P_sealed) · √[ M / (2π·R·T) ] · A_surface

    where:
        α          = 1.0            (accommodation/sticking coefficient; unity
                                     for clean ice surface in vacuum)
        P_eq       — equilibrium vapour pressure [Pa] (Step 1)
        P_sealed   — current partial pressure of water vapour in the sealed
                     chamber [Pa]
        M          = 0.018015 kg/mol (molar mass of water)
        R          = 8.314 J/(mol·K) (universal gas constant)
        T          — surface temperature [K]
        A_surface  = config.drill_contact_area_m2 [m²]

    Net flux is set to zero if P_eq < P_sealed (condensation regime; not
    modelled here — we simply clamp dm/dt to 0).

    Parameters
    ----------
    T_surface_k : float
        Temperature of the ice-bearing regolith surface immediately after
        drilling.  Units: Kelvin.  Valid range: > 0.
    P_sealed_pa : float
        Current partial pressure of water vapour inside the sealed sampling
        chamber.  Units: Pascals.  Valid range: ≥ 0.
    sealed_volume_m3 : float
        Volume of the sealed chamber (used for logging only; actual accumulation
        is computed in :func:`compute_vapor_density`).
        Units: m³.  Valid range: > 0.
    config : Module5Config
        EFPI module configuration.

    Returns
    -------
    float
        Net sublimation mass flux dm/dt [kg/s].  Always ≥ 0 (condensation
        is clamped to zero).

    Raises
    ------
    ValueError
        If T_surface_k ≤ 0 (absolute zero or below is unphysical).
    ValueError
        If P_sealed_pa < 0 or sealed_volume_m3 ≤ 0.

    References
    ----------
    Knudsen (1909); Murphy & Koop (2005).

    Examples
    --------
    >>> from module_5.config import Module5Config
    >>> cfg = Module5Config()
    >>> dm_dt = compute_sublimation_rate(100.0, 0.0, cfg.sealed_volume_m3, cfg)
    >>> dm_dt > 0
    True
    """
    # --- Input validation -----------------------------------------------
    if T_surface_k <= 0:
        raise ValueError(
            f"T_surface_k must be > 0 K (absolute zero); got {T_surface_k}"
        )
    if P_sealed_pa < 0:
        raise ValueError(f"P_sealed_pa must be >= 0; got {P_sealed_pa}")
    if sealed_volume_m3 <= 0:
        raise ValueError(f"sealed_volume_m3 must be > 0; got {sealed_volume_m3}")

    # --- Step 1: equilibrium vapour pressure via Clausius-Clapeyron -----
    exponent = (_L_SUB / _R_W) * (1.0 / _T_TRIPLE - 1.0 / T_surface_k)
    P_eq = _P_TRIPLE * math.exp(exponent)

    logger.debug(
        "compute_sublimation_rate: T=%.2f K → P_eq=%.4e Pa", T_surface_k, P_eq
    )

    # --- Step 2: Hertz-Knudsen mass flux --------------------------------
    net_pressure_pa = P_eq - P_sealed_pa
    if net_pressure_pa <= 0:
        logger.debug(
            "P_eq (%.4e Pa) <= P_sealed (%.4e Pa): condensation regime, dm/dt = 0",
            P_eq, P_sealed_pa,
        )
        return 0.0

    # Thermal velocity factor: sqrt(M / (2π·R·T))
    thermal_factor = math.sqrt(_M_MOL / (2.0 * math.pi * _R_UNIVERSAL * T_surface_k))

    alpha = 1.0  # accommodation coefficient (clean ice, vacuum)
    A_surface = config.drill_contact_area_m2

    dm_dt = alpha * net_pressure_pa * thermal_factor * A_surface

    logger.info(
        "compute_sublimation_rate: T=%.2f K, P_eq=%.4e Pa, P_sealed=%.4e Pa, "
        "A=%.2e m² → dm/dt=%.4e kg/s",
        T_surface_k, P_eq, P_sealed_pa, A_surface, dm_dt,
    )
    return float(dm_dt)


def compute_vapor_density(
    sublimation_rate_kg_s: float,
    sealed_volume_m3: float,
    duration_s: float,
) -> float:
    """Compute the accumulated water-vapour density inside the sealed chamber.

    Model
    -----
    Assuming the sublimation rate is constant over the accumulation interval
    (valid for short durations where T and P_sealed change slowly):

        m_vapour = (dm/dt) · t

        ρ_vapour = m_vapour / V = (dm/dt · t) / V

    where:
        dm/dt  — sublimation_rate_kg_s [kg/s]
        t      — duration_s [s]
        V      — sealed_volume_m3 [m³]

    Parameters
    ----------
    sublimation_rate_kg_s : float
        Net sublimation mass flux from :func:`compute_sublimation_rate`.
        Units: kg/s.  Valid range: ≥ 0.
    sealed_volume_m3 : float
        Volume of the sealed sampling chamber.
        Units: m³.  Valid range: > 0.
    duration_s : float
        Duration over which vapour accumulates.
        Units: seconds.  Valid range: > 0.

    Returns
    -------
    float
        Vapour mass density ρ_vapour inside the sealed chamber.
        Units: kg/m³.

    Raises
    ------
    ValueError
        If *sublimation_rate_kg_s* < 0, *sealed_volume_m3* ≤ 0, or
        *duration_s* ≤ 0.

    Examples
    --------
    >>> compute_vapor_density(1e-9, 1e-5, 60.0)
    6e-03
    """
    if sublimation_rate_kg_s < 0:
        raise ValueError(
            f"sublimation_rate_kg_s must be >= 0; got {sublimation_rate_kg_s}"
        )
    if sealed_volume_m3 <= 0:
        raise ValueError(f"sealed_volume_m3 must be > 0; got {sealed_volume_m3}")
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0; got {duration_s}")

    vapor_mass_kg = sublimation_rate_kg_s * duration_s
    vapor_density = vapor_mass_kg / sealed_volume_m3

    logger.info(
        "compute_vapor_density: dm/dt=%.4e kg/s, t=%.1f s, V=%.2e m³ "
        "→ m_vapour=%.4e kg → ρ=%.4e kg/m³",
        sublimation_rate_kg_s, duration_s, sealed_volume_m3,
        vapor_mass_kg, vapor_density,
    )
    return float(vapor_density)
