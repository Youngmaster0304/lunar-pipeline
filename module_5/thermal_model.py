"""
module_5/thermal_model.py
=========================
Drill-regolith conductive heat transfer model for the EFPI ice-sensing module.

When the rover drill bit contacts regolith at cryogenic temperatures (≈ 25 K),
frictional and conductive heating sublimes any water-ice present in the sample.
This module computes the total heat transferred from the drill bit to the
regolith volume using Fourier's law of conduction, and propagates that heat
into a post-drill equilibrium temperature used by the sublimation model.

References
----------
- Fourier, J. B. J. (1822). *Théorie analytique de la chaleur*. Firmin Didot.
- Incropera, F. P., DeWitt, D. P., Bergman, T. L., & Lavine, A. S. (2007).
  *Fundamentals of Heat and Mass Transfer*, 6th ed. Wiley.
- Hemingway, B. S., Robie, R. A., & Wilson, W. H. (1973). Specific heats of
  lunar soils, basalt, and breccias from the Apollo 14, 15, and 16 landing
  sites. *Proceedings of the Lunar Science Conference*, 4, 2481-2487.
"""
from __future__ import annotations

import logging

from .config import Module5Config

logger = logging.getLogger(__name__)


def compute_drill_heat_transfer(config: Module5Config) -> float:
    """Compute the total conductive heat transferred from the drill bit to regolith.

    Governing Equation (Fourier's Law of Conduction — Incropera et al. 2007)
    -------------------------------------------------------------------------
    The steady-state heat flux through a solid medium of thickness *d* is:

        q = k · (T_drill − T_reg) / d          [W/m²]

    The total power transferred across contact area *A* is:

        P = q · A = k · A · (T_drill − T_reg) / d    [W]

    Integrating over the contact duration *t*:

        Q = P · t = k · A · (T_drill − T_reg) · t / d    [J]

    where:
        k         — config.drill_thermal_conductivity [W/(m·K)]
        A         — config.drill_contact_area_m2 [m²]
        T_drill   — config.drill_temp_k [K]
        T_reg     — config.regolith_temp_k [K]
        d         — config.drill_depth_m [m]  (conduction path length)
        t         — config.drill_contact_duration_s [s]

    This model assumes:
    - One-dimensional, planar heat flow (valid when contact radius ≫ cavity
      length, which holds for the 1 cm² bit area at 10 cm depth).
    - Constant bulk thermal conductivity (temperature-independent within the
      drilled interval).
    - No radiative or convective losses (valid in high-vacuum lunar environment).

    Parameters
    ----------
    config : Module5Config
        Validated EFPI module configuration.

    Returns
    -------
    float
        Total heat *Q* transferred from drill bit to regolith sample.
        Units: Joules.  Always positive (heat flows from hot drill to cold
        regolith).

    Raises
    ------
    ValueError
        If T_drill ≤ T_reg (heat cannot flow from cold to hot without work).
    ValueError
        If any required config parameter is non-positive (defensive check in
        addition to Module5Config.__post_init__).

    Examples
    --------
    >>> from module_5.config import Module5Config
    >>> cfg = Module5Config()
    >>> Q = compute_drill_heat_transfer(cfg)
    >>> round(Q, 2)
    48750.0
    """
    # --- defensive validation -------------------------------------------
    if config.drill_thermal_conductivity <= 0:
        raise ValueError(
            f"drill_thermal_conductivity must be > 0; "
            f"got {config.drill_thermal_conductivity}"
        )
    if config.drill_contact_area_m2 <= 0:
        raise ValueError(
            f"drill_contact_area_m2 must be > 0; "
            f"got {config.drill_contact_area_m2}"
        )
    if config.drill_depth_m <= 0:
        raise ValueError(f"drill_depth_m must be > 0; got {config.drill_depth_m}")
    if config.drill_contact_duration_s <= 0:
        raise ValueError(
            f"drill_contact_duration_s must be > 0; "
            f"got {config.drill_contact_duration_s}"
        )
    if config.drill_temp_k <= config.regolith_temp_k:
        raise ValueError(
            f"drill_temp_k ({config.drill_temp_k} K) must be strictly greater than "
            f"regolith_temp_k ({config.regolith_temp_k} K) for heat to flow "
            f"into the regolith."
        )
    # --- Fourier conduction -------------------------------------------
    delta_T = config.drill_temp_k - config.regolith_temp_k  # [K]
    Q = (
        config.drill_thermal_conductivity
        * config.drill_contact_area_m2
        * delta_T
        / config.drill_depth_m
        * config.drill_contact_duration_s
    )
    logger.info(
        "compute_drill_heat_transfer: k=%.3f W/(m·K), A=%.2e m², ΔT=%.1f K, "
        "d=%.3f m, t=%.1f s → Q=%.4e J",
        config.drill_thermal_conductivity,
        config.drill_contact_area_m2,
        delta_T,
        config.drill_depth_m,
        config.drill_contact_duration_s,
        Q,
    )
    return float(Q)


def compute_post_drill_temperature(
    Q_joules: float,
    regolith_mass_kg: float,
    specific_heat_j_kg_k: float = 750.0,
    initial_temp_k: float = 25.0,
) -> float:
    """Compute the post-drilling equilibrium temperature of the regolith sample.

    Governing Equation (Sensible heat — calorimetry)
    -------------------------------------------------
    Assuming all transferred heat *Q* raises the temperature of a regolith
    plug of mass *m* with specific heat *Cp*:

        ΔT = Q / (m · Cp)

    Final temperature:

        T_final = T_initial + ΔT

    where:
        Q         — total heat input [J] (from :func:`compute_drill_heat_transfer`)
        m         — regolith_mass_kg [kg]
        Cp        — specific_heat_j_kg_k [J/(kg·K)]
        T_initial — initial_temp_k [K]

    Default specific heat *Cp* = 750 J/(kg·K) is the representative value for
    bulk lunar regolith at cryogenic temperatures derived from Apollo lunar
    soil calorimetry (Hemingway et al. 1973).

    Parameters
    ----------
    Q_joules : float
        Total heat transferred into the regolith sample.
        Units: Joules.  Valid range: ≥ 0.
    regolith_mass_kg : float
        Mass of the regolith plug that absorbs the heat.
        Units: kg.  Valid range: > 0.
    specific_heat_j_kg_k : float, optional
        Specific heat capacity of lunar regolith.
        Units: J/(kg·K).  Default: 750 J/(kg·K) (Hemingway et al. 1973).
        Valid range: > 0.
    initial_temp_k : float, optional
        Initial temperature of the regolith sample before drilling.
        Units: Kelvin.  Default: 25 K (Lunar PSR ambient).
        Valid range: > 0.

    Returns
    -------
    float
        Post-drill equilibrium temperature *T_final* [K].

    Raises
    ------
    ValueError
        If *Q_joules* < 0, *regolith_mass_kg* ≤ 0,
        *specific_heat_j_kg_k* ≤ 0, or *initial_temp_k* ≤ 0.

    References
    ----------
    Hemingway, B. S., Robie, R. A., & Wilson, W. H. (1973). Specific heats of
    lunar soils, basalt, and breccias from the Apollo 14, 15, and 16 landing
    sites. *Proceedings of the Lunar Science Conference*, 4, 2481-2487.

    Examples
    --------
    >>> compute_post_drill_temperature(48750.0, 0.5)
    155.0
    """
    if Q_joules < 0:
        raise ValueError(f"Q_joules must be >= 0; got {Q_joules}")
    if regolith_mass_kg <= 0:
        raise ValueError(f"regolith_mass_kg must be > 0; got {regolith_mass_kg}")
    if specific_heat_j_kg_k <= 0:
        raise ValueError(
            f"specific_heat_j_kg_k must be > 0; got {specific_heat_j_kg_k}"
        )
    if initial_temp_k <= 0:
        raise ValueError(f"initial_temp_k must be > 0; got {initial_temp_k}")

    delta_T = Q_joules / (regolith_mass_kg * specific_heat_j_kg_k)
    T_final = initial_temp_k + delta_T

    logger.info(
        "compute_post_drill_temperature: Q=%.4e J, m=%.3f kg, "
        "Cp=%.1f J/(kg·K) → ΔT=%.2f K → T_final=%.2f K",
        Q_joules, regolith_mass_kg, specific_heat_j_kg_k, delta_T, T_final,
    )
    return float(T_final)
