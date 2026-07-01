"""
module_5/efpi_model.py
======================
Extrinsic Fabry-Pérot Interferometer (EFPI) sensor model for in-situ detection
of water-ice vapour in lunar regolith samples.

An EFPI consists of two partially reflective surfaces separated by a gap of
length L.  Interference between light reflected from the two surfaces produces
a fringe pattern whose order *m* shifts when the optical path length 2·n·L
changes — either because the gap closes (pressure/mechanical stimulus) or
because the refractive index *n* of the gap medium changes (vapour ingress).

This module implements the optical transfer function from cavity length to
fringe order, the refractive-index sensitivity to water-vapour density, and
the calibrated inversion from fringe shift to ice-mass percentage.

References
----------
- Rao, Y. J. (1999). Recent progress in applications of in-fibre Bragg grating
  sensors. *Optics and Lasers in Engineering*, 31(4), 297-324.  (EFPI fringe
  order equation; cited as Rao 1999 throughout.)
- Edlén, B. (1966). The refractive index of air. *Metrologia*, 2(2), 71-80.
  (Simplified extension used here for water vapour at 1550 nm.)
- Hernandez, G. (1988). *Fabry-Perot Interferometers*. Cambridge University
  Press.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .config import Module5Config

logger = logging.getLogger(__name__)


class EFPIModel:
    """Optical model of an Extrinsic Fabry-Pérot Interferometer.

    The model encapsulates:
    1. Fringe-order calculation from cavity length.
    2. Cavity-length inference from observed fringe shift.
    3. Refractive-index sensitivity to water-vapour density.
    4. Fringe-shift calculation due to humidity change.
    5. Ice-density inversion from measured fringe shift.

    Parameters
    ----------
    config : Module5Config
        EFPI and calibration configuration.

    Attributes
    ----------
    config : Module5Config
        Stored reference to the configuration.
    _m0 : float
        Reference fringe order computed from the nominal cavity length at
        construction time.  Cached to avoid redundant computation.
    """

    def __init__(self, config: Module5Config) -> None:
        """Initialise the EFPI model and cache the reference fringe order.

        Parameters
        ----------
        config : Module5Config
            Validated configuration object.
        """
        self.config: Module5Config = config
        # Cache reference fringe order for the nominal cavity length
        self._m0: float = self.fringe_order(config.efpi_cavity_length_m)
        logger.debug(
            "EFPIModel initialised: L0=%.3e m, λ=%.3e m, n=%.4f → m0=%.4f",
            config.efpi_cavity_length_m,
            config.efpi_wavelength_m,
            config.efpi_n_gap,
            self._m0,
        )

    # ------------------------------------------------------------------
    # Core EFPI optics
    # ------------------------------------------------------------------

    def fringe_order(self, cavity_length_m: float) -> float:
        """Compute the EFPI interference fringe order for a given cavity length.

        Governing Equation (Rao 1999, Eq. 1)
        --------------------------------------
        Constructive interference occurs when the round-trip optical path
        equals an integer multiple of the probe wavelength:

            2 · n · L = m · λ

        Solving for m:

            m = (2 · n · L) / λ

        where:
            n  — refractive index of the gap medium (dimensionless)
            L  — cavity length [m]
            λ  — probe laser wavelength [m]
            m  — fringe order (dimensionless; need not be an integer in
                  practice because the cavity length varies continuously)

        Parameters
        ----------
        cavity_length_m : float
            Physical length of the EFPI air gap.
            Units: metres.  Valid range: > 0.

        Returns
        -------
        float
            Fringe order *m* (dimensionless).

        Raises
        ------
        ValueError
            If *cavity_length_m* ≤ 0.

        References
        ----------
        Rao, Y. J. (1999). Recent progress in applications of in-fibre Bragg
        grating sensors. *Optics and Lasers in Engineering*, 31(4), 297-324.
        """
        if cavity_length_m <= 0:
            raise ValueError(
                f"cavity_length_m must be > 0; got {cavity_length_m}"
            )
        m = (2.0 * self.config.efpi_n_gap * cavity_length_m) / self.config.efpi_wavelength_m
        return float(m)

    def cavity_length_from_fringe_shift(
        self, delta_m: float, reference_L: float
    ) -> float:
        """Infer the new cavity length from an observed fractional fringe shift.

        Derivation
        ----------
        The reference fringe order for *reference_L* is:

            m0 = (2 · n · reference_L) / λ

        After a shift of Δm fringes the new order is m1 = m0 + Δm.
        Inverting the fringe-order equation:

            L_new = (m0 + Δm) · λ / (2 · n)

        Parameters
        ----------
        delta_m : float
            Fractional fringe shift relative to the reference.
            Dimensionless.  Positive → cavity expanded; negative → compressed.
        reference_L : float
            Reference cavity length from which the shift is measured.
            Units: metres.  Valid range: > 0.

        Returns
        -------
        float
            New cavity length *L_new* in metres.

        Raises
        ------
        ValueError
            If *reference_L* ≤ 0 or if the inferred cavity length is
            non-positive (physically impossible).
        """
        if reference_L <= 0:
            raise ValueError(f"reference_L must be > 0; got {reference_L}")
        m0 = self.fringe_order(reference_L)
        m_new = m0 + delta_m
        L_new = m_new * self.config.efpi_wavelength_m / (2.0 * self.config.efpi_n_gap)
        if L_new <= 0:
            raise ValueError(
                f"Inferred cavity length is non-positive ({L_new:.3e} m); "
                f"delta_m={delta_m:.4f} is too large and negative."
            )
        logger.debug(
            "cavity_length_from_fringe_shift: m0=%.4f, Δm=%.6f → L_new=%.3e m",
            m0, delta_m, L_new,
        )
        return float(L_new)

    # ------------------------------------------------------------------
    # Refractive-index model
    # ------------------------------------------------------------------

    def refractive_index_from_density(self, vapor_density_kg_m3: float) -> float:
        """Compute the effective refractive index of the humid gap medium.

        Governing Equation (Edlén formula, simplified for water vapour)
        ----------------------------------------------------------------
        The refractive index of a humid gas is related to its vapour density
        by a linear Clausius-Mossotti approximation valid for small densities:

            n(ρ) = n₀ + (dn/dρ) · ρ

        where:
            n₀     — baseline refractive index (vacuum = 1.0)
            dn/dρ  — config.humidity_refractive_slope = 2.3×10⁻⁴ m³/kg
                     (measured for water vapour at λ = 1550 nm)
            ρ      — vapour density [kg/m³]

        The value 2.3×10⁻⁴ m³/kg is derived from the molar refractivity of
        water vapour evaluated at 1550 nm using the Edlén dispersion formula
        (Edlén 1966).

        Parameters
        ----------
        vapor_density_kg_m3 : float
            Molar concentration of water vapour expressed as mass density.
            Units: kg/m³.  Valid range: ≥ 0.

        Returns
        -------
        float
            Effective refractive index of the gap medium (dimensionless).

        Raises
        ------
        ValueError
            If *vapor_density_kg_m3* < 0.

        References
        ----------
        Edlén, B. (1966). The refractive index of air. *Metrologia*, 2(2),
        71-80.
        """
        if vapor_density_kg_m3 < 0:
            raise ValueError(
                f"vapor_density_kg_m3 must be >= 0; got {vapor_density_kg_m3}"
            )
        n0 = 1.0  # vacuum baseline
        n = n0 + self.config.humidity_refractive_slope * vapor_density_kg_m3
        return float(n)

    # ------------------------------------------------------------------
    # Fringe shift from humidity
    # ------------------------------------------------------------------

    def fringe_shift_from_humidity(self, vapor_density_kg_m3: float) -> float:
        """Compute the fringe shift induced by a given water-vapour density.

        Method
        ------
        1. Compute effective refractive index of the humid gap:
               n_humid = refractive_index_from_density(vapor_density_kg_m3)

        2. Compute the new fringe order using the *nominal* cavity length and
           the updated refractive index:
               m_humid = (2 · n_humid · L₀) / λ

        3. Fringe shift relative to the vacuum baseline (m₀):
               Δm = m_humid − m₀

        The cavity length is held constant; only n changes.  This accurately
        models an EFPI exposed to humidity ingress with a rigid housing.

        Parameters
        ----------
        vapor_density_kg_m3 : float
            Water-vapour mass density in the EFPI cavity.
            Units: kg/m³.  Valid range: ≥ 0.

        Returns
        -------
        float
            Fringe shift Δm (dimensionless).  Positive values indicate
            increased optical path length due to vapour ingress.

        Raises
        ------
        ValueError
            Propagated from :meth:`refractive_index_from_density`.
        """
        n_humid = self.refractive_index_from_density(vapor_density_kg_m3)
        L0 = self.config.efpi_cavity_length_m
        lam = self.config.efpi_wavelength_m
        m_humid = (2.0 * n_humid * L0) / lam
        delta_m = m_humid - self._m0
        logger.debug(
            "fringe_shift_from_humidity: ρ=%.4e kg/m³, n=%.6f, m_humid=%.6f, Δm=%.6f",
            vapor_density_kg_m3, n_humid, m_humid, delta_m,
        )
        return float(delta_m)

    # ------------------------------------------------------------------
    # Ice density inversion
    # ------------------------------------------------------------------

    def infer_ice_density(self, fringe_shift: float) -> float:
        """Invert a measured fringe shift into an estimated ice-mass percentage.

        Inversion Steps
        ---------------
        **Step 1 — Vapour density from fringe shift (linear calibration):**

            ρ_vapour = (Δm − intercept) / slope

        where:
            Δm         — measured fringe shift (dimensionless)
            intercept  — config.fringe_to_density_intercept [fringes]
            slope      — config.fringe_to_density_slope [(fringes/mm)/(kg/m³)]

        **Step 2 — Volumetric ice percentage:**

        The sublimated vapour density ρ_vapour is treated as a mass-fraction
        proxy for the ice content of the sampled regolith.  Using a 1:1
        mass-fraction assumption (1 kg/m³ of vapour corresponds to 1 kg/m³
        of parent ice, neglecting porosity):

            ice_density_pct = (ρ_vapour / ρ_max_vapour) × 100

        where ρ_max_vapour is calibrated to 100 kg/m³ (empirical upper bound
        for a fully ice-saturated regolith sample in the sealed volume).

        The result is clamped to [0, 100] %.

        Parameters
        ----------
        fringe_shift : float
            Observed EFPI fringe shift Δm (dimensionless).

        Returns
        -------
        float
            Estimated ice volume percentage in [0.0, 100.0] %.

        Notes
        -----
        The 1:1 mass-fraction proxy is a first-order approximation.  A full
        sublimation kinetics model (see :mod:`module_5.clausius_clapeyron`)
        should be used for quantitative ice-abundance retrieval.
        """
        # Step 1: invert linear calibration
        vapor_density_kg_m3 = (
            (fringe_shift - self.config.fringe_to_density_intercept)
            / self.config.fringe_to_density_slope
        )
        vapor_density_kg_m3 = max(0.0, vapor_density_kg_m3)

        # Step 2: convert to ice percentage using 1:1 mass-fraction proxy
        # Empirical upper bound: 100 kg/m³ → 100% ice-saturated sample
        _rho_max = 100.0  # kg/m³ — saturated ice regolith upper bound
        ice_density_pct = (vapor_density_kg_m3 / _rho_max) * 100.0

        # Clamp to physically valid range
        ice_density_pct = float(np.clip(ice_density_pct, 0.0, 100.0))

        logger.info(
            "infer_ice_density: Δm=%.6f → ρ_vapour=%.4e kg/m³ → ice=%.3f %%",
            fringe_shift, vapor_density_kg_m3, ice_density_pct,
        )
        return ice_density_pct
