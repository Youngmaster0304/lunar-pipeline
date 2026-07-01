"""Configuration dataclass for Module 1: Radar Polarimetric Decomposition (DFSAR).

All threshold and physical constants used throughout Module 1 are centralised
here so that no magic numbers appear in the algorithmic code.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Module1Config:
    """Top-level configuration for the DFSAR polarimetric decomposition pipeline.

    Attributes
    ----------
    cpr_threshold : float
        Circular Polarisation Ratio threshold above which a pixel is flagged as
        a potential ice-bearing unit.  Dimensionless.  Typical ice anomaly
        CPR > 1.0 (Stacy et al. 1997).
    dop_threshold : float
        Degree-of-Polarisation threshold *below* which a pixel is considered
        partly-depolarised (consistent with volume scattering from ice).
        Dimensionless, range [0, 1].  Default 0.13 from Rao et al. 2022.
    cpr_band : str
        Which radar band(s) to use for CPR computation.
        Allowed values: ``'L'``, ``'S'``, or ``'both'``.
    crim_eps_ice : complex
        Relative complex permittivity of pure water ice at microwave
        frequencies (3 GHz range).
        Reference: Cumming (1952), real part ≈ 3.15, loss tangent ≈ 3×10⁻⁴.
        Units: dimensionless (SI relative permittivity).
    crim_eps_regolith : complex
        Relative complex permittivity of dry bulk lunar regolith.
        Reference: Olhoeft & Strangway (1975), real part ≈ 2.7.
        Units: dimensionless.
    crim_eps_vacuum : complex
        Relative complex permittivity of free space / vacuum / pore space.
        Always 1.0 + 0j.  Units: dimensionless.
    crim_tolerance : float
        Convergence tolerance (absolute) for the brentq root-finder used in
        the CRIM inversion.  Dimensionless.
    crim_porosity : float
        Fixed volumetric porosity of the lunar regolith (fraction of vacuum
        pore space).  f_vac is fixed at this value so that only two degrees
        of freedom remain: f_ice and f_reg.
        Physically motivated value: ~40 % (Carrier et al. 1991).
        Units: dimensionless, range (0, 1).
    crim_sigma_to_eps_slope : float
        Empirical linear slope mapping linear backscatter coefficient σ₀
        (dimensionless linear) to effective bulk relative permittivity.
        Document: This is a simplified calibration proxy; a full EM forward
        model should be used for quantitative retrievals.
        Units: [ε per (m²/m²)].
    crim_sigma_to_eps_intercept : float
        Empirical linear intercept in the σ₀ → ε mapping.
        Units: [ε].
    """

    cpr_threshold: float = 1.0
    dop_threshold: float = 0.13
    cpr_band: str = "both"  # 'L', 'S', or 'both'

    # CRIM end-member permittivities (complex)
    crim_eps_ice: complex = field(default_factory=lambda: complex(3.15, 0.001))
    crim_eps_regolith: complex = field(default_factory=lambda: complex(2.7, 0.002))
    crim_eps_vacuum: complex = field(default_factory=lambda: complex(1.0, 0.0))

    # CRIM numerical settings
    crim_tolerance: float = 0.01
    crim_porosity: float = 0.40  # fixed vacuum fraction (Carrier et al. 1991)

    # Empirical σ₀ → ε calibration (simplified linear proxy)
    crim_sigma_to_eps_slope: float = 3.0
    crim_sigma_to_eps_intercept: float = 1.5

    def __post_init__(self) -> None:
        """Validate configuration values after initialisation."""
        allowed_bands = {"L", "S", "both"}
        if self.cpr_band not in allowed_bands:
            raise ValueError(
                f"cpr_band must be one of {allowed_bands}; got '{self.cpr_band}'"
            )
        if not 0.0 < self.crim_porosity < 1.0:
            raise ValueError(
                f"crim_porosity must be in (0, 1); got {self.crim_porosity}"
            )
        if self.cpr_threshold <= 0:
            raise ValueError(
                f"cpr_threshold must be positive; got {self.cpr_threshold}"
            )
        if not 0.0 <= self.dop_threshold <= 1.0:
            raise ValueError(
                f"dop_threshold must be in [0, 1]; got {self.dop_threshold}"
            )
