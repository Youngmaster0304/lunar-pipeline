"""
module_5/config.py
==================
Configuration dataclass for the In-Situ Fiber-Optic Hydrodynamics Model
(Extrinsic Fabry-Pérot Interferometer — EFPI), Module 5 of the Lunar South
Pole Autonomous Exploration Pipeline.

All physical constants, optical cavity parameters, and calibration coefficients
are centralised here.  No magic numbers appear in any algorithmic module.

References
----------
- Rao, Y. J. (1999). Recent progress in applications of in-fibre Bragg
  grating sensors. *Optics and Lasers in Engineering*, 31(4), 297-324.
- Hertz-Knudsen equation: Knudsen, M. (1909). *Annalen der Physik*, 334(1),
  179-193.
- Fourier conduction: Incropera, F. P. et al. (2007). *Fundamentals of Heat
  and Mass Transfer*, 6th ed., Wiley.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Module5Config:
    """Centralised configuration for the EFPI ice-sensing module.

    All parameters carry SI units unless explicitly stated otherwise.

    Parameters
    ----------
    efpi_cavity_length_m : float
        Nominal Fabry-Pérot cavity length (air/vacuum gap).
        Units: metres.  Typical range: 10 µm – 200 µm.
        Default: 50 µm (50e-6 m).
    efpi_n_gap : float
        Refractive index of the gap medium (vacuum or dry air).
        Dimensionless.  Valid range: ≥ 1.0.
        Default: 1.0 (vacuum).
    efpi_wavelength_m : float
        Probe laser centre wavelength.
        Units: metres.  Telecom C-band: 1550 nm (1550e-9 m).
    drill_contact_area_m2 : float
        Geometric contact area between the drill bit and regolith surface.
        Units: m².  Default: 1 cm² (1e-4 m²).
    drill_thermal_conductivity : float
        Thermal conductivity of the lunar regolith (conduction path medium).
        Units: W/(m·K).  Typical range: 0.002 – 0.01 W/(m·K) for porous
        regolith; 0.5 W/(m·K) used here as a conservative upper bound for
        compact subsurface regolith.
    drill_contact_duration_s : float
        Duration of active drilling contact (time over which heat is
        transferred from drill bit into regolith).
        Units: seconds.  Default: 30 s.
    regolith_temp_k : float
        Ambient regolith temperature at the Lunar South Pole Permanently
        Shadowed Region (PSR).
        Units: Kelvin.  Typical range: 20 – 40 K.  Default: 25 K.
    drill_temp_k : float
        Operational temperature of the drill bit during active drilling.
        Units: Kelvin.  Default: 350 K (rough diamond-bit operational temp).
    drill_depth_m : float
        Effective conduction path length (drilling depth) used as the
        denominator in Fourier's law.
        Units: metres.  Default: 0.1 m (10 cm).
    sealed_volume_m3 : float
        Volume of the hermetically sealed sampling chamber into which
        sublimated vapour accumulates.
        Units: m³.  Default: 10 cm³ (1e-5 m³).
    fringe_to_density_slope : float
        Calibration coefficient mapping observed fringe shift (fringes/mm
        cavity change) to vapour density.
        Units: (fringes/mm) per (kg/m³ vapour).
        This is a linear empirical calibration constant.
    fringe_to_density_intercept : float
        Calibration offset for the fringe-shift → vapour-density inversion.
        Units: fringes (dimensionless).  Default: 0.0 (zero-offset calibration).
    humidity_refractive_slope : float
        Rate of change of refractive index with water vapour density,
        dn/dρ_vapour, at the probe wavelength of 1550 nm.
        Units: m³/kg (dimensionless index per kg/m³).
        Reference: Edlen (1966) simplified for water vapour at IR wavelengths.
    """

    # -------------------------------------------------------------------
    # EFPI optical cavity parameters
    # -------------------------------------------------------------------
    efpi_cavity_length_m: float = 50e-6       # 50 micrometres nominal
    efpi_n_gap: float = 1.0                   # vacuum/air refractive index
    efpi_wavelength_m: float = 1550e-9        # 1550 nm telecom C-band

    # -------------------------------------------------------------------
    # Drill thermal parameters
    # -------------------------------------------------------------------
    drill_contact_area_m2: float = 1e-4       # 1 cm²
    drill_thermal_conductivity: float = 0.5   # W/(m·K)
    drill_contact_duration_s: float = 30.0    # seconds
    regolith_temp_k: float = 25.0             # K (PSR ambient)
    drill_temp_k: float = 350.0              # K (bit operational)
    drill_depth_m: float = 0.1               # 10 cm conduction path

    # -------------------------------------------------------------------
    # Sealed sampling volume
    # -------------------------------------------------------------------
    sealed_volume_m3: float = 1e-5            # 10 cm³

    # -------------------------------------------------------------------
    # EFPI calibration coefficients
    # -------------------------------------------------------------------
    fringe_to_density_slope: float = 1.2e-3   # (fringes/mm) / (kg/m³)
    fringe_to_density_intercept: float = 0.0  # fringes

    # -------------------------------------------------------------------
    # Refractive index sensitivity
    # -------------------------------------------------------------------
    humidity_refractive_slope: float = 2.3e-4  # m³/kg, dn/dρ at 1550 nm

    def __post_init__(self) -> None:
        """Validate all configuration parameters at construction time.

        Raises
        ------
        ValueError
            If any parameter is out of its physically meaningful range.
        """
        if self.efpi_cavity_length_m <= 0:
            raise ValueError(
                f"efpi_cavity_length_m must be > 0; got {self.efpi_cavity_length_m}"
            )
        if self.efpi_n_gap < 1.0:
            raise ValueError(
                f"efpi_n_gap must be >= 1.0 (vacuum baseline); got {self.efpi_n_gap}"
            )
        if self.efpi_wavelength_m <= 0:
            raise ValueError(
                f"efpi_wavelength_m must be > 0; got {self.efpi_wavelength_m}"
            )
        if self.drill_contact_area_m2 <= 0:
            raise ValueError(
                f"drill_contact_area_m2 must be > 0; got {self.drill_contact_area_m2}"
            )
        if self.drill_thermal_conductivity <= 0:
            raise ValueError(
                f"drill_thermal_conductivity must be > 0; "
                f"got {self.drill_thermal_conductivity}"
            )
        if self.drill_contact_duration_s <= 0:
            raise ValueError(
                f"drill_contact_duration_s must be > 0; "
                f"got {self.drill_contact_duration_s}"
            )
        if self.regolith_temp_k <= 0:
            raise ValueError(
                f"regolith_temp_k must be > 0 K; got {self.regolith_temp_k}"
            )
        if self.drill_temp_k <= 0:
            raise ValueError(
                f"drill_temp_k must be > 0 K; got {self.drill_temp_k}"
            )
        if self.drill_depth_m <= 0:
            raise ValueError(
                f"drill_depth_m must be > 0; got {self.drill_depth_m}"
            )
        if self.sealed_volume_m3 <= 0:
            raise ValueError(
                f"sealed_volume_m3 must be > 0; got {self.sealed_volume_m3}"
            )
        if self.fringe_to_density_slope == 0:
            raise ValueError("fringe_to_density_slope must not be zero (inversion undefined)")
        if self.humidity_refractive_slope < 0:
            raise ValueError(
                f"humidity_refractive_slope must be >= 0; "
                f"got {self.humidity_refractive_slope}"
            )
