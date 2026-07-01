"""Module 5: In-Situ Fiber-Optic Hydrodynamics Model (EFPI).

This module implements the Extrinsic Fabry-Pérot Interferometer (EFPI) sensor
model for detecting and quantifying water ice in lunar regolith samples during
active drilling at the Lunar South Pole.

The pipeline comprises three sub-components:
1. **EFPIModel** — optical sensor model relating cavity optics to ice-vapour
   concentration.
2. **Thermal model** — Fourier conduction from drill bit to regolith, and
   post-drill temperature calculation.
3. **Clausius-Clapeyron model** — sublimation mass flux and sealed-chamber
   vapour accumulation.

Public API
----------
EFPIModel
    Optical interferometric sensor model class.
compute_drill_heat_transfer
    Total conductive heat input from drill bit [J].
compute_post_drill_temperature
    Post-drilling regolith equilibrium temperature [K].
compute_sublimation_rate
    Hertz-Knudsen sublimation mass flux [kg/s].
compute_vapor_density
    Accumulated vapour density inside sealed chamber [kg/m³].
infer_ice_density
    Convenience wrapper: fringe shift → ice mass percentage [%].
"""
from __future__ import annotations

from .clausius_clapeyron import compute_sublimation_rate, compute_vapor_density
from .config import Module5Config
from .efpi_model import EFPIModel
from .thermal_model import compute_drill_heat_transfer, compute_post_drill_temperature


def infer_ice_density(fringe_shift: float, config: Module5Config) -> float:
    """Module-level convenience wrapper: invert a fringe shift to ice percentage.

    Instantiates a fresh :class:`EFPIModel` and delegates to
    :meth:`EFPIModel.infer_ice_density`.

    Parameters
    ----------
    fringe_shift : float
        EFPI fringe shift Δm (dimensionless).
    config : Module5Config
        EFPI module configuration.

    Returns
    -------
    float
        Estimated ice volume percentage in [0.0, 100.0] %.
    """
    model = EFPIModel(config)
    return model.infer_ice_density(fringe_shift)


__all__ = [
    "EFPIModel",
    "Module5Config",
    "compute_drill_heat_transfer",
    "compute_post_drill_temperature",
    "compute_sublimation_rate",
    "compute_vapor_density",
    "infer_ice_density",
]
