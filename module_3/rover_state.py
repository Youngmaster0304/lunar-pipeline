"""
module_3/rover_state.py
=======================
Dataclass representing the runtime state of the lunar rover.

Includes battery energy accounting, sunlight awareness, and energy-step
calculations used by both the planner and the mission orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Module3Config


@dataclass
class RoverState:
    """Runtime state of the lunar micro-rover.

    Parameters
    ----------
    position : tuple[int, int]
        Current grid position as (row, col) indices.
    battery_wh : float
        Current battery state of charge.
        Units: Watt-hours (Wh).
        Valid range: [0, battery_max_wh].
    battery_max_wh : float
        Maximum battery capacity.
        Units: Watt-hours (Wh).
        Valid range: > 0.
    in_sunlight : bool
        Whether the rover's current cell has direct solar illumination.
    timestep : int
        Discrete simulation timestep counter.
        Valid range: >= 0.
    """

    position: tuple[int, int]
    battery_wh: float
    battery_max_wh: float
    in_sunlight: bool
    timestep: int

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def soc_fraction(self) -> float:
        """State-of-charge as a dimensionless fraction in [0, 1].

        Formula
        -------
        soc = battery_wh / battery_max_wh

        Returns
        -------
        float
            SoC fraction in [0.0, 1.0].
        """
        return self.battery_wh / self.battery_max_wh

    def above_reserve(self, config: Module3Config) -> bool:
        """Check whether the rover is above the safety reserve threshold.

        Parameters
        ----------
        config : Module3Config
            Planner configuration supplying battery_reserve_pct.

        Returns
        -------
        bool
            True when soc_fraction > config.battery_reserve_pct.
        """
        return self.soc_fraction > config.battery_reserve_pct

    # ------------------------------------------------------------------
    # Energy step
    # ------------------------------------------------------------------

    def step_energy(
        self,
        distance_m: float,
        slope_deg: float,  # noqa: ARG002  (reserved for future slope-adjusted power)
        config: Module3Config,
    ) -> float:
        """Compute net energy consumed traversing *distance_m* over given slope.

        Energy Model
        ------------
        1. Travel time:

            t = distance_m / config.rover_speed_ms              [seconds]

        2. Motion energy drawn from battery:

            E_motion_J = config.rover_power_w * t               [Joules]
            E_motion_Wh = E_motion_J / 3600                     [Wh]

        3. Solar charge (if in sunlight):

            E_solar_Wh = config.solar_charge_w * t / 3600       [Wh]

        4. Net energy consumed (positive = drained, negative = net charging):

            net_Wh = E_motion_Wh - E_solar_Wh   (if in_sunlight)
            net_Wh = E_motion_Wh                 (if not in_sunlight)

        Note: slope_deg is accepted for signature compatibility and future
        slope-adjusted power draw but is not used in the current linear model.

        Parameters
        ----------
        distance_m : float
            Traversal distance.
            Units: metres. Valid range: >= 0.
        slope_deg : float
            Terrain slope angle along the traversal direction.
            Units: degrees. Valid range: [0, config.max_slope_deg).
        config : Module3Config
            Runtime configuration.

        Returns
        -------
        float
            Net Wh consumed. Positive means battery level drops; negative
            means it rises (solar surplus).
        """
        if distance_m < 0:
            raise ValueError(f"distance_m must be >= 0, got {distance_m}")

        travel_time_s = distance_m / config.rover_speed_ms
        e_motion_wh = config.rover_power_w * travel_time_s / 3600.0

        if self.in_sunlight:
            e_solar_wh = config.solar_charge_w * travel_time_s / 3600.0
            net_wh = e_motion_wh - e_solar_wh
        else:
            net_wh = e_motion_wh

        return net_wh
