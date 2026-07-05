"""
module_3/config.py
==================
Configuration dataclass for the Hybrid Risk-Aware Path Planner (Module 3).

All physical constants, energy parameters, and algorithmic weights are stored
here so that no magic numbers appear in planner logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Module3Config:
    """Centralised configuration for the Gorilla Traversal path planner.

    Parameters
    ----------
    alpha1 : float
        Weight for motion-energy cost in Total Cost formula.
        Total = α₁ * EnergyCost + α₂ * RiskCost.
        Units: dimensionless.  Valid range: > 0.
    alpha2 : float
        Weight for risk cost (DEM slope, pitch angle) in Total Cost.
        Total = α₁ * EnergyCost + α₂ * RiskCost.
        Units: dimensionless.  Valid range: > 0.
    beta0 : float
        Exponential pre-factor for the slope penalty term (risk cost).
        Units: dimensionless.
        Valid range: > 0.
    gamma : float
        Revisit-penalty weight applied to P_visited heatmap.
        Units: dimensionless.
        Valid range: >= 0.
    max_slope_deg : float
        Hard slope limit beyond which edges are pruned from the graph.
        Units: degrees.
        Valid range: (0, 90).
    slope_clamp_eps : float
        Safety margin subtracted from max_slope_deg during clamping to
        prevent division-by-zero in the slope-penalty formula.
        Units: degrees.
        Valid range: (0, max_slope_deg).
    revisit_decay : float
        Multiplicative exponential decay applied to P_visited each timestep.
        Units: dimensionless (probability-like fraction).
        Valid range: (0, 1].
    battery_reserve_pct : float
        Minimum SoC fraction the rover must maintain (safety reserve).
        Units: dimensionless fraction.
        Valid range: [0, 1).
    rover_power_w : float
        Nominal electrical power draw during forward motion.
        Units: Watts.
        Valid range: > 0.
    rover_power_rotate_w : float
        Power draw during spot-turn (rotation in place).
        Units: Watts.  Valid range: > 0.
    rover_speed_ms : float
        Nominal rover traversal speed.
        Units: m/s.
        Valid range: > 0.
    solar_charge_w : float
        Solar panel input power when the rover is in direct sunlight.
        Units: Watts.
        Valid range: >= 0.
    grid_resolution_m : float
        Physical size of one grid cell edge.
        Units: metres.
        Valid range: > 0.
    myopic_cost_weight : float
        Weight for real-time (myopic) sensor cost in hybrid planning.
        The global DEM-based cost is computed in advance; the myopic
        cost is computed at runtime from the rover's local sensor data.
        Total = (1 - myopic_weight) * global_cost + myopic_weight * myopic_cost.
        Valid range: [0, 1].
    """

    alpha1: float = 1.0
    alpha2: float = 1.5
    beta0: float = 2.0
    gamma: float = 0.5
    max_slope_deg: float = 25.0
    slope_clamp_eps: float = 0.5
    revisit_decay: float = 0.9
    battery_reserve_pct: float = 0.15
    rover_power_w: float = 50.0
    rover_power_rotate_w: float = 30.0
    rover_speed_ms: float = 0.1
    solar_charge_w: float = 30.0
    grid_resolution_m: float = 1.0
    myopic_cost_weight: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration parameters at construction time."""
        if self.alpha1 <= 0:
            raise ValueError(f"alpha1 must be > 0, got {self.alpha1}")
        if self.alpha2 <= 0:
            raise ValueError(f"alpha2 must be > 0, got {self.alpha2}")
        if self.beta0 <= 0:
            raise ValueError(f"beta0 must be > 0, got {self.beta0}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")
        if not (0 < self.max_slope_deg < 90):
            raise ValueError(
                f"max_slope_deg must be in (0, 90), got {self.max_slope_deg}"
            )
        if not (0 < self.slope_clamp_eps < self.max_slope_deg):
            raise ValueError(
                f"slope_clamp_eps must be in (0, max_slope_deg={self.max_slope_deg}), "
                f"got {self.slope_clamp_eps}"
            )
        if not (0 < self.revisit_decay <= 1.0):
            raise ValueError(
                f"revisit_decay must be in (0, 1], got {self.revisit_decay}"
            )
        if not (0.0 <= self.battery_reserve_pct < 1.0):
            raise ValueError(
                f"battery_reserve_pct must be in [0, 1), got {self.battery_reserve_pct}"
            )
        if self.rover_power_w <= 0:
            raise ValueError(f"rover_power_w must be > 0, got {self.rover_power_w}")
        if self.rover_speed_ms <= 0:
            raise ValueError(f"rover_speed_ms must be > 0, got {self.rover_speed_ms}")
        if self.solar_charge_w < 0:
            raise ValueError(f"solar_charge_w must be >= 0, got {self.solar_charge_w}")
        if self.grid_resolution_m <= 0:
            raise ValueError(
                f"grid_resolution_m must be > 0, got {self.grid_resolution_m}"
            )
        if self.rover_power_rotate_w <= 0:
            raise ValueError(
                f"rover_power_rotate_w must be > 0, got {self.rover_power_rotate_w}"
            )
        if not 0.0 <= self.myopic_cost_weight <= 1.0:
            raise ValueError(
                f"myopic_cost_weight must be in [0, 1], got {self.myopic_cost_weight}"
            )
