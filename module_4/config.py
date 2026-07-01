"""
module_4/config.py
==================
Configuration dataclass for Module 4: Reactive Obstacle Avoidance & SLAM.

This module implements Bug-2 reactive path planning layered over SLAM-based
localisation for real-time obstacle avoidance during rover traversal.

References
----------
- Lumelsky, V., & Stepanov, A. (1987). Path-planning strategies for a point
  mobile automaton moving amidst unknown obstacles of arbitrary shape.
  *Algorithmica*, 2(1-4), 403-430.  (Bug-2 algorithm.)
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*.
  MIT Press.  (EKF-SLAM.)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Module4Config:
    """Configuration for the reactive obstacle avoidance and SLAM module.

    Parameters
    ----------
    obstacle_threshold : float
        Sensor distance below which a cell is classified as an obstacle.
        Units: metres.  Valid range: > 0.
        Default: 0.5 m.
    bug2_step_size_m : float
        Step size used by the Bug-2 planner when tracing around obstacles.
        Units: metres.  Valid range: > 0.
        Default: 1.0 m (one grid cell at 1 m resolution).
    slam_process_noise : float
        Process noise covariance diagonal entry for the EKF-SLAM filter.
        Represents uncertainty in rover motion model per timestep.
        Units: m² (position variance per step).  Valid range: > 0.
        Default: 0.01 m².
    slam_measurement_noise : float
        Measurement noise covariance diagonal entry for the EKF-SLAM filter.
        Represents LiDAR/range-sensor ranging uncertainty.
        Units: m².  Valid range: > 0.
        Default: 0.05 m².
    max_bug2_iterations : int
        Maximum number of Bug-2 boundary-following steps before declaring
        the target unreachable.  Valid range: > 0.
        Default: 10 000.
    lidar_range_m : float
        Maximum effective range of the LiDAR sensor used for obstacle detection
        and SLAM landmark extraction.
        Units: metres.  Valid range: > 0.
        Default: 5.0 m.
    autonomy_tick_hz : float
        Rate at which the autonomy loop executes (control frequency).
        Units: Hz.  Valid range: > 0.
        Default: 10.0 Hz.
    """

    obstacle_threshold: float = 0.5       # metres
    bug2_step_size_m: float = 1.0         # metres
    slam_process_noise: float = 0.01      # m² per step
    slam_measurement_noise: float = 0.05  # m²
    max_bug2_iterations: int = 10_000
    lidar_range_m: float = 5.0            # metres
    autonomy_tick_hz: float = 10.0        # Hz

    def __post_init__(self) -> None:
        """Validate configuration at construction time.

        Raises
        ------
        ValueError
            If any parameter is outside its physically meaningful range.
        """
        if self.obstacle_threshold <= 0:
            raise ValueError(
                f"obstacle_threshold must be > 0; got {self.obstacle_threshold}"
            )
        if self.bug2_step_size_m <= 0:
            raise ValueError(
                f"bug2_step_size_m must be > 0; got {self.bug2_step_size_m}"
            )
        if self.slam_process_noise <= 0:
            raise ValueError(
                f"slam_process_noise must be > 0; got {self.slam_process_noise}"
            )
        if self.slam_measurement_noise <= 0:
            raise ValueError(
                f"slam_measurement_noise must be > 0; "
                f"got {self.slam_measurement_noise}"
            )
        if self.max_bug2_iterations <= 0:
            raise ValueError(
                f"max_bug2_iterations must be > 0; got {self.max_bug2_iterations}"
            )
        if self.lidar_range_m <= 0:
            raise ValueError(
                f"lidar_range_m must be > 0; got {self.lidar_range_m}"
            )
        if self.autonomy_tick_hz <= 0:
            raise ValueError(
                f"autonomy_tick_hz must be > 0; got {self.autonomy_tick_hz}"
            )
