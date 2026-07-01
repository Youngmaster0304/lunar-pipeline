"""Module 4: Reactive Obstacle Avoidance & Simplified SLAM.

This module implements the Bug-2 reactive path planner for real-time obstacle
circumnavigation during rover traversal, combined with a simplified EKF-SLAM
(Extended Kalman Filter Simultaneous Localisation and Mapping) localisation
system.

The autonomy stack operates as follows:
1. **AutonomyMode** manages the overall control mode (nominal path following
   vs. obstacle circumnavigation).
2. **Bug2Planner** executes Bug-2: the rover follows the nominal planned path
   until an obstacle is detected, then hugs the obstacle boundary until the
   M-line to the goal is re-intersected.
3. **EKFSLAMState** maintains a simplified covariance estimate for rover
   position uncertainty, updated at each sensor tick.

References
----------
- Lumelsky & Stepanov (1987). Path-planning strategies for a point mobile
  automaton. *Algorithmica*, 2(1-4), 403-430.
- Thrun, Burgard & Fox (2005). *Probabilistic Robotics*. MIT Press.

Public API
----------
AutonomyMode
    Manages rover autonomy state machine.
Bug2Planner
    Bug-2 reactive obstacle avoidance planner.
EKFSLAMState
    Simplified EKF-SLAM position covariance tracker.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import Module4Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AutonomyMode
# ---------------------------------------------------------------------------


class AutonomyMode:
    """State machine managing rover autonomy level.

    States
    ------
    'nominal'
        Rover follows the pre-planned path from Module 3.
    'obstacle'
        Rover is in Bug-2 boundary-following mode (obstacle detected).
    'recovery'
        Rover has lost the planned path and is attempting re-localisation.

    Parameters
    ----------
    config : Module4Config
        Autonomy module configuration.
    """

    _VALID_STATES = frozenset({"nominal", "obstacle", "recovery"})

    def __init__(self, config: Module4Config) -> None:
        self.config: Module4Config = config
        self._state: str = "nominal"
        logger.debug("AutonomyMode initialised in 'nominal' state.")

    @property
    def state(self) -> str:
        """Current autonomy state (str)."""
        return self._state

    def transition(self, new_state: str) -> None:
        """Transition to a new autonomy state.

        Parameters
        ----------
        new_state : str
            Target state.  Must be one of 'nominal', 'obstacle', 'recovery'.

        Raises
        ------
        ValueError
            If *new_state* is not a valid state.
        """
        if new_state not in self._VALID_STATES:
            raise ValueError(
                f"Invalid autonomy state '{new_state}'. "
                f"Must be one of {self._VALID_STATES}."
            )
        if new_state != self._state:
            logger.info(
                "AutonomyMode transition: '%s' → '%s'", self._state, new_state
            )
            self._state = new_state

    def compute_mae(
        self, estimated_path: List[Tuple[int, int]], ground_truth_path: List[Tuple[int, int]]
    ) -> float:
        """Compute Mean Absolute Error (MAE) between estimated and ground-truth positions.

        MAE = (1 / N) * Σᵢ |x1ᵢ − x2ᵢ|   [Equation (6)]

        where x1ᵢ is the SLAM-estimated position and x2ᵢ is the ground truth.
        LiDAR and map coordinate axes must be aligned for accurate MAE.

        Parameters
        ----------
        estimated_path : list of (row, col)
            Positions estimated by the SLAM / autonomy stack.
        ground_truth_path : list of (row, col)
            True positions (from the nominal planned path).

        Returns
        -------
        float
            Mean Absolute Error in grid cells.
        """
        if not estimated_path or not ground_truth_path:
            return 0.0
        n = min(len(estimated_path), len(ground_truth_path))
        total_error = 0.0
        for i in range(n):
            dr = estimated_path[i][0] - ground_truth_path[i][0]
            dc = estimated_path[i][1] - ground_truth_path[i][1]
            total_error += math.sqrt(dr ** 2 + dc ** 2)
        mae = total_error / n
        logger.debug("SLAM MAE: %.4f grid cells (n=%d)", mae, n)
        return mae

    def detect_obstacle(self, obstacle_map: np.ndarray, position: Tuple[int, int]) -> bool:
        """Check whether the rover's immediate neighbourhood contains obstacles.

        Scans a 3×3 neighbourhood around *position* in *obstacle_map*.

        Parameters
        ----------
        obstacle_map : np.ndarray, shape (R, C), dtype bool or float
            Map where True / non-zero indicates an obstacle cell.
        position : tuple[int, int]
            Current rover position as (row, col).

        Returns
        -------
        bool
            True if any obstacle is detected in the neighbourhood.
        """
        r, c = position
        rows, cols = obstacle_map.shape
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    if obstacle_map[rr, cc]:
                        return True
        return False

    def is_in_shadow_mode(self, dte_ok: bool) -> bool:
        """Check if rover is in strategic autonomy mode due to DTE loss."""
        return not dte_ok



# ---------------------------------------------------------------------------
# EKF SLAM State (simplified)
# ---------------------------------------------------------------------------


@dataclass
class EKFSLAMState:
    """Simplified EKF-SLAM position uncertainty tracker.

    Maintains a 2×2 position covariance matrix updated via a linear
    predict-update cycle (simplified: no landmark map, just pose covariance).

    Governing Equations
    -------------------
    **Predict step** (constant-velocity motion model):

        P_pred = F · P · Fᵀ + Q

    where F = I (identity for zero-velocity assumption between ticks) and
    Q = σ_proc² · I (isotropic process noise).

    **Update step** (range measurement):

        K = P_pred · Hᵀ · (H · P_pred · Hᵀ + R)⁻¹
        P_updated = (I - K · H) · P_pred

    where H = I (direct position measurement) and R = σ_meas² · I.

    Parameters
    ----------
    config : Module4Config
        SLAM configuration (process and measurement noise).
    """

    config: Module4Config
    covariance: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Initialise covariance to zero (perfect knowledge at start)."""
        self.covariance = np.zeros((2, 2), dtype=float)

    def predict(self) -> None:
        """EKF predict step: add process noise to position covariance.

        Equation:
            P ← P + Q   where Q = diag(σ_proc², σ_proc²)
        """
        Q = np.eye(2) * self.config.slam_process_noise
        self.covariance = self.covariance + Q

    def update(self) -> None:
        """EKF update step: fuse a range measurement to reduce covariance.

        Simplified: H = I, R = diag(σ_meas², σ_meas²).

        Kalman gain:  K = P · (P + R)⁻¹
        Update:       P ← (I - K) · P
        """
        R = np.eye(2) * self.config.slam_measurement_noise
        K = self.covariance @ np.linalg.inv(self.covariance + R)
        self.covariance = (np.eye(2) - K) @ self.covariance

    @property
    def position_std_m(self) -> float:
        """RMS position uncertainty (scalar, metres).

        Returns
        -------
        float
            sqrt(trace(P) / 2) — average 1-σ positional uncertainty [m].
        """
        return float(math.sqrt(np.trace(self.covariance) / 2.0))


# ---------------------------------------------------------------------------
# Bug2Planner
# ---------------------------------------------------------------------------


class Bug2Planner:
    """Bug-2 reactive obstacle avoidance planner.

    Bug-2 Algorithm (Lumelsky & Stepanov 1987)
    ------------------------------------------
    1. Draw the M-line: straight line from start to goal.
    2. Move toward the goal along the M-line.
    3. If an obstacle is hit at point H:
       a. Record H.
       b. Follow the obstacle boundary (left-hand or right-hand rule).
       c. When the M-line is re-intersected at a point closer to the goal
          than H, leave the boundary and resume step 2.
    4. If the robot returns to H without finding a closer M-line crossing,
       the goal is unreachable.

    This implementation works on a discrete grid.  Obstacle boundary following
    uses 8-connected clockwise rotation of the approach direction.

    Parameters
    ----------
    nominal_path : list[tuple[int, int]]
        Pre-planned path from Module 3 A* (used as the M-line waypoints).
    obstacle_map : np.ndarray, shape (R, C), dtype bool
        Static obstacle map (True = obstacle).  Dynamic updates are not
        modelled in this simplified version.
    config : Module4Config
        Autonomy module configuration.
    """

    # 8-connected direction vectors (clockwise from East)
    _DIRECTIONS: List[Tuple[int, int]] = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
    ]

    def __init__(
        self,
        nominal_path: List[Tuple[int, int]],
        obstacle_map: np.ndarray,
        config: Module4Config,
    ) -> None:
        self.nominal_path: List[Tuple[int, int]] = nominal_path
        self.obstacle_map: np.ndarray = obstacle_map
        self.config: Module4Config = config
        self._slam = EKFSLAMState(config=config)
        logger.debug(
            "Bug2Planner initialised with nominal_path length=%d, "
            "obstacle_map shape=%s",
            len(nominal_path), obstacle_map.shape,
        )

    def _is_obstacle(self, pos: Tuple[int, int]) -> bool:
        """Return True if *pos* is an obstacle or out of bounds."""
        r, c = pos
        rows, cols = self.obstacle_map.shape
        if not (0 <= r < rows and 0 <= c < cols):
            return True
        return bool(self.obstacle_map[r, c])

    def _on_mline(
        self,
        pos: Tuple[int, int],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> bool:
        """Check whether *pos* lies on the integer M-line between start and goal.

        Uses cross-product collinearity test on integer grid coordinates.
        A cell is considered 'on' the M-line if the cross product
        |(goal-start) × (pos-start)| < 1.5 (sub-cell tolerance).

        Parameters
        ----------
        pos, start, goal : tuple[int, int]
            Grid coordinates.

        Returns
        -------
        bool
        """
        ds = (goal[0] - start[0], goal[1] - start[1])
        dp = (pos[0] - start[0], pos[1] - start[1])
        cross = abs(ds[0] * dp[1] - ds[1] * dp[0])
        return cross < 1.5

    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance between two grid cells."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def run(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Execute the Bug-2 traversal from *start* to *goal*.

        The planner follows the nominal path waypoints and switches to
        boundary-following when an obstacle is encountered, returning to
        the nominal path when the M-line is re-intersected.

        Parameters
        ----------
        start : tuple[int, int]
            Starting grid position (row, col).
        goal : tuple[int, int]
            Goal grid position (row, col).

        Returns
        -------
        list[tuple[int, int]]
            Executed trajectory as a sequence of (row, col) grid positions.
            May not reach goal if max_bug2_iterations is exceeded.
        """
        trajectory: List[Tuple[int, int]] = [start]
        current = start
        hit_point: Optional[Tuple[int, int]] = None
        boundary_mode = False
        dir_idx = 0  # current facing direction index into _DIRECTIONS
        iterations = 0
        max_iter = self.config.max_bug2_iterations

        # Build goal set (just the goal cell for this simplified version)
        waypoints = list(self.nominal_path) if self.nominal_path else [goal]
        waypoint_idx = 0

        logger.info("Bug2Planner.run: start=%s, goal=%s", start, goal)

        while current != goal and iterations < max_iter:
            iterations += 1
            # EKF predict + update each tick
            self._slam.predict()
            self._slam.update()

            if not boundary_mode:
                # --- Move toward next waypoint ---
                target = waypoints[min(waypoint_idx, len(waypoints) - 1)]
                if target == current:
                    waypoint_idx = min(waypoint_idx + 1, len(waypoints) - 1)
                    target = waypoints[waypoint_idx]

                # Determine step direction toward target
                dr = target[0] - current[0]
                dc = target[1] - current[1]
                # Normalise to unit grid step
                step_r = (1 if dr > 0 else -1 if dr < 0 else 0)
                step_c = (1 if dc > 0 else -1 if dc < 0 else 0)
                next_pos = (current[0] + step_r, current[1] + step_c)

                if self._is_obstacle(next_pos):
                    # Enter boundary-following mode
                    hit_point = current
                    boundary_mode = True
                    logger.debug(
                        "Obstacle hit at %s; entering boundary mode.", current
                    )
                    # Find initial boundary direction (rotate until free)
                    for i in range(8):
                        dr_dir, dc_dir = self._DIRECTIONS[dir_idx]
                        candidate = (current[0] + dr_dir, current[1] + dc_dir)
                        if not self._is_obstacle(candidate):
                            current = candidate
                            trajectory.append(current)
                            break
                        dir_idx = (dir_idx + 1) % 8
                    continue

                if not self._is_obstacle(next_pos):
                    current = next_pos
                    trajectory.append(current)
                else:
                    # Steer around — shouldn't reach here in this flow
                    dir_idx = (dir_idx + 1) % 8

            else:
                # --- Boundary-following (left-hand rule) ---
                # Try to turn left (counter-clockwise) and step
                left_idx = (dir_idx - 1) % 8
                dr_l, dc_l = self._DIRECTIONS[left_idx]
                left_pos = (current[0] + dr_l, current[1] + dc_l)

                if not self._is_obstacle(left_pos):
                    dir_idx = left_idx
                    current = left_pos
                else:
                    # Turn right until free
                    for _ in range(8):
                        dir_idx = (dir_idx + 1) % 8
                        dr_d, dc_d = self._DIRECTIONS[dir_idx]
                        next_try = (current[0] + dr_d, current[1] + dc_d)
                        if not self._is_obstacle(next_try):
                            current = next_try
                            break

                trajectory.append(current)

                # Check if we're back on the M-line and closer to goal than H
                if (
                    hit_point is not None
                    and self._on_mline(current, start, goal)
                    and self._distance(current, goal) < self._distance(hit_point, goal)
                    and current != hit_point
                ):
                    boundary_mode = False
                    hit_point = None
                    logger.debug(
                        "M-line re-intersected at %s; resuming nominal path.", current
                    )

                # Check if we've looped back to hit_point (unreachable)
                if hit_point is not None and current == hit_point and iterations > 1:
                    logger.warning(
                        "Bug2: returned to hit_point %s — goal unreachable.", hit_point
                    )
                    break

        if current == goal:
            logger.info(
                "Bug2Planner reached goal %s in %d iterations. "
                "Final SLAM uncertainty: %.4f m",
                goal, iterations, self._slam.position_std_m,
            )
        else:
            logger.warning(
                "Bug2Planner stopped at %s after %d iterations (max=%d). "
                "Goal %s not reached.",
                current, iterations, max_iter, goal,
            )

        return trajectory

    def compute_mae(
        self, estimated_path: List[Tuple[int, int]], ground_truth_path: List[Tuple[int, int]]
    ) -> float:
        """Mean Absolute Error between estimated and ground-truth path.

        MAE = (1 / N) * Σᵢ √(Δrow² + Δcol²)

        Parameters
        ----------
        estimated_path : list of (row, col)
            Positions estimated by the SLAM / autonomy stack.
        ground_truth_path : list of (row, col)
            True positions from the nominal planned path.

        Returns
        -------
        float
            Mean Absolute Error in grid cells.
        """
        if not estimated_path or not ground_truth_path:
            return 0.0
        n = min(len(estimated_path), len(ground_truth_path))
        total = 0.0
        for i in range(n):
            dr = estimated_path[i][0] - ground_truth_path[i][0]
            dc = estimated_path[i][1] - ground_truth_path[i][1]
            total += math.sqrt(dr ** 2 + dc ** 2)
        mae = total / n
        logger.debug("SLAM MAE: %.4f grid cells (n=%d)", mae, n)
        return mae


__all__ = [
    "Module4Config",
    "AutonomyMode",
    "Bug2Planner",
    "EKFSLAMState",
]
