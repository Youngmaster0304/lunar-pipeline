"""Tests for Module 4: Reactive Obstacle Avoidance."""
from __future__ import annotations

import math
import numpy as np
import pytest

from module_4.config import Module4Config
from module_4.__init__ import Bug2Planner, AutonomyMode, EKFSLAMState


@pytest.fixture
def config():
    return Module4Config(
        obstacle_threshold=0.5,
        max_bug2_iterations=100
    )


def test_bug2_no_obstacles(config):
    """Test Bug-2 on an empty grid: should go straight to goal."""
    obstacle_map = np.zeros((10, 10), dtype=bool)
    start = (1, 1)
    goal = (5, 5)
    
    # Path is straight line
    nominal_path = [(i, i) for i in range(1, 6)]
    
    planner = Bug2Planner(nominal_path, obstacle_map, config)
    trajectory = planner.run(start, goal)
    
    assert trajectory[-1] == goal
    # Trajectory length should be exact number of diagonal steps + 1 for start
    assert len(trajectory) == 5


def test_bug2_blocked_completely(config):
    """Test Bug-2 when goal is completely blocked."""
    obstacle_map = np.zeros((10, 10), dtype=bool)
    start = (1, 1)
    goal = (5, 5)
    
    # Create an impenetrable wall around the goal
    obstacle_map[4:7, 4:7] = True
    obstacle_map[5, 5] = False # goal itself is free, but surrounded
    
    planner = Bug2Planner([], obstacle_map, config)
    trajectory = planner.run(start, goal)
    
    assert trajectory[-1] != goal


def test_autonomy_nominal_speed(config):
    """Test nominal speed selection."""
    mode = AutonomyMode(config)
    assert not mode.is_in_shadow_mode(dte_ok=True)


def test_autonomy_shadow_speed(config):
    """Test strategic autonomy speed selection."""
    mode = AutonomyMode(config)
    assert mode.is_in_shadow_mode(dte_ok=False)


def test_compute_mae_perfect_match(config):
    """MAE = 0 when estimated path matches ground truth."""
    planner = Bug2Planner([], np.zeros((5, 5), dtype=bool), config)
    path = [(0, 0), (1, 1), (2, 2)]
    mae = planner.compute_mae(path, path)
    assert mae == 0.0


def test_compute_mae_known_offset(config):
    """MAE for a constant 1-cell offset."""
    planner = Bug2Planner([], np.zeros((5, 5), dtype=bool), config)
    estimated = [(0, 0), (1, 1), (2, 2)]
    ground_truth = [(0, 1), (1, 2), (2, 3)]  # col offset of 1
    mae = planner.compute_mae(estimated, ground_truth)
    expected = 1.0  # each step has |dc| = 1
    assert abs(mae - expected) < 1e-10


def test_compute_mae_empty_input(config):
    """Empty input returns 0.0."""
    planner = Bug2Planner([], np.zeros((5, 5), dtype=bool), config)
    assert planner.compute_mae([], []) == 0.0


def test_slam_predict_update(config):
    """Test basic EKF slam covariance updates."""
    slam = EKFSLAMState(config)
    initial_std = slam.position_std_m
    assert initial_std == 0.0
    
    slam.predict()
    assert slam.position_std_m > 0.0
    
    std_after_predict = slam.position_std_m
    slam.update()
    
    # Uncertainty should decrease after a measurement update
    assert slam.position_std_m < std_after_predict
