"""
module_3/tests/test_module3.py
==============================
Pytest test suite for Module 3 – Hybrid Risk-Aware Path Planner.

Tests cover:
  1. Edge cost formula correctness.
  2. Simple path on a flat 5×5 grid.
  3. Battery-limited infeasibility.
  4. Completely blocked graph (no_path).
  5. Revisit decay: P_visited influences second-run cost.
  6. Slope clamping at max_slope_deg boundary.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from module_3.config import Module3Config
from module_3.graph import _edge_cost, build_terrain_graph
from module_3.grid_state import GridState, CellState
from module_3.planner import PlannerResult, plan_path
from module_3.rover_state import RoverState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_state(
    battery_wh: float = 100.0,
    in_sunlight: bool = False,
) -> RoverState:
    """Create a RoverState at (0,0) with given battery."""
    return RoverState(
        position=(0, 0),
        battery_wh=battery_wh,
        battery_max_wh=100.0,
        in_sunlight=in_sunlight,
        timestep=0,
    )


def _flat_grid(size: int, config: Module3Config) -> tuple:
    """Build zero-slope grid arrays and a terrain graph."""
    slope_map = np.zeros((size, size), dtype=np.float64)
    ice_mask = np.zeros((size, size), dtype=bool)
    sunlit_mask = np.zeros((size, size), dtype=bool)
    graph = build_terrain_graph(slope_map, ice_mask, config)
    return slope_map, ice_mask, sunlit_mask, graph


# ---------------------------------------------------------------------------
# Test 1: edge cost formula
# ---------------------------------------------------------------------------


class TestEdgeCostFormula:
    """Verify the composite edge cost formula with hand-computed values."""

    def test_zero_slope(self) -> None:
        """At zero slope: Total = α₁·d + α₂·β₀·exp(0).

        Formula (is_turn=False, slope=0):
            E_forward = α₁ · d_m                    = 1.0 · 1.0 = 1.0
            slope_penalty = β₀ · exp(0/(15-0))      = 2.0 · 1.0 = 2.0
            total = E_forward + α₂ · slope_penalty  = 1.0 + 1.5 · 2.0 = 4.0
        """
        cfg = Module3Config(alpha1=1.0, alpha2=1.5, beta0=2.0,
                            max_slope_deg=15.0, slope_clamp_eps=0.5)
        cost = _edge_cost(1.0, 0.0, cfg)
        s_pen = cfg.beta0 * math.exp(0.0)
        expected = cfg.alpha1 * 1.0 * cfg.grid_resolution_m + cfg.alpha2 * s_pen
        assert abs(cost - expected) < 1e-10, f"Expected {expected}, got {cost}"

    def test_known_slope(self) -> None:
        """At slope = 7.5°, Total = α₁·d + α₂·β₀·exp(s/(max-s))."""
        cfg = Module3Config(alpha1=1.0, alpha2=1.5, beta0=2.0,
                            max_slope_deg=15.0, slope_clamp_eps=0.5)
        slope = 7.5
        s_clamp = min(slope, cfg.max_slope_deg - cfg.slope_clamp_eps)
        s_pen = cfg.beta0 * math.exp(s_clamp / (cfg.max_slope_deg - s_clamp))
        expected = cfg.alpha1 * cfg.grid_resolution_m + cfg.alpha2 * s_pen
        cost = _edge_cost(1.0, slope, cfg)
        assert abs(cost - expected) < 1e-10

    def test_diagonal_distance(self) -> None:
        """Diagonal edge with sqrt(2) distance: α₂ scales slope penalty."""
        cfg = Module3Config(alpha1=1.0, alpha2=1.5, beta0=2.0,
                            max_slope_deg=15.0, slope_clamp_eps=0.5)
        cost = _edge_cost(math.sqrt(2), 0.0, cfg)
        s_pen = cfg.beta0 * math.exp(0.0)
        expected = cfg.alpha1 * math.sqrt(2) * cfg.grid_resolution_m + cfg.alpha2 * s_pen
        assert abs(cost - expected) < 1e-10

    def test_alpha2_scales_risk(self) -> None:
        """Increasing α₂ must increase total cost when slope > 0."""
        cfg_base = Module3Config(alpha1=1.0, alpha2=1.0, beta0=2.0,
                                 max_slope_deg=15.0, slope_clamp_eps=0.5)
        cfg_high = Module3Config(alpha1=1.0, alpha2=5.0, beta0=2.0,
                                 max_slope_deg=15.0, slope_clamp_eps=0.5)
        cost_base = _edge_cost(1.0, 10.0, cfg_base)
        cost_high = _edge_cost(1.0, 10.0, cfg_high)
        assert cost_high > cost_base, "Higher α₂ should increase cost for non-zero slope"

    def test_turn_penalty(self) -> None:
        """is_turn=True adds E_rotate = α₁ · (P_rotate/P_forward) · d."""
        cfg = Module3Config(alpha1=1.0, rover_power_w=50.0, rover_power_rotate_w=30.0)
        cost_straight = _edge_cost(1.0, 0.0, cfg, is_turn=False)
        cost_turn = _edge_cost(1.0, 0.0, cfg, is_turn=True)
        rotate_ratio = cfg.rover_power_rotate_w / cfg.rover_power_w  # 0.6
        expected_extra = cfg.alpha1 * rotate_ratio * cfg.grid_resolution_m
        assert abs(cost_turn - cost_straight - expected_extra) < 1e-10


# ---------------------------------------------------------------------------
# Test 2: simple path on a 5×5 flat grid
# ---------------------------------------------------------------------------


class TestPlannerSimplePath:
    """A* should find a path on a fully traversable 5×5 grid."""

    def test_path_found(self) -> None:
        """Path from (0,0) to (4,4) must exist and be non-empty."""
        cfg = Module3Config(max_slope_deg=15.0)
        _, _, sunlit, graph = _flat_grid(5, cfg)
        p_visited = np.zeros((5, 5), dtype=np.float64)
        state = _flat_state(battery_wh=100.0)

        result = plan_path(graph, (0, 0), (4, 4), state, p_visited, cfg, sunlit)

        assert result.feasible, f"Expected feasible path, got: {result.failure_reason}"
        assert len(result.path) >= 2, "Path should have at least start and goal"
        assert result.path[0] == (0, 0), "Path must start at (0,0)"
        assert result.path[-1] == (4, 4), "Path must end at (4,4)"

    def test_path_cost_positive(self) -> None:
        """Total cost must be strictly positive."""
        cfg = Module3Config()
        _, _, sunlit, graph = _flat_grid(5, cfg)
        p_visited = np.zeros((5, 5), dtype=np.float64)
        state = _flat_state(battery_wh=100.0)

        result = plan_path(graph, (0, 0), (4, 4), state, p_visited, cfg, sunlit)
        assert result.total_cost > 0.0

    def test_path_distance_reasonable(self) -> None:
        """Distance from (0,0) to (4,4) is at most 4*sqrt(2) ≈ 5.66 m (diagonal)."""
        cfg = Module3Config()
        _, _, sunlit, graph = _flat_grid(5, cfg)
        p_visited = np.zeros((5, 5), dtype=np.float64)
        state = _flat_state(battery_wh=100.0)

        result = plan_path(graph, (0, 0), (4, 4), state, p_visited, cfg, sunlit)
        assert result.total_distance_m <= 4 * math.sqrt(2) * cfg.grid_resolution_m + 1e-9


# ---------------------------------------------------------------------------
# Test 3: battery-limited infeasibility on 11×11 grid
# ---------------------------------------------------------------------------


class TestPlannerBatteryLimited:
    """When battery is too low the planner must report insufficient_battery."""

    def test_infeasible_battery(self) -> None:
        """Rover starts with near-zero battery; path to (10,10) is infeasible."""
        cfg = Module3Config(
            rover_power_w=500.0,   # high power draw to exhaust tiny battery
            rover_speed_ms=0.1,
            battery_reserve_pct=0.15,
        )
        size = 11
        slope_map = np.zeros((size, size), dtype=np.float64)
        ice_mask = np.zeros((size, size), dtype=bool)
        sunlit_mask = np.zeros((size, size), dtype=bool)
        graph = build_terrain_graph(slope_map, ice_mask, cfg)
        p_visited = np.zeros((size, size), dtype=np.float64)

        # Battery so small that even one step drains below reserve
        tiny_battery = 0.001  # Wh
        state = RoverState(
            position=(0, 0),
            battery_wh=tiny_battery,
            battery_max_wh=100.0,   # max is large so reserve_wh is meaningful
            in_sunlight=False,
            timestep=0,
        )

        result = plan_path(graph, (0, 0), (10, 10), state, p_visited, cfg, sunlit_mask)

        assert not result.feasible, "Expected infeasible result"
        assert result.failure_reason == "insufficient_battery", (
            f"Expected 'insufficient_battery', got '{result.failure_reason}'"
        )


# ---------------------------------------------------------------------------
# Test 4: completely blocked graph → no_path
# ---------------------------------------------------------------------------


class TestPlannerNoPath:
    """When all edges are pruned, planner must return no_path."""

    def test_no_path_all_slopes_at_limit(self) -> None:
        """Set all neighbour cells to slope >= max_slope_deg to block all edges."""
        cfg = Module3Config(max_slope_deg=15.0)
        size = 5
        # All cells at slope equal to max_slope_deg → edges pruned
        slope_map = np.full((size, size), cfg.max_slope_deg, dtype=np.float64)
        # Start cell must be in the graph; set it to 0 so it is a valid node
        slope_map[0, 0] = 0.0
        ice_mask = np.zeros((size, size), dtype=bool)
        sunlit_mask = np.zeros((size, size), dtype=bool)
        graph = build_terrain_graph(slope_map, ice_mask, cfg)
        p_visited = np.zeros((size, size), dtype=np.float64)

        state = _flat_state(battery_wh=100.0)

        result = plan_path(graph, (0, 0), (4, 4), state, p_visited, cfg, sunlit_mask)

        assert not result.feasible, "Expected infeasible result"
        assert result.failure_reason == "no_path", (
            f"Expected 'no_path', got '{result.failure_reason}'"
        )


# ---------------------------------------------------------------------------
# Test 5: revisit decay affects second-run path cost
# ---------------------------------------------------------------------------


class TestRevisitDecay:
    """After decay, P_visited values should be lower, potentially changing cost."""

    def test_revisit_penalty_increases_cost(self) -> None:
        """Second run with P_visited=1 should cost more than with P_visited=0."""
        cfg = Module3Config(gamma=5.0)  # high gamma to amplify penalty
        size = 5
        slope_map = np.zeros((size, size), dtype=np.float64)
        ice_mask = np.zeros((size, size), dtype=bool)
        sunlit_mask = np.zeros((size, size), dtype=bool)
        graph = build_terrain_graph(slope_map, ice_mask, cfg)
        state = _flat_state(battery_wh=100.0)

        # First run: no visited cells
        p_no_visit = np.zeros((size, size), dtype=np.float64)
        result_no_visit = plan_path(graph, (0, 0), (4, 4), state, p_no_visit, cfg, sunlit_mask)

        # Second run: all cells marked as visited
        p_full_visit = np.ones((size, size), dtype=np.float64)
        result_with_visit = plan_path(
            graph, (0, 0), (4, 4), state, p_full_visit, cfg, sunlit_mask
        )

        assert result_no_visit.feasible
        assert result_with_visit.feasible
        assert result_with_visit.total_cost > result_no_visit.total_cost, (
            "Cost with revisit penalty should exceed cost without."
        )

    def test_decay_reduces_penalty(self) -> None:
        """After applying decay factor, P_visited values must decrease."""
        decay = 0.9
        p = np.ones((5, 5), dtype=np.float64)
        p_decayed = p * decay
        assert np.all(p_decayed < p)
        assert np.allclose(p_decayed, 0.9)


# ---------------------------------------------------------------------------
# Test 6: slope clamping at boundary doesn't cause math error
# ---------------------------------------------------------------------------


class TestSlopeClamp:
    """Slope at max_slope_deg - 0.1 must be clamped and not raise any error."""

    def test_slope_near_max_no_error(self) -> None:
        """_edge_cost at slope = max_slope_deg - 0.1 must return a finite value."""
        cfg = Module3Config(max_slope_deg=15.0, slope_clamp_eps=0.5)
        slope = cfg.max_slope_deg - 0.1   # 14.9°, within limit but very steep
        cost = _edge_cost(1.0, slope, cfg)
        assert math.isfinite(cost), f"Expected finite cost, got {cost}"
        assert cost > 0.0

    def test_slope_above_max_is_pruned_from_graph(self) -> None:
        """Cells with slope >= max_slope_deg should produce no outgoing edges."""
        cfg = Module3Config(max_slope_deg=15.0)
        # 3×3 grid: centre cell has slope at limit, surrounded by impassable cells
        slope_map = np.full((3, 3), cfg.max_slope_deg, dtype=np.float64)
        slope_map[1, 1] = 0.0  # only the centre is safe
        ice_mask = np.zeros((3, 3), dtype=bool)
        graph = build_terrain_graph(slope_map, ice_mask, cfg)

        # Centre cell should have NO outgoing edges because all neighbours are at limit
        edges_from_centre = graph["edges"][(1, 1)]
        assert len(edges_from_centre) == 0, (
            f"Expected 0 edges from centre when all neighbours at slope limit, "
            f"got {len(edges_from_centre)}"
        )

    def test_clamp_prevents_div_by_zero(self) -> None:
        """Providing slope exactly equal to max_slope_deg should be clamped safely."""
        cfg = Module3Config(max_slope_deg=15.0, slope_clamp_eps=0.5, alpha2=1.5)
        slope = cfg.max_slope_deg
        cost = _edge_cost(1.0, slope, cfg)
        s_clamp = cfg.max_slope_deg - cfg.slope_clamp_eps
        denom = cfg.max_slope_deg - s_clamp
        s_pen = cfg.beta0 * math.exp(s_clamp / denom)
        expected = cfg.alpha1 * cfg.grid_resolution_m + cfg.alpha2 * s_pen
        assert abs(cost - expected) < 1e-9
        assert math.isfinite(cost)


class TestGridState:
    """Grid cell state model (UNKNOWN / FREE / OBSTACLE / VISITED)."""

    def test_init_all_unknown(self) -> None:
        gs = GridState((5, 5))
        assert gs.n_unknown == 25
        assert gs.n_visited == 0
        assert gs.unknown_fraction() == 1.0

    def test_mark_visited_tracks_cells(self) -> None:
        gs = GridState((5, 5))
        gs.mark_visited([(1, 1), (2, 2), (3, 3)])
        assert gs.n_visited == 3
        assert gs.get_state(1, 1) == CellState.VISITED

    def test_mark_out_of_bounds_safe(self) -> None:
        gs = GridState((3, 3))
        gs.mark_visited([(10, 10)])
        assert gs.n_visited == 0

    def test_summary_keys(self) -> None:
        gs = GridState((4, 4))
        s = gs.summary()
        for k in ("unknown", "free", "obstacle", "visited", "unknown_fraction", "total_cells"):
            assert k in s
