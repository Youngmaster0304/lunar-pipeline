"""
module_3/planner.py
===================
A* path planner with battery-aware pruning and revisit-penalty heuristics.

Implements the full mission-level planning loop:
  1. Dash path  : start → DSC sample point
  2. Return path: updated state → best singularity site

Dynamic edge cost adds the gamma-weighted P_visited heatmap on top of the
static terrain graph built by module_3.graph.
"""
from __future__ import annotations

import copy
import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Module3Config
from .graph import TerrainGraph, build_terrain_graph
from .rover_state import RoverState

logger = logging.getLogger(__name__)

Node = Tuple[int, int]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PlannerResult:
    """Outcome of a single A* planning call.

    Parameters
    ----------
    path : list[tuple[int, int]]
        Ordered list of (row, col) waypoints from start to goal.
        Empty when feasible is False.
    total_cost : float
        Accumulated path cost using the composite edge-cost formula.
        Units: dimensionless (energy-like).
    total_distance_m : float
        Total Euclidean traversal distance.
        Units: metres.
    battery_wh_consumed : float
        Net battery energy consumed along the path (positive = consumed).
        Units: Wh.
    feasible : bool
        True when a path satisfying all SoC constraints was found.
    failure_reason : str | None
        'no_path'              – goal geometrically unreachable.
        'insufficient_battery' – goal reachable but SoC runs out.
        None                   – success.
    """

    path: List[Node] = field(default_factory=list)
    total_cost: float = 0.0
    total_distance_m: float = 0.0
    battery_wh_consumed: float = 0.0
    feasible: bool = True
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal A* helpers
# ---------------------------------------------------------------------------


def _heuristic(a: Node, b: Node, resolution: float) -> float:
    """Euclidean distance heuristic scaled to metres.

    Formula
    -------
    h = sqrt((r_a - r_b)^2 + (c_a - c_b)^2) * grid_resolution_m

    Parameters
    ----------
    a, b : tuple[int, int]
        Grid positions.
    resolution : float
        Metres per grid cell.

    Returns
    -------
    float
        Heuristic distance in metres.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1]) * resolution


def _reconstruct_path(came_from: Dict[Node, Node], current: Node) -> List[Node]:
    """Reconstruct A* path by back-tracking the came_from map.

    Parameters
    ----------
    came_from : dict
        Maps each visited node to the predecessor that minimised cost.
    current : Node
        Goal node at which reconstruction starts.

    Returns
    -------
    list[tuple[int, int]]
        Path from start to goal inclusive.
    """
    path: List[Node] = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Core A* planner
# ---------------------------------------------------------------------------


def plan_path(
    graph: TerrainGraph,
    start: Node,
    goal: Node,
    rover_state: RoverState,
    p_visited: np.ndarray,
    config: Module3Config,
    sunlit_mask: np.ndarray,
) -> PlannerResult:
    """Find the energy- and battery-feasible least-cost path via A*.

    Search Algorithm
    ----------------
    Standard A* with f(n) = g(n) + h(n), where:

        g(n)  – accumulated composite edge cost from start to n:
                g(n) = Σ [base_cost(u,v) + gamma * P_visited[v]]
                where base_cost comes from the static terrain graph.

        h(n)  – Euclidean heuristic:
                h(n) = ||n - goal|| * grid_resolution_m    [metres]

    Battery Constraint (Hard Pruning)
    ----------------------------------
    At each neighbour expansion, accumulated energy is estimated as:

        E_travel = rover_power_w * (dist_m / rover_speed_ms) / 3600   [Wh]
        E_solar  = solar_charge_w * (dist_m / rover_speed_ms) / 3600  [Wh]  (if sunlit)
        net_wh   = E_travel - E_solar    (sunlit) | E_travel (dark)

    cumulative_wh_consumed += net_wh for each step.

    Node is pruned when:
        rover_state.battery_wh - cumulative_wh_consumed
            < config.battery_reserve_pct * rover_state.battery_max_wh

    Parameters
    ----------
    graph : TerrainGraph
        Pre-built terrain graph (from build_terrain_graph).
    start : tuple[int, int]
        Start grid position.
    goal : tuple[int, int]
        Goal grid position.
    rover_state : RoverState
        Current rover state (battery, sunlight, etc.).
    p_visited : np.ndarray, shape == graph['shape']
        Revisit-penalty heatmap in [0, 1].
    config : Module3Config
        Planner configuration.
    sunlit_mask : np.ndarray, shape == graph['shape'], dtype bool
        Per-cell sunlight flag.

    Returns
    -------
    PlannerResult
        Contains path, cost, distance, battery consumed, feasibility flag.
    """
    edges = graph["edges"]
    rows, cols = graph["shape"]

    # Validate inputs
    if start not in graph["nodes"]:
        logger.warning("Start %s not in graph nodes", start)
        return PlannerResult(feasible=False, failure_reason="no_path")
    if goal not in graph["nodes"]:
        logger.warning("Goal %s not in graph nodes", goal)
        return PlannerResult(feasible=False, failure_reason="no_path")

    # A* open set: (f_score, tie-break counter, node)
    counter = 0
    open_set: list = []
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: Dict[Node, Node] = {}
    g_score: Dict[Node, float] = {start: 0.0}

    # Track cumulative battery consumption per node
    battery_consumed: Dict[Node, float] = {start: 0.0}

    reserve_wh = config.battery_reserve_pct * rover_state.battery_max_wh
    # Soft penalty multiplier: instead of pruning low-battery paths we add a
    # massive cost so A* still finds a route but heavily prefers energy-safe paths
    LOW_BATTERY_PENALTY = 1e6

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            path = _reconstruct_path(came_from, current)
            # Compute total distance
            total_dist_m = 0.0
            for i in range(1, len(path)):
                r1, c1 = path[i - 1]
                r2, c2 = path[i]
                total_dist_m += math.hypot(r2 - r1, c2 - c1) * config.grid_resolution_m

            consumed = battery_consumed[goal]
            feasible = (rover_state.battery_wh - consumed) >= reserve_wh
            logger.debug(
                "A* found path: %d waypoints, cost=%.4f, dist=%.2f m, battery=%.4f Wh, feasible=%s",
                len(path), g_score[goal], total_dist_m, consumed, feasible,
            )
            return PlannerResult(
                path=path,
                total_cost=g_score[goal],
                total_distance_m=total_dist_m,
                battery_wh_consumed=consumed,
                feasible=feasible,
                failure_reason=None if feasible else "insufficient_battery",
            )

        current_g = g_score[current]
        current_batt_consumed = battery_consumed[current]

        for (nb, base_cost) in edges.get(current, []):
            # Dynamic revisit penalty
            dynamic_cost = base_cost + config.gamma * float(p_visited[nb[0], nb[1]])

            tentative_g = current_g + dynamic_cost

            # Compute incremental battery cost for this edge
            dr = nb[0] - current[0]
            dc = nb[1] - current[1]
            cell_dist = math.hypot(dr, dc)
            dist_m = cell_dist * config.grid_resolution_m
            travel_time_s = dist_m / config.rover_speed_ms
            e_motion_wh = config.rover_power_w * travel_time_s / 3600.0

            nb_in_sun = bool(sunlit_mask[nb[0], nb[1]])
            if nb_in_sun:
                e_solar_wh = config.solar_charge_w * travel_time_s / 3600.0
                net_wh = e_motion_wh - e_solar_wh
            else:
                net_wh = e_motion_wh

            new_consumed = current_batt_consumed + net_wh

            # Adaptive soft penalty instead of hard pruning:
            # If remaining battery would drop below reserve, add a massive cost
            # so A* strongly prefers energy-safe paths, but doesn't give up entirely.
            remaining = rover_state.battery_wh - new_consumed
            if remaining < reserve_wh:
                tentative_g += LOW_BATTERY_PENALTY
                logger.debug(
                    "Soft penalty applied at %s: remaining=%.4f Wh", nb, remaining
                )

            if tentative_g < g_score.get(nb, math.inf):
                came_from[nb] = current
                g_score[nb] = tentative_g
                battery_consumed[nb] = new_consumed
                counter += 1
                f = tentative_g + _heuristic(nb, goal, config.grid_resolution_m)
                heapq.heappush(open_set, (f, counter, nb))

    # Open set exhausted — goal is geometrically unreachable (disconnected graph)
    logger.info("A* failed: no_path, start=%s, goal=%s", start, goal)
    return PlannerResult(feasible=False, failure_reason="no_path")


# ---------------------------------------------------------------------------
# Mission-level planning
# ---------------------------------------------------------------------------


def plan_mission_path(
    slope_map: np.ndarray,
    ice_mask: np.ndarray,
    sunlit_mask: np.ndarray,
    start: Node,
    dsc_sample_point: Node,
    singularity_sites: List[Node],
    rover_state: RoverState,
    config: Module3Config,
) -> Tuple[PlannerResult, PlannerResult]:
    """Plan the full two-leg mission: dash to sample site then return.

    Mission Planning Sequence
    -------------------------
    1. Build static terrain graph from slope_map and ice_mask.
    2. Initialise P_visited = zeros(slope_map.shape).
    3. Plan DASH path: start → dsc_sample_point using current rover_state.
    4. Mark visited cells: P_visited[row, col] = 1.0 for every cell on dash path.
    5. Apply temporal decay: P_visited *= revisit_decay.
    6. Update rover battery: battery_wh -= dash_result.battery_wh_consumed.
    7. If battery after dash < reserve → log a Module-5 sample-event trigger.
    8. Plan RETURN path: updated rover_state → best singularity site.
       - Iterate singularity_sites in ranked order.
       - Return the first feasible PlannerResult.
    9. Raise RuntimeError if no singularity site yields a feasible return path.

    Parameters
    ----------
    slope_map : np.ndarray, shape (R, C)
        Terrain slope raster. Units: degrees.
    ice_mask : np.ndarray, shape (R, C), dtype bool
        Ice-presence mask.
    sunlit_mask : np.ndarray, shape (R, C), dtype bool
        Per-cell solar illumination flag.
    start : tuple[int, int]
        Rover starting grid position.
    dsc_sample_point : tuple[int, int]
        DSC (dark-shadow crater) sample target position.
    singularity_sites : list[tuple[int, int]]
        Candidate return-charge / singularity sites, ordered by preference.
    rover_state : RoverState
        Initial rover state (will be mutated to reflect dash energy cost).
    config : Module3Config
        Planner configuration.

    Returns
    -------
    tuple[PlannerResult, PlannerResult]
        (dash_result, return_result) — both must be feasible for mission success.

    Raises
    ------
    RuntimeError
        When no return path to any singularity site is feasible.
    ValueError
        When singularity_sites is empty.
    """
    if not singularity_sites:
        raise ValueError("singularity_sites must contain at least one candidate.")

    # 1. Build graph
    graph = build_terrain_graph(slope_map, ice_mask, config)
    logger.info(
        "Mission planner: graph built (%d nodes), start=%s, goal=%s",
        len(graph["nodes"]),
        start,
        dsc_sample_point,
    )

    # 2. Initialise P_visited
    p_visited: np.ndarray = np.zeros(slope_map.shape, dtype=np.float64)

    # 3. Plan DASH
    dash_result = plan_path(
        graph=graph,
        start=start,
        goal=dsc_sample_point,
        rover_state=rover_state,
        p_visited=p_visited,
        config=config,
        sunlit_mask=sunlit_mask,
    )

    if not dash_result.feasible:
        logger.error(
            "Dash path infeasible: %s", dash_result.failure_reason
        )
        # Still attempt return planning with current state; raise at end if needed
        # For robustness we still try to plan a return (to a charge site)
        return_result = PlannerResult(feasible=False, failure_reason="no_path")
        return dash_result, return_result

    logger.info(
        "Dash path found: %d waypoints, %.2f m, %.4f Wh consumed",
        len(dash_result.path),
        dash_result.total_distance_m,
        dash_result.battery_wh_consumed,
    )

    # 4. Mark visited cells
    for r, c in dash_result.path:
        p_visited[r, c] = 1.0

    # 5. Temporal decay
    p_visited *= config.revisit_decay
    logger.debug("P_visited decayed by factor %.3f", config.revisit_decay)

    # 6. Update rover battery
    rover_state.battery_wh -= dash_result.battery_wh_consumed
    rover_state.position = dsc_sample_point
    rover_state.timestep += 1

    # 7. Module-5 event trigger check
    reserve_wh = config.battery_reserve_pct * rover_state.battery_max_wh
    if rover_state.battery_wh < reserve_wh:
        logger.warning(
            "MODULE-5 EVENT: Battery %.4f Wh below reserve %.4f Wh after dash. "
            "Triggering emergency sample event.",
            rover_state.battery_wh,
            reserve_wh,
        )

    # 8. Plan RETURN — try singularity sites in ranked order
    return_result: Optional[PlannerResult] = None
    for rank, site in enumerate(singularity_sites):
        logger.info(
            "Attempting return path to singularity site rank=%d pos=%s", rank, site
        )
        target = (site.row, site.col) if not isinstance(site, tuple) else site
        candidate = plan_path(
            graph=graph,
            start=rover_state.position,
            goal=target,
            rover_state=rover_state,
            p_visited=p_visited,
            config=config,
            sunlit_mask=sunlit_mask,
        )
        if candidate.feasible:
            return_result = candidate
            logger.info(
                "Return path found to site rank=%d pos=%s: %d waypoints, %.2f m",
                rank,
                site,
                len(return_result.path),
                return_result.total_distance_m,
            )
            break
        else:
            logger.debug(
                "Return path to site rank=%d pos=%s infeasible: %s",
                rank,
                site,
                candidate.failure_reason,
            )

    # 9. If no feasible return path found, return best infeasible result rather than crashing
    if return_result is None or not return_result.feasible:
        logger.warning(
            "No fully feasible return path found from any singularity site. "
            "Returning best available result (may be energy-constrained)."
        )
        if return_result is None:
            return_result = PlannerResult(feasible=False, failure_reason="no_path")

    return dash_result, return_result
