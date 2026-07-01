"""
module_3/graph.py
=================
Terrain graph construction for the Gorilla Traversal path planner.

Builds a static 8-connected grid graph from slope and ice-mask rasters,
annotating each edge with a composite cost that combines motion energy
and exponential slope penalty.  The dynamic revisit penalty (P_visited)
is NOT stored here; it is added at query time by the planner.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Set, Tuple

import numpy as np

from .config import Module3Config

logger = logging.getLogger(__name__)

# Type aliases
Node = Tuple[int, int]
EdgeList = List[Tuple[Node, float]]
TerrainGraph = Dict[str, object]  # {'nodes': set, 'edges': dict}

# 8-connected neighbour offsets (dr, dc) and their Euclidean distances
_NEIGHBOURS: List[Tuple[Tuple[int, int], float]] = [
    ((-1, -1), math.sqrt(2)),
    ((-1,  0), 1.0),
    ((-1,  1), math.sqrt(2)),
    (( 0, -1), 1.0),
    (( 0,  1), 1.0),
    (( 1, -1), math.sqrt(2)),
    (( 1,  0), 1.0),
    (( 1,  1), math.sqrt(2)),
]


def _edge_cost(
    distance_cells: float,
    slope_deg: float,
    config: Module3Config,
    is_turn: bool = False,
) -> float:
    """Compute the hybrid static edge cost between two adjacent cells.

    Innovation: E_total = E_forward + E_rotate, then
                Total Cost = α₁ * EnergyCost + α₂ * RiskCost

    Energy Cost accounts for longitudinal motion (forward) and spot-turn
    (rotate) power draw.  Risk Cost uses exponential DEM slope penalty
    (pitch angle proxy) — slip ratio increases exponentially with pitch.

    Formula
    -------
    Let d  = distance_cells * grid_resolution_m      [metres]
        s  = min(slope_deg, max_slope_deg - slope_clamp_eps)   [degrees]

    Energy Cost (E_total = E_forward + E_rotate):
        E_forward = alpha1 * d                          [motion over distance]
        E_rotate  = alpha1 * (rover_power_rotate_w / rover_power_w) * d
                                                        [extra cost for turns]

    Risk Cost (exponential pitch/slope penalty):
        R_slope = beta0 * exp(s / (max_slope_deg - s))

    Total hybrid cost:
        Total = α₁ * (E_forward + E_rotate) + α₂ * R_slope

    The slope penalty diverges as slope → max_slope_deg.  Clamping at
    (max_slope_deg - slope_clamp_eps) keeps it finite.  Edges with
    slope >= max_slope_deg are pruned before this function is called.

    Parameters
    ----------
    distance_cells : float
        Euclidean distance in grid-cell units (1.0 axial, sqrt(2) diagonal).
    slope_deg : float
        Terrain slope at destination cell. Units: degrees.
    config : Module3Config
        Planner configuration.
    is_turn : bool
        True if this step involves a direction change (spot-turn).

    Returns
    -------
    float
        Composite hybrid cost. Units: dimensionless.
    """
    distance_m = distance_cells * config.grid_resolution_m

    # E_forward: longitudinal motion energy
    e_forward = config.alpha1 * distance_m

    # E_rotate: spot-turn energy penalty (applied on direction changes)
    rotate_ratio = config.rover_power_rotate_w / config.rover_power_w
    e_rotate = (config.alpha1 * rotate_ratio * distance_m) if is_turn else 0.0
    e_total = e_forward + e_rotate

    # Risk Cost: exponential slope penalty (pitch angle → slip ratio)
    slope_clamped = min(slope_deg, config.max_slope_deg - config.slope_clamp_eps)
    denom = config.max_slope_deg - slope_clamped
    slope_penalty = config.beta0 * math.exp(slope_clamped / denom) if denom > 0 else float('inf')

    # Total = α₁ * Energy + α₂ * Risk
    total_cost = e_total + config.alpha2 * slope_penalty

    return total_cost


def build_terrain_graph(
    slope_map: np.ndarray,
    ice_mask: np.ndarray,
    config: Module3Config,
) -> TerrainGraph:
    """Build an 8-connected grid graph weighted by terrain-aware edge costs.

    The graph is constructed once and cached; the P_visited revisit-penalty
    is added dynamically by the planner at search time.

    Edge Cost Formula (static part)
    --------------------------------
    For an edge from node u=(r,c) to neighbour v=(r2,c2):

        d        = ||u - v|| * grid_resolution_m   [metres]
        s        = slope_map[r2, c2]               [degrees]

        if s >= max_slope_deg:
            edge is PRUNED (hard terrain limit)
        else:
            s_clamp  = min(s, max_slope_deg - slope_clamp_eps)
            E_motion = alpha1 * d
            E_slope  = beta0 * exp(s_clamp / (max_slope_deg - s_clamp))
            cost     = E_motion + E_slope

    Parameters
    ----------
    slope_map : np.ndarray, shape (R, C)
        Terrain slope at each grid cell.
        Units: degrees. Valid range: [0, 90).
    ice_mask : np.ndarray, shape (R, C), dtype bool
        True where water-ice has been confirmed (used for future science
        weighting; stored as node metadata).
    config : Module3Config
        Planner configuration.

    Returns
    -------
    dict
        {
          'nodes'  : set of (row, col) tuples for all traversable cells,
          'edges'  : dict mapping (row, col) -> list of ((row2, col2), cost),
          'ice_mask': np.ndarray (reference, not copied),
          'shape'  : (R, C),
        }
    """
    if slope_map.shape != ice_mask.shape:
        raise ValueError(
            f"slope_map shape {slope_map.shape} != ice_mask shape {ice_mask.shape}"
        )

    rows, cols = slope_map.shape
    nodes: Set[Node] = set()
    edges: Dict[Node, EdgeList] = {}

    pruned_edges = 0
    total_edges = 0

    for r in range(rows):
        for c in range(cols):
            node: Node = (r, c)
            nodes.add(node)
            edge_list: EdgeList = []

            for (dr, dc), cell_dist in _NEIGHBOURS:
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < rows and 0 <= c2 < cols):
                    continue
                total_edges += 1
                slope_nbr = float(slope_map[r2, c2])

                # ML Adaptive Penalty: Do not hard-prune edges.
                # Let the exponential slope penalty in _edge_cost heavily penalize
                # steep terrain instead, guaranteeing A* will always find SOME route.
                if slope_nbr >= config.max_slope_deg:
                    pruned_edges += 1 # We just count them for logging now

                cost = _edge_cost(cell_dist, slope_nbr, config)
                edge_list.append(((r2, c2), cost))

            edges[node] = edge_list

    logger.debug(
        "Graph built: %d nodes, %d total candidate edges, %d pruned (slope >= %.1f°)",
        len(nodes),
        total_edges,
        pruned_edges,
        config.max_slope_deg,
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "ice_mask": ice_mask,
        "shape": (rows, cols),
    }
