"""
module_3/__init__.py
====================
Public API for Module 3 – Hybrid Risk-Aware Path Planner ("Gorilla Traversal").

Exports
-------
RoverState        : Runtime rover state dataclass.
TerrainGraph      : Type alias for the terrain graph dict.
plan_mission_path : Mission-level two-leg planner (dash + return).
PlannerResult     : Result container for a single A* planning call.
GridState         : Grid cell state tracker (unknown/free/obstacle/visited).
CellState         : Enum for grid cell states.
"""
from __future__ import annotations

from .config import Module3Config
from .graph import TerrainGraph, build_terrain_graph
from .grid_state import GridState, CellState
from .planner import PlannerResult, plan_mission_path, plan_path
from .rover_state import RoverState

__all__ = [
    "Module3Config",
    "RoverState",
    "TerrainGraph",
    "plan_mission_path",
    "PlannerResult",
    "build_terrain_graph",
    "plan_path",
    "GridState",
    "CellState",
]
