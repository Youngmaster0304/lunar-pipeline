"""
module_3/grid_state.py
======================
Grid cell state model for rover exploration mapping.

Each cell in the discretized environment starts as UNKNOWN.  As the rover
traverses, cells are classified into FREE, OBSTACLE, or VISITED.  This
mapping is used by the dynamic cost function to penalize revisits and
to guide the rover toward unexplored (UNKNOWN) ice-rich areas.

States
------
UNKNOWN  : initial state — not yet observed by any rover sensor.
FREE     : sensed and confirmed traversable with no hazard.
OBSTACLE : sensed and contains a hazard (boulder, steep wall, etc.).
VISITED  : the rover has physically driven through this cell.

Innovation: Grid Decomposition
The environment is discretized into n = rows × cols cells.
Each cell's state informs the dynamic revisit penalty in the A* planner.
"""
from __future__ import annotations

import logging
from enum import IntEnum
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CellState(IntEnum):
    """Enum for grid cell state (numeric for array performance)."""
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    VISITED = 3


class GridState:
    """Tracks the state of every cell in the discretized environment.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(rows, cols)`` of the grid.
    """

    def __init__(self, shape: tuple[int, int]) -> None:
        self._grid: np.ndarray = np.full(shape, CellState.UNKNOWN, dtype=np.int8)
        self._shape: tuple[int, int] = shape
        logger.debug("GridState initialised: shape=%s, all UNKNOWN", shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def get_state(self, row: int, col: int) -> CellState:
        return CellState(int(self._grid[row, col]))

    def set_state(self, row: int, col: int, state: CellState) -> None:
        self._grid[row, col] = int(state)

    def mark_visited(self, path: List[Tuple[int, int]]) -> None:
        for r, c in path:
            if 0 <= r < self._shape[0] and 0 <= c < self._shape[1]:
                self._grid[r, c] = int(CellState.VISITED)

    def mark_obstacle(self, row: int, col: int) -> None:
        if 0 <= row < self._shape[0] and 0 <= col < self._shape[1]:
            self._grid[row, col] = int(CellState.OBSTACLE)

    def mark_free(self, row: int, col: int) -> None:
        if 0 <= row < self._shape[0] and 0 <= col < self._shape[1]:
            if self._grid[row, col] == int(CellState.UNKNOWN):
                self._grid[row, col] = int(CellState.FREE)

    @property
    def n_unknown(self) -> int:
        return int(np.sum(self._grid == int(CellState.UNKNOWN)))

    @property
    def n_visited(self) -> int:
        return int(np.sum(self._grid == int(CellState.VISITED)))

    @property
    def n_free(self) -> int:
        return int(np.sum(self._grid == int(CellState.FREE)))

    @property
    def n_obstacle(self) -> int:
        return int(np.sum(self._grid == int(CellState.OBSTACLE)))

    def unknown_fraction(self) -> float:
        return self.n_unknown / (self._shape[0] * self._shape[1])

    def summary(self) -> dict:
        return {
            "unknown": self.n_unknown,
            "free": self.n_free,
            "obstacle": self.n_obstacle,
            "visited": self.n_visited,
            "unknown_fraction": self.unknown_fraction(),
            "total_cells": self._shape[0] * self._shape[1],
        }
