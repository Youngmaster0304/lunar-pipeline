"""
config.py — Global Pipeline Configuration
==========================================
Top-level orchestration configuration for the Lunar South Pole Autonomous
Exploration Pipeline (Team Thunderbolts, ISRO Space Tech Hackathon 2024).

All module-level configurations are embedded as nested dataclass fields.
Global settings (paths, mission geometry, battery limits) are defined here.

Usage
-----
::

    from config import PipelineConfig
    cfg = PipelineConfig()                # all defaults
    cfg = PipelineConfig(log_level='DEBUG')

The :class:`PipelineConfig` dataclass is the single source of truth for
the entire pipeline run.  Pass ``cfg.module_N`` to each module's functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from module_1.config import Module1Config
from module_2.config import Module2Config
from module_3.config import Module3Config
from module_4.config import Module4Config
from module_5.config import Module5Config
from module_6.config import Module6Config


@dataclass
class PipelineConfig:
    """Master configuration for the Lunar South Pole Exploration Pipeline.

    Attributes
    ----------
    module1 : Module1Config
        Configuration for Module 1 (DFSAR Radar Polarimetric Decomposition).
    module2 : Module2Config
        Configuration for Module 2 (Landing Site Seeding / MCDA).
    module3 : Module3Config
        Configuration for Module 3 (Gorilla Traversal Path Planner).
    module4 : Module4Config
        Configuration for Module 4 (Reactive Obstacle Avoidance & SLAM).
    module5 : Module5Config
        Configuration for Module 5 (EFPI In-Situ Ice Sensing).
    dem_pixel_spacing_m : float
        Physical pixel spacing of the input Digital Elevation Model (DEM).
        Corresponds to Chandrayaan-2 TMC-2 DEM product resolution.
        Units: metres.  Default: 5.0 m.
    dfsar_tile_path : str
        Filesystem path to the DFSAR Chandrayaan-2 compact-polarimetry product
        (GeoTIFF or HDF5).  If the file does not exist, the pipeline uses
        synthetic fixtures automatically.
    dem_path : str
        Filesystem path to the DEM raster for the Faustini Crater region.
        Units: metres elevation.  If absent → synthetic.
    dte_path : str
        Filesystem path to the Direct-To-Earth (DTE) boolean raster.
        Values: 1 (DTE available) or 0 (DTE occluded).  If absent → synthetic.
    illumination_path : str
        Filesystem path to the solar illumination fraction raster.
        Values: [0.0, 1.0] fraction of time pixel is sunlit.  If absent →
        synthetic.
    rover_battery_max_wh : float
        Maximum rover battery capacity.
        Units: Watt-hours (Wh).  Default: 200 Wh.
    rover_battery_initial_wh : float
        Initial rover battery state-of-charge at mission start.
        Units: Watt-hours (Wh).  Default: 180 Wh (90 % SoC).
    mission_start : tuple[int, int]
        Mission start position as (row, col) grid indices within the DEM.
        Default: (50, 50) — centred in a 100×100 synthetic grid.
    dsc_sample_point : tuple[int, int]
        Target sampling location (Doubly Shadowed Crater entry point) as
        (row, col) grid indices.
        Default: (80, 80).
    log_level : str
        Python logging level for the entire pipeline.
        Allowed values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        Default: 'INFO'.
    """

    # --- Per-module configurations ---
    module1: Module1Config = field(default_factory=Module1Config)
    module2: Module2Config = field(default_factory=Module2Config)
    module3: Module3Config = field(default_factory=Module3Config)
    module4: Module4Config = field(default_factory=Module4Config)
    module5: Module5Config = field(default_factory=Module5Config)
    module6: Module6Config = field(default_factory=Module6Config)

    # --- DEM / raster geometry ---
    dem_pixel_spacing_m: float = 5.0          # Chandrayaan-2 TMC-2 resolution

    # --- Data file paths (absent → synthetic) ---
    dfsar_tile_path: str = "data/dfsar_tile.tif"
    dem_path: str = "data/dem_faustini.tif"
    dte_path: str = "data/dte_faustini.tif"
    illumination_path: str = "data/illumination_faustini.tif"
    ohrc_path: str = "data/ohrc_faustini.tif"

    # --- Rover energy budget ---
    rover_battery_max_wh: float = 200.0
    rover_battery_initial_wh: float = 180.0

    # --- Mission geometry ---
    dsc_name: str = "Crater F2"
    dsc_elevation_m: float = -2860.0
    dsc_description: str = (
        "Doubly-shadowed crater F2 within Faustini PSR (~1.1 km diameter, "
        "lobate-rim morphology, ~180 m below Faustini floor, ~300 m below "
        "upper Faustini surface).  Lobate rim is the most prominent among "
        "all lunar polar DSCs, indicating deep-excavated subsurface ice."
    )
    mission_start: tuple = (50, 50)            # (row, col) in DEM grid
    dsc_sample_point: tuple = (80, 80)         # Doubly Shadowed Crater target (F2)

    # --- Logging ---
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate global configuration parameters at construction time.

        Raises
        ------
        ValueError
            If battery parameters are invalid, mission coordinates are negative,
            or log level is unrecognised.
        """
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in allowed_levels:
            raise ValueError(
                f"log_level must be one of {allowed_levels}; got '{self.log_level}'"
            )
        if self.rover_battery_max_wh <= 0:
            raise ValueError(
                f"rover_battery_max_wh must be > 0; got {self.rover_battery_max_wh}"
            )
        if not (0 < self.rover_battery_initial_wh <= self.rover_battery_max_wh):
            raise ValueError(
                f"rover_battery_initial_wh must be in (0, rover_battery_max_wh="
                f"{self.rover_battery_max_wh}]; got {self.rover_battery_initial_wh}"
            )
        if self.dem_pixel_spacing_m <= 0:
            raise ValueError(
                f"dem_pixel_spacing_m must be > 0; got {self.dem_pixel_spacing_m}"
            )
        for coord_name, coord in (
            ("mission_start", self.mission_start),
            ("dsc_sample_point", self.dsc_sample_point),
        ):
            if len(coord) != 2 or any(c < 0 for c in coord):
                raise ValueError(
                    f"{coord_name} must be a 2-tuple of non-negative integers; "
                    f"got {coord}"
                )
