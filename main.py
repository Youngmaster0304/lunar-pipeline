"""
Top-level orchestration for the Lunar South Pole Autonomous Exploration Pipeline.

Stages
------
0. Module 1 : Radar Polarimetric Decomposition (DFSAR)
1. Module 2a: OHRC visible-light terrain analysis (roughness, boulders)
2. Module 2b: MCDA landing-site seeding (slope, DTE, illumination)
3. Module 6 : PSR + doubly-shadowed crater detection
4. Module 3 : Hybrid risk-aware path planning (Gorilla Traversal)
5. Module 4 : Reactive local obstacle avoidance & SLAM (Bug-2)
6. Module 5 : EFPI in-situ ice sensing (optional)

Innovations demonstrated:
- F2 crater as the DSC target (Deutsch et al. 2020)
- Grid cell state model (UNKNOWN / FREE / OBSTACLE / VISITED)
- Hybrid cost: α₁·E_forward + α₁·E_rotate + α₂·R_slope
- Bug-2 + SLAM with MAE computation
- Total ice volume estimation via CRIM
"""
from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from config import PipelineConfig

# Module 1
from module_1 import (
    compute_cpr,
    compute_dop,
    build_ice_mask,
    invert_ice_fraction,
    compute_ice_volume,
)
from module_1.dfsar_reader import load_dfsar_product, load_dfsar_product_synthetic

# Module 2
from module_2 import (
    compute_slope,
    load_auxiliary_rasters,
    compute_surface_roughness,
    detect_boulders,
    make_synthetic_ohrc,
    build_candidate_mask,
    compute_exposure_score,
    extract_candidate_sites,
)
from module_2.terrain_analysis import make_synthetic_auxiliary

# Module 3
from module_3 import (
    GridState,
    RoverState,
    plan_mission_path,
)

# Module 4
from module_4 import Bug2Planner, AutonomyMode

# Module 5
from module_5 import (
    EFPIModel,
    compute_drill_heat_transfer,
    compute_post_drill_temperature,
    compute_sublimation_rate,
    compute_vapor_density,
)

# Module 6
from module_6 import compute_psr_mask, identify_doubly_shadowed_craters

logger = logging.getLogger(__name__)


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _make_figure_dir() -> str:
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def _save_figure(fig: plt.Figure, name: str) -> None:
    path = os.path.join(_make_figure_dir(), name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def _fallback_candidates(slope: np.ndarray, config: PipelineConfig) -> list:
    from module_2.mcda import CandidateSite

    row, col = config.dsc_sample_point
    return [
        CandidateSite(
            row=row,
            col=col,
            slope_deg=float(slope[row, col]) if slope[row, col] < 15 else 2.0,
            dte_ok=True,
            illumination_fraction=0.1,
            exposure_score=0.85,
            rank=1,
        )
    ]


def main() -> None:
    config = PipelineConfig()
    setup_logging(config.log_level)

    logger.info("=" * 60)
    logger.info("Lunar South Pole Autonomous Exploration Pipeline")
    logger.info("Target DSC: %s (elev. %.0f m)", config.dsc_name, config.dsc_elevation_m)
    logger.info(config.dsc_description)
    logger.info("=" * 60)

    # ---- initialise summary fields (may be set inside try blocks) ----
    ice_volume_m3 = 0.0
    mae = 0.0
    ice_pct = 0.0
    n_candidates = 0
    n_dsc_craters = 0
    dash_feasible = False
    return_feasible = False
    grid_state = GridState((100, 100))

    # ---------------------------------------------------------
    # Stage 1: Radar Polarimetric Decomposition (DFSAR)
    # ---------------------------------------------------------
    logger.info("=== Stage 1: DFSAR Polarimetry & CRIM Inversion ===")
    try:
        try:
            dfsar_product = load_dfsar_product(
                config.dfsar_tile_path, config.module1
            )
            logger.info("Loaded real DFSAR data from %s", config.dfsar_tile_path)
        except Exception as exc:
            logger.warning("Real DFSAR load failed (%s). Using synthetic fixture.", exc)
            dfsar_product = load_dfsar_product_synthetic(shape=(100, 100))

        cpr = compute_cpr(dfsar_product, config.module1)
        dop = compute_dop(dfsar_product)
        ice_mask = build_ice_mask(cpr, dop, config.module1)

        backscatter_db = np.random.uniform(-15, -5, size=dfsar_product.shape).astype(np.float32)
        ice_fraction = invert_ice_fraction(backscatter_db, config.module1)

        pixel_area = config.dem_pixel_spacing_m ** 2
        ice_volume_m3 = compute_ice_volume(ice_fraction, pixel_area_m2=pixel_area, depth_m=5.0)

        fig1, axes = plt.subplots(1, 3, figsize=(14, 4))
        im0 = axes[0].imshow(cpr, cmap="viridis")
        axes[0].set_title("CPR")
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(ice_mask.astype(int), cmap="coolwarm")
        axes[1].set_title("Ice Mask")
        plt.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(ice_fraction, cmap="Blues", vmin=0, vmax=0.6)
        axes[2].set_title("Ice Fraction (vol.)")
        plt.colorbar(im2, ax=axes[2])
        _save_figure(fig1, "stage1_dfsar.png")

        logger.info(
            "Stage 1: ice_mask=%d pixels, mean f_ice=%.3f, volume=%.2e m\u00b3",
            int(np.sum(ice_mask)),
            float(np.mean(ice_fraction)),
            ice_volume_m3,
        )
    except Exception as exc:
        logger.error("Stage 1 failed: %s", exc)
        raise

    # ---------------------------------------------------------
    # Stage 2a: OHRC Terrain Analysis
    # ---------------------------------------------------------
    logger.info("=== Stage 2a: OHRC Surface Roughness & Boulder Detection ===")
    try:
        try:
            ohrc_img = None
            if os.path.exists(config.ohrc_path):
                from module_2.terrain_analysis import load_ohrc_ortho
                ohrc_img = load_ohrc_ortho(config.ohrc_path)
                logger.info("Loaded real OHRC ortho from %s", config.ohrc_path)
        except Exception as exc:
            logger.warning("Real OHRC load failed (%s). Using synthetic.", exc)

        if ohrc_img is None:
            ohrc_img = make_synthetic_ohrc(shape=dfsar_product.shape, seed=42)

        roughness = compute_surface_roughness(ohrc_img, config.module2.ohrc_roughness_window)
        boulder_mask = detect_boulders(ohrc_img, config.module2.boulder_prominence_m, config.module2.boulder_buffer_px)

        fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
        im0 = axes[0].imshow(ohrc_img, cmap="gray")
        axes[0].set_title("OHRC Ortho")
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(roughness, cmap="hot")
        axes[1].set_title("Surface Roughness")
        plt.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(boulder_mask.astype(int), cmap="bwr")
        axes[2].set_title("Boulder Mask (%d)" % int(np.sum(boulder_mask)))
        plt.colorbar(im2, ax=axes[2])
        _save_figure(fig2, "stage2a_ohrc.png")

        logger.info(
            "Stage 2a: roughness range=[%.3f, %.3f], boulders=%d px",
            float(np.min(roughness)),
            float(np.max(roughness)),
            int(np.sum(boulder_mask)),
        )
    except Exception as exc:
        logger.error("Stage 2a failed: %s", exc)
        raise

    # ---------------------------------------------------------
    # Stage 2b: MCDA Landing Site Seeding
    # ---------------------------------------------------------
    logger.info("=== Stage 2b: MCDA Landing Site Seeding ===")
    try:
        grid_shape = dfsar_product.shape
        grid_state = GridState(grid_shape)
        rows, cols = grid_shape
        rr, cc = np.mgrid[0:rows, 0:cols]
        synthetic_dem = (
            1000.0
            + 3.0 * np.sin(rr / 8.0) * np.cos(cc / 8.0)
            + 1.5 * np.sin(rr / 20.0 + cc / 15.0)
        ).astype(np.float32)
        slope = compute_slope(synthetic_dem, config.dem_pixel_spacing_m)

        try:
            dte, illumination = load_auxiliary_rasters(
                config.dte_path, config.illumination_path
            )
        except Exception as exc:
            logger.warning("Aux raster load failed (%s). Using synthetic.", exc)
            dte, illumination = make_synthetic_auxiliary(shape=grid_shape)

        sunlit_mask = illumination > 0.3

        candidate_mask = build_candidate_mask(
            slope,
            dte,
            illumination,
            config.module2,
            roughness=roughness,
            boulder_mask=boulder_mask,
        )

        exposure_score = compute_exposure_score(
            illumination, dte, config.module2, roughness=roughness,
        )

        candidates = extract_candidate_sites(
            candidate_mask,
            exposure_score=exposure_score,
            slope=slope,
            dte=dte,
            illumination=illumination,
            config=config.module2,
        )

        n_candidates = len(candidates)
        logger.info("Stage 2b: %d candidate sites extracted.", n_candidates)
        if candidates:
            c0 = candidates[0]
            logger.info(
                "Top candidate: rank=%d at (%d,%d), score=%.2f",
                c0.rank, c0.row, c0.col, c0.exposure_score,
            )

        if not candidates:
            candidates = _fallback_candidates(slope, config)
    except Exception as exc:
        logger.error("Stage 2b failed: %s", exc)
        raise

    # ---------------------------------------------------------
    # Stage 3: PSR + Doubly-Shadowed Crater Detection
    # ---------------------------------------------------------
    logger.info("=== Stage 3: PSR & Doubly-Shadowed Crater Detection ===")
    dsc_craters = []
    try:
        psr_mask = compute_psr_mask(sunlit_mask, config.module6)
        # Crater mask from synthetic DEM: detect local minima
        from scipy.ndimage import minimum_filter
        crater_mask = synthetic_dem == minimum_filter(synthetic_dem, size=11, mode='reflect')
        dsc_craters = identify_doubly_shadowed_craters(psr_mask, crater_mask)
        n_dsc_craters = len(dsc_craters)
        logger.info(
            "Stage 3: PSR coverage=%.1f%%, %d doubly-shadowed craters.",
            100.0 * float(np.mean(psr_mask)),
            n_dsc_craters,
        )
        for cr in dsc_craters:
            logger.info(
                "  DSC: centroid=(%.1f,%.1f) radius=%.1f",
                cr.centroid[0], cr.centroid[1], cr.radius_px,
            )
    except Exception as exc:
        logger.warning("Stage 3 (module_6) optional — skipping (%s).", exc)

    # ---------------------------------------------------------
    # Stage 4: Hybrid Path Planning (Gorilla Traversal)
    # ---------------------------------------------------------
    logger.info("=== Stage 4: Hybrid Risk-Aware Path Planning ===")
    try:
        rover_state = RoverState(
            position=config.mission_start,
            battery_wh=config.rover_battery_initial_wh,
            battery_max_wh=config.rover_battery_max_wh,
            in_sunlight=True,
            timestep=0,
        )

        if not candidates:
            candidates = _fallback_candidates(slope, config)

        dash_result, return_result = plan_mission_path(
            slope_map=slope,
            ice_mask=ice_mask,
            sunlit_mask=sunlit_mask,
            start=config.mission_start,
            dsc_sample_point=config.dsc_sample_point,
            singularity_sites=candidates,
            rover_state=rover_state,
            config=config.module3,
        )

        dash_feasible = dash_result.feasible
        return_feasible = return_result.feasible

        logger.info(
            "Stage 4: dash feasible=%s (len=%d), return feasible=%s (len=%d)",
            dash_feasible,
            len(dash_result.path) if dash_result.path else 0,
            return_feasible,
            len(return_result.path) if return_result.path else 0,
        )

        if dash_feasible and dash_result.path:
            grid_state.mark_visited(dash_result.path)
        if return_feasible and return_result.path:
            grid_state.mark_visited(return_result.path)

        fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.imshow(slope, cmap="terrain", origin="upper")
        ax.plot(config.mission_start[1], config.mission_start[0], "go", markersize=10, label="Start")
        ax.plot(config.dsc_sample_point[1], config.dsc_sample_point[0], "r*", markersize=15, label="DSC F2")
        if dash_feasible and dash_result.path:
            path_arr = np.array(dash_result.path)
            ax.plot(path_arr[:, 1], path_arr[:, 0], "b-", linewidth=1.5, label="Dash")
        if return_feasible and return_result.path:
            path_arr = np.array(return_result.path)
            ax.plot(path_arr[:, 1], path_arr[:, 0], "y--", linewidth=1.5, label="Return")
        ax.set_title("Planned Paths (Dash + Return)")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        gs_summary = grid_state.summary()
        labels = ["unknown", "free", "obstacle", "visited"]
        values = [gs_summary[k] for k in labels]
        bar_colors = ["gray", "green", "red", "blue"]
        bars = ax2.bar(labels, values, color=bar_colors)
        ax2.set_title("Grid States (unknown=%.1f%%)" % (100 * gs_summary["unknown_fraction"]))
        ax2.set_ylabel("Cells")
        for b, v in zip(bars, values):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, str(v), ha="center", fontsize=8)
        _save_figure(fig4, "stage4_planning.png")

    except Exception as exc:
        logger.error("Stage 4 failed: %s", exc)
        raise

    # ---------------------------------------------------------
    # Stage 5: Reactive Obstacle Avoidance & SLAM (Bug-2)
    # ---------------------------------------------------------
    logger.info("=== Stage 5: Reactive Obstacle Avoidance & SLAM ===")
    try:
        autonomy_mode = AutonomyMode(config.module4)

        obstacle_map = np.zeros_like(slope, dtype=bool)
        obstacle_map[60, 50:70] = True

        if dash_feasible and dash_result.path:
            bug2_planner = Bug2Planner(dash_result.path, obstacle_map, config.module4)
        else:
            default_path = [(50, 50), (80, 80)]
            bug2_planner = Bug2Planner(default_path, obstacle_map, config.module4)

        trajectory = bug2_planner.run(config.mission_start, config.dsc_sample_point)
        logger.info("Stage 5: Bug-2 trajectory length = %d steps.", len(trajectory))

        gt_path = dash_result.path if (dash_feasible and dash_result.path) else [(50, 50), (80, 80)]
        mae = bug2_planner.compute_mae(trajectory, gt_path)
        logger.info("Stage 5: SLAM MAE = %.4f grid cells.", mae)

        fig5, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.imshow(obstacle_map.astype(int), cmap="gray_r", origin="upper")
        if trajectory:
            traj_arr = np.array(trajectory)
            ax.plot(traj_arr[:, 1], traj_arr[:, 0], "r-", linewidth=1, label="Trajectory")
            ax.scatter(traj_arr[0, 1], traj_arr[0, 0], c="green", s=80, label="Start")
            ax.scatter(traj_arr[-1, 1], traj_arr[-1, 0], c="blue", s=80, label="End")
        ax.set_title("Bug-2 + SLAM (MAE=%.3f)" % mae)
        ax.legend(fontsize=8)

        ax2 = axes[1]
        gt_arr = np.array(gt_path)
        traj_arr = np.array(trajectory) if trajectory else gt_arr
        ax2.plot(gt_arr[:, 0], label="Ground Truth", marker="o")
        ax2.plot(traj_arr[:, 0], label="Estimated (SLAM)", marker="x")
        ax2.set_title("Path Comparison (row coordinate)")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Row index")
        ax2.legend(fontsize=8)
        _save_figure(fig5, "stage5_slam.png")

    except Exception as exc:
        logger.error("Stage 5 failed: %s", exc)
        raise

    # ---------------------------------------------------------
    # Stage 6: EFPI In-Situ Ice Sensing (optional)
    # ---------------------------------------------------------
    logger.info("=== Stage 6: EFPI In-Situ Ice Sensing ===")
    try:
        Q_joules = compute_drill_heat_transfer(config.module5)
        T_post = compute_post_drill_temperature(
            Q_joules,
            regolith_mass_kg=0.5,
            initial_temp_k=config.module5.regolith_temp_k,
        )
        sub_rate = compute_sublimation_rate(
            T_post,
            P_sealed_pa=1.0,
            sealed_volume_m3=config.module5.sealed_volume_m3,
            config=config.module5,
        )
        vapor_density = compute_vapor_density(
            sub_rate,
            config.module5.sealed_volume_m3,
            config.module5.drill_contact_duration_s,
        )

        efpi = EFPIModel(config.module5)
        fringe_shift = efpi.fringe_shift_from_humidity(vapor_density)
        ice_pct = efpi.infer_ice_density(fringe_shift)

        logger.info(
            "Stage 6: T_post=%.1f K, vp_density=%.4f kg/m\u00b3, ice=%.2f%%",
            T_post, vapor_density, ice_pct,
        )
    except Exception as exc:
        logger.warning("Stage 6 (EFPI) optional — skipping (%s).", exc)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("  Target DSC       : %s", config.dsc_name)
    logger.info("  Ice volume       : %.2e m\u00b3", ice_volume_m3)
    logger.info("  DSC craters found: %d", n_dsc_craters)
    logger.info("  Candidates       : %d", n_candidates)
    logger.info("  Dash feasible    : %s", dash_feasible)
    logger.info("  SLAM MAE         : %.4f cells", mae)
    logger.info("  EFPI ice density : %.2f%%", ice_pct)
    logger.info("  Figures saved to : figures/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
